"""Prop-plane backend parameter-profile variant sweep.

Mirrors tests/strategy/test_variant_sweep.py for the prop seam: the sweep
selects the better-calibrated variant on RAW prop validation metrics, seals the
holdout (only the winner is scored on it, once), reuses the no-leak partition,
grades through the single shared ``prop_pairs_for_trace`` rule (one-to-many
outcome rows, push/void excluded), and fails closed on bad inputs. A tiny
registered sharpness-backend stands in for a real prop backend: it reads
``prior_payload["nb_k_scale"]`` and a per-record base over-rate (carried in
``projection_mean``), so a sweep over knob values has a knowable winner.
"""

from __future__ import annotations

import pytest

from omega.core.simulation.backends import (
    PropSimulationInput,
    register_prop_backend,
    resolve_prop_backend,
)
from omega.core.simulation.parameter_profile import (
    BackendParameterProfile,
    make_parameter_profile_id,
)
from omega.strategy.artifacts import FrozenPropArtifact, compute_prop_artifact_id
from omega.strategy.backtest.variant_sweep import sweep_prop_backend_variants

_TEST_BACKEND = "prop_sharp_test"
_BUCKET = "NFL__RUSHING_YARDS"

_LAST_REQUEST: dict = {}


class _SharpnessPropBackend:
    """Deterministic prop backend: over_prob = 0.5 + (base - 0.5) * 1.2 / w.

    ``base`` is the record's true over-rate (rides ``projection_mean`` as a
    percentage); ``w = prior_payload["nb_k_scale"]``. At w=1.2 the prediction
    equals the empirical rate (ECE ~ 0); any other w over/under-sharpens.
    A ``skip_flagged`` prior raises for records flagged via ``projection_std``,
    exercising the ValueError-skip (absence) semantic and intersection scoring.
    """

    backend_name = _TEST_BACKEND
    component_version = "prop_sharp_test_v1"

    def run(self, request: PropSimulationInput) -> dict:
        _LAST_REQUEST["exact"] = request.exact
        prior = request.prior_payload or {}
        if prior.get("nb_dispersion_k") is None:
            raise ValueError("nb_dispersion_k required")
        if prior.get("skip_flagged") and (request.projection_std or 0) > 900:
            raise ValueError("skipped by test knob")
        w = float(prior.get("nb_k_scale", 1.0))
        base = float(request.projection_mean) / 100.0
        over = max(0.01, min(0.99, 0.5 + (base - 0.5) * 1.2 / w))
        return {"over_prob": over, "under_prob": 1.0 - over}


if resolve_prop_backend(_TEST_BACKEND) is None:
    register_prop_backend(_TEST_BACKEND, _SharpnessPropBackend())


def _prop_artifact(
    i: int,
    date: str,
    over_win: bool,
    base: float = 50.0,
    flagged: bool = False,
    outcomes: list[dict] | None = None,
) -> FrozenPropArtifact:
    player = f"P{i}"
    return FrozenPropArtifact(
        artifact_id=compute_prop_artifact_id(player, "NFL", "rushing_yards", 74.5, date),
        player_name=player,
        league="NFL",
        stat_type="rushing_yards",
        line=74.5,
        date=date,
        projection_mean=base,  # test backend reads this as the true over-rate (%)
        nb_dispersion_k=25.0,
        projection_std=999.0 if flagged else 15.0,
        simulation_seed=7,
        prop_outcomes=(
            outcomes
            if outcomes is not None
            else [{"side": "over", "result": "win" if over_win else "loss"}]
        ),
    )


def _group(base: float, n: int, date: str, start: int) -> list[FrozenPropArtifact]:
    """``n`` props at true over-rate ``base``% on ``date`` (exact via first-k wins)."""
    wins = round(base / 100.0 * n)
    return [_prop_artifact(start + i, date, over_win=(i < wins), base=base) for i in range(n)]


def _artifacts() -> list[FrozenPropArtifact]:
    # Validation (Jan/Feb) + holdout (Mar), split across two true over-rates so a
    # sharpness knob is visible to ECE (at a flat 50% rate every scale predicts
    # 0.5 and all candidates tie).
    arts: list[FrozenPropArtifact] = []
    arts += _group(90, 20, "2026-01-15", 0)
    arts += _group(10, 20, "2026-02-15", 100)
    arts += _group(90, 10, "2026-03-10", 200)
    arts += _group(10, 10, "2026-03-20", 300)
    return arts


def _candidate_params(params: dict, version: int) -> BackendParameterProfile:
    return BackendParameterProfile(
        profile_id=make_parameter_profile_id(_TEST_BACKEND, _BUCKET, version, params),
        version=version,
        backend_name=_TEST_BACKEND,
        backend_component_version="prop_sharp_test_v1",
        competition_bucket=_BUCKET,
        params=params,
        dataset_hash="h",
        sample_size=0,
    )


def _candidate(scale: float, version: int) -> BackendParameterProfile:
    return _candidate_params({"nb_k_scale": scale}, version)


_KW = {"validation_start": "2026-01-01", "holdout_start": "2026-03-01", "n_iterations": 1}


def test_winner_is_the_better_calibrated_variant():
    calibrated = _candidate(1.2, 1)  # predicts the empirical rate -> ~0 ECE
    sharp = _candidate(0.6, 2)  # doubles distance from 0.5 -> large ECE
    report = sweep_prop_backend_variants(_artifacts(), [calibrated, sharp], **_KW)

    assert report.plane == "prop"
    assert report.winner_profile_id == calibrated.profile_id
    by_id = {s.profile_id: s for s in report.scores}
    assert by_id[calibrated.profile_id].validation.raw_ece < 0.05
    assert (
        by_id[calibrated.profile_id].validation.raw_ece < by_id[sharp.profile_id].validation.raw_ece
    )


def test_holdout_is_sealed_only_winner_scored():
    calibrated = _candidate(1.2, 1)
    sharp = _candidate(0.6, 2)
    report = sweep_prop_backend_variants(_artifacts(), [calibrated, sharp], **_KW)
    by_id = {s.profile_id: s for s in report.scores}
    assert by_id[calibrated.profile_id].holdout is not None
    assert by_id[calibrated.profile_id].n_holdout == 20
    assert by_id[sharp.profile_id].holdout is None
    assert by_id[sharp.profile_id].n_holdout == 0


def test_partition_has_no_leak_and_counts_match():
    report = sweep_prop_backend_variants(_artifacts(), [_candidate(1.0, 1)], **_KW)
    assert report.n_validation_events == 40
    assert report.n_holdout_events == 20
    assert report.holdout_sealed is True


def test_exact_flag_reaches_the_prop_backend():
    sweep_prop_backend_variants(_artifacts(), [_candidate(1.0, 1)], **_KW)
    assert _LAST_REQUEST["exact"] is True  # default exact=True (NB honors, MC ignores)
    sweep_prop_backend_variants(_artifacts(), [_candidate(1.0, 1)], exact=False, **_KW)
    assert _LAST_REQUEST["exact"] is False


def test_unregistered_prop_backend_fails_closed():
    bogus = _candidate(1.0, 1).model_copy(update={"backend_name": "nope_not_registered"})
    with pytest.raises(ValueError, match="not registered"):
        sweep_prop_backend_variants(_artifacts(), [bogus], **_KW)


def test_mixed_bucket_raises():
    a = _candidate(1.0, 1)
    b = _candidate(1.2, 2).model_copy(update={"competition_bucket": "OTHER"})
    with pytest.raises(ValueError, match="within one"):
        sweep_prop_backend_variants(_artifacts(), [a, b], **_KW)


def test_empty_candidates_raises():
    with pytest.raises(ValueError, match="at least one candidate"):
        sweep_prop_backend_variants(_artifacts(), [], **_KW)


def test_holdout_after_validation_required():
    with pytest.raises(ValueError, match="strictly after"):
        sweep_prop_backend_variants(
            _artifacts(),
            [_candidate(1.0, 1)],
            validation_start="2026-03-01",
            holdout_start="2026-01-01",
            n_iterations=1,
        )


def test_intersection_scoring_equalizes_event_sets():
    """A variant whose sim raises on some records is scored on the SAME common set
    as one that prices everything — no winning on a smaller, easier subset."""
    arts: list[FrozenPropArtifact] = []
    for i in range(40):
        month = 1 if i < 20 else 2
        arts.append(
            _prop_artifact(
                i,
                f"2026-{month:02d}-{(i % 20) + 1:02d}",
                over_win=(i % 2 == 0),
                flagged=(i % 4 == 0),
            )
        )
    for i in range(40, 60):
        arts.append(_prop_artifact(i, f"2026-03-{(i - 40) + 1:02d}", over_win=(i % 2 == 0)))

    full = _candidate_params({"nb_k_scale": 1.0}, 1)
    skipper = _candidate_params({"nb_k_scale": 1.0, "skip_flagged": True}, 2)
    report = sweep_prop_backend_variants(arts, [full, skipper], **_KW)
    by_id = {s.profile_id: s for s in report.scores}
    # 10 of 40 validation props raise for the skipper -> common = 30 for BOTH.
    assert by_id[full.profile_id].n_validation == 30
    assert by_id[skipper.profile_id].n_validation == 30
    assert by_id[full.profile_id].n_simulated == 40
    assert by_id[skipper.profile_id].n_simulated == 30


def test_one_to_many_rows_each_pair_is_a_decision():
    """Every gradeable attached row is its own (prob, outcome) pair; push/void
    rows are excluded by the shared grading rule."""
    arts: list[FrozenPropArtifact] = []
    for i in range(30):
        month = 1 if i < 15 else 2
        arts.append(
            _prop_artifact(
                i,
                f"2026-{month:02d}-{(i % 15) + 1:02d}",
                over_win=True,
                outcomes=[
                    {"side": "over", "result": "win" if i % 2 == 0 else "loss"},
                    {"side": "under", "result": "loss" if i % 2 == 0 else "win"},
                    {"side": "over", "result": "push"},  # no-action, excluded
                ],
            )
        )
    arts.append(_prop_artifact(999, "2026-03-01", over_win=True))
    report = sweep_prop_backend_variants(arts, [_candidate(1.0, 1)], **_KW)
    # 30 validation records x 2 gradeable rows = 60 pairs on the common set.
    assert report.scores[0].n_validation == 60
    assert report.n_validation_events == 30


def test_tie_is_inconclusive_no_winner():
    a = _candidate_params({"nb_k_scale": 1.0}, 1)
    b = _candidate_params({"nb_k_scale": 1.0}, 2)  # identical params, different id
    report = sweep_prop_backend_variants(_artifacts(), [a, b], **_KW)
    assert report.winner_profile_id is None
    assert report.selection_inconclusive is True
    assert report.selection_note is not None


def test_single_candidate_wins_without_discrimination_guard():
    a = _candidate(1.0, 1)
    report = sweep_prop_backend_variants(_artifacts(), [a], **_KW)
    assert report.winner_profile_id == a.profile_id
    assert report.selection_inconclusive is False
    assert report.scores[0].holdout is not None


def test_all_ineligible_returns_no_winner_with_note():
    report = sweep_prop_backend_variants(
        _artifacts(), [_candidate(1.0, 1)], **{**_KW, "min_validation_samples": 1000}
    )
    assert report.winner_profile_id is None
    assert report.selection_inconclusive is False
    assert "min_validation_samples" in (report.selection_note or "")
