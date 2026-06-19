"""P8.1 — backend parameter-profile variant sweep (the lab axis).

Verifies the sweep selects the better-calibrated variant on RAW validation
metrics, seals the holdout (only the winner is scored on it, once), reuses the
no-leak partition, and fails closed on bad inputs. A tiny registered bias-backend
stands in for a real sport backend: it reads ``prior_payload["home_bias"]`` and a
per-game base home rate, so a sweep over two bias values has a knowable winner.
"""

from __future__ import annotations

import pytest

from omega.core.simulation.backends import (
    GameSimulationInput,
    register_game_backend,
    resolve_game_backend,
)
from omega.core.simulation.parameter_profile import (
    BackendParameterProfile,
    make_parameter_profile_id,
)
from omega.strategy.artifacts import FrozenArtifact, compute_artifact_id
from omega.strategy.backtest.variant_sweep import sweep_backend_variants

_TEST_BACKEND = "p81_bias_test"
_DIST_ROW = {
    "target": "home_margin",
    "distribution_type": "empirical",
    "distribution_params": {},
    "params_schema_version": 10,
    "sample_mean": 0.0,
    "sample_std": 1.0,
    "p10": -1.0,
    "p50": 0.0,
    "p90": 1.0,
    "n_iterations": 1,
    "seed": 0,
    "context_hash": "x",
    "component_version": "p81_bias_test_v1",
}


class _BiasBackend:
    """Deterministic backend: home_win_prob = base_home_prob + prior_payload bias."""

    backend_name = _TEST_BACKEND
    component_version = "p81_bias_test_v1"
    evidence_mode = "plane_adjustment"

    def run(self, request: GameSimulationInput) -> dict:
        bias = float((request.prior_payload or {}).get("home_bias", 0.0))
        base = float((request.home_context or {}).get("base_home_prob", 50.0))
        hwp = max(1.0, min(99.0, base + bias))
        return {
            "success": True,
            "home_team": request.home_team,
            "away_team": request.away_team,
            "league": request.league,
            "iterations": request.n_iterations,
            "home_win_prob": hwp,
            "away_win_prob": 100.0 - hwp,
            "draw_prob": 0.0,
            "predicted_home_score": 1.0,
            "predicted_away_score": 1.0,
            "predicted_spread": 0.0,
            "predicted_total": 2.0,
            "context_source": "provided",
            "baseline_used": False,
            "simulation_distributions": [dict(_DIST_ROW)],
        }


if resolve_game_backend(_TEST_BACKEND) is None:
    register_game_backend(_TEST_BACKEND, _BiasBackend())


def _artifact(i: int, date: str, home_win: bool) -> FrozenArtifact:
    ht, at = f"H{i}", f"A{i}"
    return FrozenArtifact(
        artifact_id=compute_artifact_id(ht, at, "NBA", date),
        home_team=ht,
        away_team=at,
        league="NBA",
        date=date,
        home_context={"base_home_prob": 50.0},
        odds={"moneyline_home": -110, "moneyline_away": -110},
        outcome={"home_score": 2 if home_win else 0, "away_score": 0 if home_win else 2},
    )


def _artifacts() -> list[FrozenArtifact]:
    arts: list[FrozenArtifact] = []
    # 40 validation games (Jan–Feb), alternating outcomes -> 50% home rate.
    for i in range(40):
        month = 1 if i < 20 else 2
        day = (i % 20) + 1
        arts.append(_artifact(i, f"2026-{month:02d}-{day:02d}", home_win=(i % 2 == 0)))
    # 20 holdout games (March), alternating outcomes.
    for i in range(40, 60):
        arts.append(_artifact(i, f"2026-03-{(i - 40) + 1:02d}", home_win=(i % 2 == 0)))
    return arts


def _candidate(bias: float, version: int) -> BackendParameterProfile:
    params = {"home_bias": bias}
    return BackendParameterProfile(
        profile_id=make_parameter_profile_id(_TEST_BACKEND, "TEST", version, params),
        version=version,
        backend_name=_TEST_BACKEND,
        backend_component_version="p81_bias_test_v1",
        competition_bucket="TEST",
        params=params,
        dataset_hash="h",
        sample_size=0,
    )


_KW = {"validation_start": "2026-01-01", "holdout_start": "2026-03-01", "n_iterations": 1}


def test_winner_is_the_better_calibrated_variant():
    unbiased = _candidate(0.0, 1)  # predicts 50% home; actual 50% -> ~0 ECE
    biased = _candidate(30.0, 2)  # predicts 80% home; actual 50% -> ~0.30 ECE
    report = sweep_backend_variants(_artifacts(), [unbiased, biased], **_KW)

    assert report.winner_profile_id == unbiased.profile_id
    by_id = {s.profile_id: s for s in report.scores}
    assert by_id[unbiased.profile_id].validation.raw_ece < 0.05
    assert by_id[biased.profile_id].validation.raw_ece > 0.20
    assert by_id[unbiased.profile_id].validation.raw_ece < by_id[biased.profile_id].validation.raw_ece


def test_holdout_is_sealed_only_winner_scored():
    unbiased = _candidate(0.0, 1)
    biased = _candidate(30.0, 2)
    report = sweep_backend_variants(_artifacts(), [unbiased, biased], **_KW)
    by_id = {s.profile_id: s for s in report.scores}
    # Only the winner is scored on the sealed holdout.
    assert by_id[unbiased.profile_id].holdout is not None
    assert by_id[unbiased.profile_id].n_holdout == 20
    assert by_id[biased.profile_id].holdout is None
    assert by_id[biased.profile_id].n_holdout == 0


def test_partition_has_no_leak_and_counts_match():
    report = sweep_backend_variants(_artifacts(), [_candidate(0.0, 1)], **_KW)
    # 40 validation + 20 holdout = the full graded set; validation strictly before
    # the holdout cutoff (the internal hard-assert would fire otherwise).
    assert report.n_validation_events == 40
    assert report.n_holdout_events == 20
    assert report.holdout_sealed is True


def test_unregistered_backend_fails_closed():
    bogus = _candidate(0.0, 1).model_copy(update={"backend_name": "nope_not_registered"})
    with pytest.raises(ValueError, match="not registered"):
        sweep_backend_variants(_artifacts(), [bogus], **_KW)


def test_mixed_bucket_raises():
    a = _candidate(0.0, 1)
    b = _candidate(10.0, 2).model_copy(update={"competition_bucket": "OTHER"})
    with pytest.raises(ValueError, match="within one"):
        sweep_backend_variants(_artifacts(), [a, b], **_KW)


def test_empty_candidates_raises():
    with pytest.raises(ValueError, match="at least one candidate"):
        sweep_backend_variants(_artifacts(), [], **_KW)


def test_holdout_after_validation_required():
    with pytest.raises(ValueError, match="strictly after"):
        sweep_backend_variants(
            _artifacts(), [_candidate(0.0, 1)],
            validation_start="2026-03-01", holdout_start="2026-01-01", n_iterations=1,
        )
