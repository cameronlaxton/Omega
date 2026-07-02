"""omega-fit-backend-structure --plane prop — the prop structural-tuning loop.

Mirrors tests/ops/test_fit_backend_structure.py on the prop seam: a synthetic
prop backend whose calibration error is minimized at a KNOWN ``nb_k_scale``
proves the driver selects that winner on raw prop OOS ECE, fills the gate's raw
CV-ECE from the sealed holdout only, produces a CANDIDATE that clears the 0.05
ECE_FLOOR, and refuses to promote on noise. A second, end-to-end test runs the
REAL ``prop_neg_binom`` backend (exact NB CDF) over outcomes generated at a
known true dispersion and recovers the known scale.
"""

from __future__ import annotations

import pytest

from omega.core.calibration.league_buckets import resolve_prop_calibration_bucket
from omega.core.governance.promotion_gates import evaluate_promotion_gates
from omega.core.simulation.backends import (
    PropSimulationInput,
    register_prop_backend,
    resolve_prop_backend,
)
from omega.ops.fit_backend_structure import (
    _prop_artifacts_from_traces,
    _register_candidate,
    tune_prop_backend_structure,
)
from omega.strategy.artifacts import FrozenPropArtifact, compute_prop_artifact_id
from omega.trace.store import TraceStore

_TUNE_BACKEND = "proptune_test"
_FLAT_BACKEND = "proptune_flat_test"
_BUCKET = "NFL__RUSHING_YARDS"


class _TunablePropBackend:
    """Overconfident by default; ``nb_k_scale=1.2`` makes it well-calibrated.

    over_prob = 0.5 + (base - 0.5) * 1.2 / w, where base is the artifact's true
    over-rate (rides ``projection_mean`` as a percentage). At w=1.2 the factor
    cancels -> prediction equals the empirical rate (ECE ~ 0); any other w
    over/under-sharpens. So a sweep selecting on raw prop ECE must pick w=1.2.
    """

    backend_name = _TUNE_BACKEND
    component_version = "proptune_test_v1"

    def run(self, request: PropSimulationInput) -> dict:
        prior = request.prior_payload or {}
        if prior.get("nb_dispersion_k") is None:
            raise ValueError("nb_dispersion_k required")
        w = float(prior.get("nb_k_scale", 1.0))
        base = float(request.projection_mean) / 100.0
        over = max(0.01, min(0.99, 0.5 + (base - 0.5) * 1.2 / w))
        return {"over_prob": over, "under_prob": 1.0 - over}


class _FlatPropBackend:
    """Overconfident and DEAF to the knob — every candidate ties (noise guard)."""

    backend_name = _FLAT_BACKEND
    component_version = "proptune_flat_test_v1"

    def run(self, request: PropSimulationInput) -> dict:
        base = float(request.projection_mean) / 100.0
        over = max(0.01, min(0.99, 0.5 + (base - 0.5) * 1.4))  # ignores prior_payload
        return {"over_prob": over, "under_prob": 1.0 - over}


for _name, _impl in ((_TUNE_BACKEND, _TunablePropBackend()), (_FLAT_BACKEND, _FlatPropBackend())):
    if resolve_prop_backend(_name) is None:
        register_prop_backend(_name, _impl)


def _prop_artifact(
    i: int, date: str, over_win: bool, base: float, line: float = 74.5, k: float = 25.0
) -> FrozenPropArtifact:
    player = f"P{i}"
    return FrozenPropArtifact(
        artifact_id=compute_prop_artifact_id(player, "NFL", "rushing_yards", line, date),
        player_name=player,
        league="NFL",
        stat_type="rushing_yards",
        line=line,
        date=date,
        projection_mean=base,
        nb_dispersion_k=k,
        simulation_seed=7,
        prop_outcomes=[{"side": "over", "result": "win" if over_win else "loss"}],
    )


def _group(base: float, n: int, date: str, start: int) -> list[FrozenPropArtifact]:
    """``n`` props at true over-rate ``base``% on ``date`` (exact via first-k wins)."""
    wins = round(base / 100.0 * n)
    return [_prop_artifact(start + i, date, over_win=(i < wins), base=base) for i in range(n)]


def _artifacts() -> list[FrozenPropArtifact]:
    # Same shape as the game driver test: validation (Jan/Feb) + holdout (Mar),
    # two extreme rates (90/10) at large n so the finite-sample CV noise floor
    # sits well below the 0.05 promotion floor.
    arts: list[FrozenPropArtifact] = []
    arts += _group(90, 300, "2026-01-15", 0)
    arts += _group(10, 300, "2026-02-15", 1000)
    arts += _group(90, 150, "2026-03-10", 2000)
    arts += _group(10, 150, "2026-03-20", 3000)
    return arts


_KW = dict(
    competition_bucket=_BUCKET,
    knob="nb_k_scale",
    grid=(0.6, 0.9, 1.0, 1.1, 1.2, 1.5),
    base_params={},
    validation_start="2026-01-01",
    holdout_start="2026-03-01",
    n_iterations=1,
)


def test_loop_selects_known_optimal_knob_and_fills_cv_ece():
    report, winner = tune_prop_backend_structure(_artifacts(), backend_name=_TUNE_BACKEND, **_KW)
    assert winner is not None, report.selection_note
    assert winner.params["nb_k_scale"] == 1.2
    assert winner.metrics["cv_n_folds"] > 0
    assert winner.metrics["cv_calibration_error"] <= 0.05
    assert winner.metrics["calibration_error"] <= 0.05  # sealed holdout single-split
    assert winner.sample_size == winner.metrics["n_eval"] > 0
    assert report.plane == "prop"


def test_cv_ece_uses_only_the_sealed_holdout(monkeypatch):
    import omega.ops.fit_backend_structure as fit_structure

    observed = {}

    def capture(predictions, outcomes, **_kwargs):
        observed["n"] = len(predictions)
        return {
            "cv_calibration_error": 0.0,
            "cv_ece_ci_low": 0.0,
            "cv_ece_ci_high": 0.0,
            "cv_n_folds": 5,
        }

    monkeypatch.setattr(fit_structure, "_raw_cv_ece", capture)
    tune_prop_backend_structure(_artifacts(), backend_name=_TUNE_BACKEND, **_KW)
    assert observed["n"] == 300  # 150+150 holdout pairs, never the 600 validation


def test_winner_passes_promotion_ece_floor():
    _report, winner = tune_prop_backend_structure(_artifacts(), backend_name=_TUNE_BACKEND, **_KW)
    pass_artifact = {"state": "PASS"}
    gate = evaluate_promotion_gates(
        winner,
        None,  # no incumbent -> improvement gates auto-pass
        min_samples=winner.sample_size,
        confirm_backtest_parity=True,
        confirm_clv_non_regression=True,
        parity_evidence=pass_artifact,
        clv_evidence=pass_artifact,
    )
    assert gate.passed, gate.failed_gates
    ece_gate = next(r for r in gate.results if r.name == "ECE_FLOOR")
    assert ece_gate.passed and "cv_calibration_error" in ece_gate.message


def test_knob_deaf_backend_ties_and_promotes_nothing():
    report, winner = tune_prop_backend_structure(_artifacts(), backend_name=_FLAT_BACKEND, **_KW)
    assert winner is None
    assert report.selection_inconclusive is True
    assert report.selection_note is not None


def test_unregistered_prop_backend_fails_closed():
    with pytest.raises(ValueError, match="not registered"):
        tune_prop_backend_structure(_artifacts(), backend_name="nope_not_registered", **_KW)


def test_global_nb_dispersion_k_in_base_params_rejected():
    """nb_dispersion_k is a per-decision prior frozen on each artifact; a global
    value would silently override them all — fail closed."""
    with pytest.raises(ValueError, match="nb_dispersion_k"):
        tune_prop_backend_structure(
            _artifacts(),
            backend_name=_TUNE_BACKEND,
            **{**_KW, "base_params": {"nb_dispersion_k": 30.0}},
        )


def test_real_nb_backend_recovers_known_dispersion():
    """End to end on the REAL prop_neg_binom backend (exact NB CDF): artifacts
    carry base k=50 while outcomes are generated at true k=25, so the sweep must
    recover nb_k_scale=0.5. Outcome fractions are set to the exact NB over-prob
    at the true k (first-k wins), so the test is deterministic and noise-free."""
    nb = resolve_prop_backend("prop_neg_binom")
    assert nb is not None
    mu, k_true, k_base = 80.0, 25.0, 50.0

    def true_over_prob(line: float) -> float:
        sim = nb.run(
            PropSimulationInput(
                player_name="truth",
                league="NFL",
                stat_type="rushing_yards",
                line=line,
                projection_mean=mu,
                n_iter=1,
                prior_payload={"nb_dispersion_k": k_true},
                exact=True,
            )
        )
        return sim["over_prob"]

    arts: list[FrozenPropArtifact] = []
    start = 0
    for line in (59.5, 69.5, 90.5, 100.5):
        p = true_over_prob(line)
        for n, date in ((100, "2026-01-15"), (100, "2026-02-15"), (50, "2026-03-10")):
            wins = round(p * n)
            arts += [
                _prop_artifact(start + i, date, over_win=(i < wins), base=mu, line=line, k=k_base)
                for i in range(n)
            ]
            start += n

    report, winner = tune_prop_backend_structure(
        arts,
        backend_name="prop_neg_binom",
        competition_bucket=_BUCKET,
        knob="nb_k_scale",
        grid=(0.5, 1.0, 2.0),
        base_params={},
        validation_start="2026-01-01",
        holdout_start="2026-03-01",
        n_iterations=1,
    )
    assert winner is not None, report.selection_note
    assert winner.params["nb_k_scale"] == 0.5
    assert winner.metrics["calibration_error"] <= 0.02  # exact recovery, rounding only


def test_prop_artifacts_from_traces_filters_and_dates():
    """The loader keys the no-leak split on decision_time, keeps only the
    requested canonical stat (aliases collapse), and drops non-prop traces and
    traces the builder rejects."""

    def prop_trace(stat: str, decision: str, mu=78.0, k=22.0):
        return {
            "trace_id": f"t-{stat}-{decision}",
            "kind": "prop",
            "timestamp": "2026-06-17T10:00:00",  # replay RUN date (batch), not the game
            "decision_time": decision,
            "league": "NFL",
            "input_snapshot": {"player_name": "P", "prop_type": stat, "line": 74.5},
            "simulation_distributions": [
                {"distribution_params": {"mu": mu, "k": k}, "n_iterations": 500, "seed": 1}
            ],
            "_prop_outcomes": [{"side": "over", "result": "win"}],
        }

    graded = [
        prop_trace("rushing_yards", "2025-11-09T18:00:00"),
        prop_trace("rush_yds", "2025-11-16T18:00:00"),  # alias -> same canonical stat
        prop_trace("receiving_yards", "2025-11-09T18:00:00"),  # other stat -> excluded
        {"kind": "game", "league": "NFL", "matchup": "A @ B"},  # not a prop -> skipped
        {  # builder-rejected: no NB params
            **prop_trace("rushing_yards", "2025-11-23T18:00:00"),
            "simulation_distributions": [{"distribution_params": {}}],
        },
    ]
    arts = _prop_artifacts_from_traces(graded, "NFL", "rushing_yards")
    assert len(arts) == 2
    assert {a.date for a in arts} == {"2025-11-09", "2025-11-16"}  # decision_time keyed
    assert all(a.stat_type == "rushing_yards" for a in arts)


def test_prop_bucket_resolver_naming():
    assert resolve_prop_calibration_bucket("NFL", "rushing_yards") == "NFL__RUSHING_YARDS"
    # Stat market-key alias collapses to the canonical bucket.
    assert resolve_prop_calibration_bucket("nfl", "pass_yds") == "NFL__PASSING_YARDS"
    # League alias collapses through the game bucket map (PREMIER_LEAGUE -> EPL).
    assert resolve_prop_calibration_bucket("premier_league", "shots") == "EPL__SHOTS"


def test_register_prop_candidate_allocates_bucket_version(tmp_path):
    from omega.ops.fit_backend_structure import build_candidates

    db = str(tmp_path / "traces.db")
    candidate = build_candidates(
        backend_name="prop_neg_binom",
        backend_component_version="prop_nb_v1",
        competition_bucket=_BUCKET,
        knob="nb_k_scale",
        grid=(1.0,),
        base_params={},
        dataset_hash="abc123",
    )[0]
    first = _register_candidate(db, candidate, candidate.backend_name, _BUCKET)
    second = _register_candidate(db, candidate, candidate.backend_name, _BUCKET)
    assert (first.version, second.version) == (1, 2)
    assert first.profile_id != second.profile_id
    TraceStore(db_path=db).close()
