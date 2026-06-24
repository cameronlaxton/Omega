"""omega-fit-backend-structure — the structural-tuning loop assembled end to end.

A synthetic backend whose calibration error is minimized at a KNOWN knob value
stands in for a real sport backend, so the sweep has a knowable winner. The tests
prove the driver: selects that winner on raw OOS ECE, fills the gate's raw CV-ECE,
produces a CANDIDATE that clears the 0.05 ECE_FLOOR, and refuses to promote on
noise (a backend the knob can't move ties -> no winner).
"""

from __future__ import annotations

from omega.core.governance.promotion_gates import evaluate_promotion_gates
from omega.core.simulation.backends import (
    GameSimulationInput,
    register_game_backend,
    resolve_game_backend,
)
from omega.ops.fit_backend_structure import (
    _register_candidate,
    build_candidates,
    tune_backend_structure,
)
from omega.strategy.artifacts import FrozenArtifact, compute_artifact_id
from omega.trace.store import TraceStore

_TUNE_BACKEND = "structtune_test"
_FLAT_BACKEND = "structtune_flat_test"
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
    "component_version": "structtune_test_v1",
}


def _success(request: GameSimulationInput, hwp: float) -> dict:
    hwp = max(1.0, min(99.0, hwp))
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


class _TunableBackend:
    """Overconfident by default; ``margin_sd_scale=1.2`` makes it well-calibrated.

    home_win_prob = 50 + (base-50) * 1.2 / w, where base is the artifact's true home
    rate. At w=1.2 the 1.2 factor cancels -> prediction equals the empirical rate
    (ECE ~ 0); any other w over/under-sharpens -> larger ECE. So a sweep that
    selects on raw ECE must pick w=1.2.
    """

    backend_name = _TUNE_BACKEND
    component_version = "structtune_test_v1"
    evidence_mode = "plane_adjustment"

    def run(self, request: GameSimulationInput) -> dict:
        prior = request.prior_payload or {}
        w = float(prior.get("margin_sd_scale", 1.0))
        base = float((request.home_context or {}).get("base", 50.0))
        return _success(request, 50.0 + (base - 50.0) * 1.2 / w)


class _FlatBackend:
    """Overconfident and DEAF to the knob — every candidate ties (noise guard)."""

    backend_name = _FLAT_BACKEND
    component_version = "structtune_flat_test_v1"
    evidence_mode = "plane_adjustment"

    def run(self, request: GameSimulationInput) -> dict:
        base = float((request.home_context or {}).get("base", 50.0))
        return _success(request, 50.0 + (base - 50.0) * 1.4)  # ignores prior_payload


for _name, _impl in ((_TUNE_BACKEND, _TunableBackend()), (_FLAT_BACKEND, _FlatBackend())):
    if resolve_game_backend(_name) is None:
        register_game_backend(_name, _impl)


def _group(base: int, n: int, date: str, start: int) -> list[FrozenArtifact]:
    """``n`` games at true home rate ``base``% on ``date`` (exact via first-k wins)."""
    wins = round(base / 100.0 * n)
    arts = []
    for i in range(n):
        ht, at = f"H{start + i}", f"A{start + i}"
        home_win = i < wins
        arts.append(
            FrozenArtifact(
                artifact_id=compute_artifact_id(ht, at, "NBA", date),
                home_team=ht,
                away_team=at,
                league="NBA",
                date=date,
                home_context={"base": base},
                odds={"moneyline_home": -110, "moneyline_away": -110},
                outcome={"home_score": 2 if home_win else 0, "away_score": 0 if home_win else 2},
            )
        )
    return arts


def _artifacts() -> list[FrozenArtifact]:
    arts: list[FrozenArtifact] = []
    # Validation (Jan/Feb) + holdout (Mar), each split across two calibrated rates
    # so the calibration map is non-trivial and CV has both classes in every fold.
    # Rates are deliberately extreme (90/10) and the sample large, so the
    # finite-sample CV binning-noise floor sits well below 0.05 (a moderate-n
    # bucket would sit AT the floor — exactly the regime the diagnostic flags).
    arts += _group(90, 300, "2026-01-15", 0)
    arts += _group(10, 300, "2026-02-15", 1000)
    arts += _group(90, 150, "2026-03-10", 2000)
    arts += _group(10, 150, "2026-03-20", 3000)
    return arts


_KW = dict(
    competition_bucket="TEST",
    knob="margin_sd_scale",
    grid=(0.85, 0.9, 1.0, 1.1, 1.2, 1.3),
    base_params={},
    validation_start="2026-01-01",
    holdout_start="2026-03-01",
    n_iterations=1,
)


def test_loop_selects_known_optimal_knob_and_fills_cv_ece():
    report, winner = tune_backend_structure(_artifacts(), backend_name=_TUNE_BACKEND, **_KW)
    assert winner is not None, report.selection_note
    assert winner.params["margin_sd_scale"] == 1.2
    # The gate's raw CV-ECE was computed (no-leak) and the candidate clears the floor.
    assert winner.metrics["cv_n_folds"] > 0
    assert winner.metrics["cv_calibration_error"] <= 0.05
    assert winner.metrics["calibration_error"] <= 0.05  # sealed holdout single-split
    assert winner.sample_size == winner.metrics["n_eval"] > 0


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
    tune_backend_structure(_artifacts(), backend_name=_TUNE_BACKEND, **_KW)
    assert observed["n"] == 300


def test_winner_passes_promotion_ece_floor():
    _report, winner = tune_backend_structure(_artifacts(), backend_name=_TUNE_BACKEND, **_KW)
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
    report, winner = tune_backend_structure(_artifacts(), backend_name=_FLAT_BACKEND, **_KW)
    assert winner is None
    assert report.selection_inconclusive is True
    assert report.selection_note is not None


def test_artifacts_from_traces_uses_decision_time_and_filters():
    """The loader keys the no-leak split on the match decision_time (not the replay
    run timestamp) and drops traces with no attached score."""
    from omega.ops.fit_backend_structure import _artifacts_from_traces

    base = {
        "matchup": "Away FC @ Home FC",
        "timestamp": "2026-06-17T10:00:00",  # replay RUN date (batch), not the match
        "execution_result": {"home_context": {"xg_for": 1.5}, "away_context": {"xg_for": 1.1}},
    }
    graded = [
        {
            **base,
            "decision_time": "2023-05-10T18:00:00",
            "_outcome": {"home_score": 2, "away_score": 1},
        },
        {
            **base,
            "decision_time": "2023-05-11T18:00:00",
            "_outcome": {"home_score": 0, "away_score": 0},
        },
        {**base, "decision_time": "2023-05-12", "outcome": None},  # no score -> dropped
    ]
    arts = _artifacts_from_traces(graded)
    assert len(arts) == 2
    assert arts[0].date == "2023-05-10"  # decision_time, NOT the 2026 run timestamp
    assert arts[0].artifact_id == compute_artifact_id("Home FC", "Away FC", "", "2023-05-10")
    assert arts[0].artifact_id != arts[1].artifact_id
    assert arts[0].home_team == "Home FC" and arts[0].away_team == "Away FC"
    assert arts[0].outcome["home_score"] == 2
    # No decision_time -> falls back to the trace timestamp date.
    fallback = _artifacts_from_traces([{**base, "_outcome": {"home_score": 1, "away_score": 0}}])
    assert fallback[0].date == "2026-06-17"


def test_build_candidates_one_per_grid_point_with_required_priors():
    cands = build_candidates(
        backend_name="soccer_bivariate_poisson_dc",
        backend_component_version="soccer_bvp_dc_v1",
        competition_bucket="FIFA_INTL",
        knob="lambda_gap_scale",
        grid=(0.7, 1.0),
        base_params={"rho": -0.12},
        dataset_hash="abc123",
    )
    assert len(cands) == 2
    assert all(c.params["rho"] == -0.12 for c in cands)  # required prior threaded through
    assert {c.params["lambda_gap_scale"] for c in cands} == {0.7, 1.0}
    assert len({c.profile_id for c in cands}) == 2  # ids are content-addressed, unique


def test_register_candidate_allocates_a_new_bucket_version(tmp_path):
    db = str(tmp_path / "traces.db")
    candidate = build_candidates(
        backend_name="soccer_bivariate_poisson_dc",
        backend_component_version="soccer_bvp_dc_v1",
        competition_bucket="FIFA_INTL",
        knob="lambda_gap_scale",
        grid=(1.0,),
        base_params={"rho": -0.12},
        dataset_hash="abc123",
    )[0]
    first = _register_candidate(db, candidate, candidate.backend_name, candidate.competition_bucket)
    second = _register_candidate(
        db, candidate, candidate.backend_name, candidate.competition_bucket
    )
    assert (first.version, second.version) == (1, 2)
    assert first.profile_id != second.profile_id
    TraceStore(db_path=db).close()
