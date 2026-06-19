"""The shared promotion-gate engine is profile-type-agnostic.

These tests pin the contract that P8.0.2 (BackendParameterProfile promotion)
depends on: ``evaluate_promotion_gates`` evaluates ANY object satisfying the
structural ``GateCandidate`` surface (``profile_id``, ``sample_size``,
``metrics``) — not just ``CalibrationProfile`` — using the identical gates,
thresholds, and fail-closed evidence discipline. ``metrics`` here are RAW
(pre-calibration) forecast-quality numbers, the way a backend parameter profile
reports them; the gate logic keys on metric *names*, so it is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omega.core.governance.promotion_gates import (
    PROMOTION_GATE_REPORT_SCHEMA_VERSION,
    PromotionGateError,
    evaluate_promotion_gates,
)


@dataclass
class _StubCandidate:
    """Minimal non-calibration GateCandidate (stands in for a parameter profile)."""

    profile_id: str
    sample_size: int = 200
    metrics: dict[str, Any] = field(default_factory=dict)


_RAW_OK = {"brier_score": 0.21, "calibration_error": 0.038, "log_loss": 0.62}
_CONFIRMS = {
    "confirm_backtest_parity": True,
    "confirm_clv_non_regression": True,
    "parity_evidence": {"state": "PASS"},
    "clv_evidence": {"verdict": "PASS"},
}


def test_non_calibration_candidate_passes_every_gate():
    report = evaluate_promotion_gates(
        _StubCandidate("soccer_bvp_dc__fifa_intl__v2", metrics=dict(_RAW_OK)),
        None,
        **_CONFIRMS,
    )
    assert report.passed
    assert report.to_dict()["schema_version"] == PROMOTION_GATE_REPORT_SCHEMA_VERSION
    names = {r.name for r in report.results}
    assert {"SAMPLE_SIZE", "ECE_FLOOR", "BRIER_IMPROVES", "LOG_LOSS_NO_REG"} <= names


def test_raw_ece_over_floor_blocks_a_parameter_profile():
    """A backend whose RAW held-out ECE is over the floor cannot be promoted —
    the whole point of the rail (structure must clear the floor, not calibration)."""
    cand = _StubCandidate(
        "soccer_bvp_dc__fifa_intl__v2",
        metrics={"brier_score": 0.21, "calibration_error": 0.11, "log_loss": 0.62},
    )
    report = evaluate_promotion_gates(cand, None, **_CONFIRMS)
    assert not report.passed
    assert "ECE_FLOOR" in report.failed_gates


def test_undersampled_parameter_profile_blocks():
    cand = _StubCandidate("nfl_nb__v2", sample_size=12, metrics=dict(_RAW_OK))
    report = evaluate_promotion_gates(cand, None, **_CONFIRMS)
    assert "SAMPLE_SIZE" in report.failed_gates


def test_parameter_profile_must_beat_incumbent_on_brier():
    incumbent = _StubCandidate("soccer_bvp_dc__fifa_intl__v1", metrics=dict(_RAW_OK))
    candidate = _StubCandidate(
        "soccer_bvp_dc__fifa_intl__v2", metrics=dict(_RAW_OK)
    )  # identical brier -> no improvement
    report = evaluate_promotion_gates(candidate, incumbent, **_CONFIRMS)
    assert "BRIER_IMPROVES" in report.failed_gates
    assert report.incumbent_id == "soccer_bvp_dc__fifa_intl__v1"


def test_bare_confirm_without_artifact_fails_closed_for_parameter_profile():
    cand = _StubCandidate("nfl_nb__v2", metrics=dict(_RAW_OK))
    report = evaluate_promotion_gates(
        cand, None, confirm_backtest_parity=True, confirm_clv_non_regression=True
    )
    assert "BACKTEST_PARITY" in report.failed_gates
    assert "CLV_NON_REG" in report.failed_gates


def test_calibration_module_reexports_shared_engine():
    """calibration.promotion is the calibration-facing view of the same engine."""
    from omega.core.calibration import promotion as calib_promotion

    assert calib_promotion.evaluate_promotion_gates is evaluate_promotion_gates
    assert calib_promotion.PromotionGateError is PromotionGateError
