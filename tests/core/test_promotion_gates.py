"""Fail-closed promotion gates: CalibrationRegistry.promote() must refuse to mint
a PRODUCTION profile unless every gate passes. There is no --force bypass."""

from __future__ import annotations

from pathlib import Path

import pytest

from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus
from omega.core.calibration.promotion import (
    PROMOTION_GATE_REPORT_SCHEMA_VERSION,
    PromotionGateError,
    evaluate_promotion_gates,
)
from omega.core.calibration.registry import CalibrationRegistry


def _profile(profile_id: str, **overrides) -> CalibrationProfile:
    defaults = {
        "profile_id": profile_id,
        "version": 1,
        "method": "shrinkage",
        "league": "NBA",
        "params": {"shrink_factor": 0.6},
        "training_window": "2025-01-01/2025-06-30",
        "sample_size": 200,
        "dataset_hash": "abc123",
        "metrics": {"brier_score": 0.22, "calibration_error": 0.04, "log_loss": 0.65},
    }
    defaults.update(overrides)
    return CalibrationProfile(**defaults)


def _registry(tmp_path) -> CalibrationRegistry:
    return CalibrationRegistry(path=str(tmp_path / "profiles.json"))


_CONFIRMS = {"confirm_backtest_parity": True, "confirm_clv_non_regression": True}


def test_clean_first_promotion_succeeds_and_records_report(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_profile("nba_v1"))
    report = reg.promote("nba_v1", **_CONFIRMS)
    assert report.passed
    prof = reg.get_profile("nba_v1")
    assert prof.status == ProfileStatus.PRODUCTION
    assert prof.promotion_gate_report is not None
    assert prof.promotion_gate_report["schema_version"] == PROMOTION_GATE_REPORT_SCHEMA_VERSION
    assert prof.promotion_gate_report["passed"] is True
    assert prof.promotion_gate_report["confirmations"]["backtest_parity"] is True


def test_missing_operator_confirmations_blocks(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_profile("nba_v1"))
    with pytest.raises(PromotionGateError) as exc:
        reg.promote("nba_v1")  # no confirmations
    assert "BACKTEST_PARITY" in exc.value.report.failed_gates
    assert "CLV_NON_REG" in exc.value.report.failed_gates
    assert reg.get_profile("nba_v1").status == ProfileStatus.CANDIDATE  # unchanged


def test_undersampled_candidate_blocked(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_profile("nba_v1", sample_size=10))
    with pytest.raises(PromotionGateError) as exc:
        reg.promote("nba_v1", **_CONFIRMS)
    assert "SAMPLE_SIZE" in exc.value.report.failed_gates
    assert reg.get_profile("nba_v1").status == ProfileStatus.CANDIDATE


def test_high_ece_candidate_blocked(tmp_path):
    reg = _registry(tmp_path)
    reg.register(
        _profile("nba_v1", metrics={"brier_score": 0.22, "calibration_error": 0.15, "log_loss": 0.65})
    )
    with pytest.raises(PromotionGateError) as exc:
        reg.promote("nba_v1", **_CONFIRMS)
    assert "ECE_FLOOR" in exc.value.report.failed_gates


def test_missing_ece_metric_blocked(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_profile("nba_v1", metrics={"brier_score": 0.22, "log_loss": 0.65}))
    with pytest.raises(PromotionGateError) as exc:
        reg.promote("nba_v1", **_CONFIRMS)
    assert "ECE_FLOOR" in exc.value.report.failed_gates


def test_cv_ece_preferred_over_single_split_when_present(tmp_path):
    """A single-split ECE over the floor must NOT block when the robust CV ECE is
    under it — this is the whole point of CV-ECE: don't reject on split noise."""
    reg = _registry(tmp_path)
    reg.register(
        _profile(
            "nba_v1",
            metrics={
                "brier_score": 0.22,
                "calibration_error": 0.0548,  # unlucky single split, over floor
                "cv_calibration_error": 0.046,  # robust CV estimate, under floor
                "cv_n_folds": 50,
                "log_loss": 0.65,
            },
        )
    )
    report = reg.promote("nba_v1", **_CONFIRMS)
    assert report.passed
    ece_gate = next(r for r in report.results if r.name == "ECE_FLOOR")
    assert ece_gate.passed
    assert "cv_calibration_error" in ece_gate.message


def test_cv_ece_over_floor_blocks_even_if_single_split_passes(tmp_path):
    """Conversely, a lucky single split under the floor must not promote a model
    whose robust CV ECE is genuinely over it."""
    reg = _registry(tmp_path)
    reg.register(
        _profile(
            "nba_v1",
            metrics={
                "brier_score": 0.22,
                "calibration_error": 0.041,  # lucky single split, under floor
                "cv_calibration_error": 0.096,  # genuine miscalibration
                "cv_n_folds": 50,
                "log_loss": 0.65,
            },
        )
    )
    with pytest.raises(PromotionGateError) as exc:
        reg.promote("nba_v1", **_CONFIRMS)
    assert "ECE_FLOOR" in exc.value.report.failed_gates


def test_zero_cv_folds_falls_back_to_single_split(tmp_path):
    """cv_n_folds==0 (too few samples to CV) → fall back to single-split metric."""
    reg = _registry(tmp_path)
    reg.register(
        _profile(
            "nba_v1",
            metrics={
                "brier_score": 0.22,
                "calibration_error": 0.04,
                "cv_calibration_error": 0.0,
                "cv_n_folds": 0,
                "log_loss": 0.65,
            },
        )
    )
    report = reg.promote("nba_v1", **_CONFIRMS)
    ece_gate = next(r for r in report.results if r.name == "ECE_FLOOR")
    assert ece_gate.passed
    assert "calibration_error=" in ece_gate.message  # used the single-split metric


def test_second_promotion_requires_brier_improvement(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_profile("nba_v1"))
    reg.promote("nba_v1", **_CONFIRMS)
    reg.register(_profile("nba_v2", version=2))  # identical brier -> no improvement
    with pytest.raises(PromotionGateError) as exc:
        reg.promote("nba_v2", **_CONFIRMS)
    assert "BRIER_IMPROVES" in exc.value.report.failed_gates
    assert reg.get_profile("nba_v1").status == ProfileStatus.PRODUCTION  # incumbent kept
    assert reg.get_profile("nba_v2").status == ProfileStatus.CANDIDATE


def test_improving_second_promotion_archives_incumbent(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_profile("nba_v1"))
    reg.promote("nba_v1", **_CONFIRMS)
    reg.register(
        _profile(
            "nba_v2",
            version=2,
            metrics={"brier_score": 0.20, "calibration_error": 0.03, "log_loss": 0.64},
        )
    )
    report = reg.promote("nba_v2", **_CONFIRMS)
    assert report.passed
    assert report.incumbent_id == "nba_v1"
    assert reg.get_profile("nba_v1").status == ProfileStatus.ARCHIVED
    assert reg.get_profile("nba_v2").status == ProfileStatus.PRODUCTION


def test_cli_has_no_force_bypass():
    # The bypass must be gone as an actual flag/usage (the docstring may still
    # mention that there is no --force).
    src = (
        Path(__file__).resolve().parents[2] / "src" / "omega" / "ops" / "promote_profile.py"
    ).read_text(encoding="utf-8")
    assert 'add_argument("--force"' not in src
    assert "args.force" not in src


def test_evaluate_gates_is_pure_and_complete():
    report = evaluate_promotion_gates(
        _profile("c"), None, confirm_backtest_parity=True, confirm_clv_non_regression=True
    )
    assert report.passed
    names = {r.name for r in report.results}
    assert {
        "SAMPLE_SIZE",
        "ECE_FLOOR",
        "BRIER_IMPROVES",
        "LOG_LOSS_NO_REG",
        "BACKTEST_PARITY",
        "CLV_NON_REG",
    } <= names
