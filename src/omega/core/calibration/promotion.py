"""Fail-closed promotion gates for calibration profiles.

The gate engine itself now lives in :mod:`omega.core.governance.promotion_gates`
so calibration-profile promotion and backend parameter-profile promotion share
one definition of "is this candidate safe to promote" and cannot drift apart.
This module is the calibration-facing view of that shared engine: it re-exports
the gate primitives unchanged, so every existing call site
(``CalibrationRegistry.promote()`` enforcement and the ``omega-promote-profile``
CLI display) keeps the identical fail-closed behavior. There is no ``--force``
bypass — the registry always evaluates these gates, so a profile can only reach
PRODUCTION if every gate passes.

Gates (see :mod:`omega.core.governance.promotion_gates` for the authoritative
definitions):
    SAMPLE_SIZE      candidate.sample_size >= min_samples
    ECE_FLOOR        candidate.calibration_error <= ece_floor (absolute quality floor)
    BRIER_IMPROVES   candidate.brier_score improves on incumbent by >= brier_improvement
    LOG_LOSS_NO_REG  candidate.log_loss does not regress vs incumbent by > log_loss_tol
    BACKTEST_PARITY  operator-confirmed + pass-indicating evidence artifact
    CLV_NON_REG      operator-confirmed + pass-indicating evidence artifact
"""

from __future__ import annotations

from omega.core.governance.promotion_gates import (
    DEFAULT_BRIER_IMPROVEMENT,
    DEFAULT_ECE_FLOOR,
    DEFAULT_LOG_LOSS_TOL,
    DEFAULT_MIN_SAMPLES,
    PROMOTION_GATE_REPORT_SCHEMA_VERSION,
    GateReport,
    GateResult,
    PromotionGateError,
    artifact_indicates_pass,
    evaluate_promotion_gates,
)

__all__ = [
    "DEFAULT_BRIER_IMPROVEMENT",
    "DEFAULT_ECE_FLOOR",
    "DEFAULT_LOG_LOSS_TOL",
    "DEFAULT_MIN_SAMPLES",
    "PROMOTION_GATE_REPORT_SCHEMA_VERSION",
    "GateReport",
    "GateResult",
    "PromotionGateError",
    "artifact_indicates_pass",
    "evaluate_promotion_gates",
]
