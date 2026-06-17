"""Shared, fail-closed promotion-gate evaluation for calibration profiles.

One definition of "is this candidate safe to promote", used by BOTH
``CalibrationRegistry.promote()`` (enforcement — raises on failure) and the
``omega-promote-profile`` CLI (display). There is no separate ``--force`` path:
the registry always evaluates these gates, so a profile can only reach
PRODUCTION through promotion if every gate passes.

Gates:
    SAMPLE_SIZE      candidate.sample_size >= min_samples
    ECE_FLOOR        candidate.calibration_error <= ece_floor (absolute quality floor)
    BRIER_IMPROVES   candidate.brier_score improves on incumbent by >= brier_improvement
    LOG_LOSS_NO_REG  candidate.log_loss does not regress vs incumbent by > log_loss_tol
    BACKTEST_PARITY  operator-confirmed (no automated check yet)
    CLV_NON_REG      operator-confirmed (no automated check yet)

When there is no incumbent (the first production profile for a
(league, market, context_slice)), the improvement/no-regression gates auto-pass,
but the SAMPLE_SIZE, ECE_FLOOR, and operator-confirmation gates still apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omega.core.calibration.profiles import CalibrationProfile

UTC = timezone.utc

DEFAULT_MIN_SAMPLES = 100
DEFAULT_BRIER_IMPROVEMENT = 0.01
DEFAULT_LOG_LOSS_TOL = 0.005
DEFAULT_ECE_FLOOR = 0.05
PROMOTION_GATE_REPORT_SCHEMA_VERSION = "promotion_gate_report.v1"


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    message: str


@dataclass
class GateReport:
    """The full set of gate outcomes for one promotion decision (audit trail)."""

    results: list[GateResult]
    thresholds: dict[str, float]
    confirm_backtest_parity: bool
    confirm_clv_non_regression: bool
    incumbent_id: str | None
    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_gates(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROMOTION_GATE_REPORT_SCHEMA_VERSION,
            "passed": self.passed,
            "evaluated_at": self.evaluated_at,
            "incumbent_id": self.incumbent_id,
            "thresholds": self.thresholds,
            "confirmations": {
                "backtest_parity": self.confirm_backtest_parity,
                "clv_non_regression": self.confirm_clv_non_regression,
            },
            "gates": [
                {"name": r.name, "passed": r.passed, "message": r.message} for r in self.results
            ],
        }


class PromotionGateError(RuntimeError):
    """Raised when a candidate fails one or more promotion gates."""

    def __init__(self, report: GateReport) -> None:
        self.report = report
        super().__init__("promotion blocked; failed gates: " + ", ".join(report.failed_gates))


def evaluate_promotion_gates(
    candidate: CalibrationProfile,
    incumbent: CalibrationProfile | None,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    brier_improvement: float = DEFAULT_BRIER_IMPROVEMENT,
    log_loss_tol: float = DEFAULT_LOG_LOSS_TOL,
    ece_floor: float = DEFAULT_ECE_FLOOR,
    confirm_backtest_parity: bool = False,
    confirm_clv_non_regression: bool = False,
) -> GateReport:
    """Evaluate every promotion gate and return a complete GateReport.

    Pure and side-effect free: callers decide what to do with ``report.passed``.
    """
    results: list[GateResult] = []

    # Gate 1: sample size.
    n = candidate.sample_size
    results.append(
        GateResult("SAMPLE_SIZE", n >= min_samples, f"candidate.sample_size={n}, required>={min_samples}")
    )

    # Gate 2: absolute ECE quality floor on the candidate itself.
    # Prefer the cross-validated ECE when the fitter recorded one (cv_n_folds>0):
    # a single 20% holdout ECE is high-variance + upward-biased at small n, so a
    # well-calibrated profile can fail this floor on split noise. The CV estimate
    # (fit-per-fold, scored out-of-sample, repeated) is the robust measure. Legacy
    # candidates without CV metrics fall back to the single-split calibration_error.
    cv_ece = candidate.metrics.get("cv_calibration_error")
    use_cv = cv_ece is not None and candidate.metrics.get("cv_n_folds", 0) > 0
    ece = cv_ece if use_cv else candidate.metrics.get("calibration_error")
    src = "cv_calibration_error" if use_cv else "calibration_error"
    if ece is None:
        results.append(
            GateResult("ECE_FLOOR", False, "candidate has no calibration_error metric; cannot verify floor")
        )
    else:
        results.append(
            GateResult("ECE_FLOOR", ece <= ece_floor, f"candidate.{src}={ece:.4f}, required<={ece_floor:.4f}")
        )

    # Gate 3: Brier improvement vs incumbent.
    cand_brier = candidate.metrics.get("brier_score")
    if incumbent is None:
        results.append(GateResult("BRIER_IMPROVES", True, "no incumbent — auto-pass"))
    elif cand_brier is None:
        results.append(GateResult("BRIER_IMPROVES", False, "candidate has no brier_score metric"))
    else:
        inc_brier = incumbent.metrics.get("brier_score")
        if inc_brier is None:
            results.append(GateResult("BRIER_IMPROVES", True, "incumbent has no brier_score — auto-pass"))
        else:
            improvement = inc_brier - cand_brier
            results.append(
                GateResult(
                    "BRIER_IMPROVES",
                    improvement >= brier_improvement,
                    f"improvement={improvement:.4f}, required>={brier_improvement:.4f} "
                    f"(candidate={cand_brier:.4f}, incumbent={inc_brier:.4f})",
                )
            )

    # Gate 4: log-loss no-regression vs incumbent.
    cand_log = candidate.metrics.get("log_loss")
    if incumbent is None:
        results.append(GateResult("LOG_LOSS_NO_REG", True, "no incumbent — auto-pass"))
    elif cand_log is None:
        results.append(GateResult("LOG_LOSS_NO_REG", False, "candidate has no log_loss metric"))
    else:
        inc_log = incumbent.metrics.get("log_loss")
        if inc_log is None:
            results.append(GateResult("LOG_LOSS_NO_REG", True, "incumbent has no log_loss — auto-pass"))
        else:
            regression = cand_log - inc_log
            results.append(
                GateResult(
                    "LOG_LOSS_NO_REG",
                    regression <= log_loss_tol,
                    f"delta={regression:+.4f}, tolerated<={log_loss_tol:.4f} "
                    f"(candidate={cand_log:.4f}, incumbent={inc_log:.4f})",
                )
            )

    # Gates 5/6: operator confirmations (no automated check yet).
    results.append(
        GateResult(
            "BACKTEST_PARITY",
            confirm_backtest_parity,
            "operator-confirmed" if confirm_backtest_parity else "no automated check — confirm after backtest review",
        )
    )
    results.append(
        GateResult(
            "CLV_NON_REG",
            confirm_clv_non_regression,
            "operator-confirmed" if confirm_clv_non_regression else "no automated check — confirm after CLV review",
        )
    )

    return GateReport(
        results=results,
        thresholds={
            "min_samples": float(min_samples),
            "brier_improvement": brier_improvement,
            "log_loss_tol": log_loss_tol,
            "ece_floor": ece_floor,
        },
        confirm_backtest_parity=confirm_backtest_parity,
        confirm_clv_non_regression=confirm_clv_non_regression,
        incumbent_id=incumbent.profile_id if incumbent else None,
    )
