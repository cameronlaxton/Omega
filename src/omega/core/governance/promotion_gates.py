"""Shared, fail-closed promotion-gate evaluation for versioned profiles.

One definition of "is this candidate safe to promote", composed by every domain
that promotes a versioned profile:

- calibration-profile promotion — ``CalibrationRegistry.promote()`` (enforcement)
  and the ``omega-promote-profile`` CLI (display);
- backend parameter-profile promotion — ``omega-promote-parameter-profile`` and
  the simulation parameter registry.

There is no ``--force`` path: callers always evaluate these gates, so a profile
can only reach PRODUCTION if every gate passes. Keeping the engine in one place
(here, neutral to both planes) is what prevents the gates, thresholds, and
evidence discipline from drifting apart between domains.

The engine is profile-type-agnostic: it reads only the structural
:class:`GateCandidate` surface (``profile_id``, ``sample_size``, ``metrics``), so
a ``CalibrationProfile`` and a ``BackendParameterProfile`` are evaluated by the
identical code. ``metrics`` is the calibration/forecast-quality block — for a
calibration profile it is the *post-calibration* held-out quality; for a backend
parameter profile it is the *raw* (pre-calibration) held-out quality. The gate
logic is the same either way because it keys on metric *names*, not on whether
the probability was calibrated.

Gates:
    SAMPLE_SIZE      candidate.sample_size >= min_samples
    ECE_FLOOR        candidate.calibration_error <= ece_floor (absolute quality floor)
    BRIER_IMPROVES   candidate.brier_score improves on incumbent by >= brier_improvement
    LOG_LOSS_NO_REG  candidate.log_loss does not regress vs incumbent by > log_loss_tol
    BACKTEST_PARITY  operator-confirmed + pass-indicating evidence artifact
    CLV_NON_REG      operator-confirmed + pass-indicating evidence artifact

When there is no incumbent (the first production profile for a key), the
improvement/no-regression gates auto-pass, but the SAMPLE_SIZE, ECE_FLOOR, and
operator-confirmation gates still apply.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

UTC = timezone.utc

DEFAULT_MIN_SAMPLES = 100
DEFAULT_BRIER_IMPROVEMENT = 0.01
DEFAULT_LOG_LOSS_TOL = 0.005
DEFAULT_ECE_FLOOR = 0.05
PROMOTION_GATE_REPORT_SCHEMA_VERSION = "promotion_gate_report.v1"


class GateCandidate(Protocol):
    """Structural surface a profile must expose to be evaluated by the gates.

    Satisfied by ``CalibrationProfile`` and ``BackendParameterProfile`` alike.
    ``metrics`` carries the held-out quality block (``brier_score``,
    ``calibration_error``, ``log_loss``, and optionally ``cv_calibration_error``
    / ``cv_n_folds``).
    """

    profile_id: str
    sample_size: int
    metrics: Mapping[str, Any]


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
    parity_evidence: dict[str, Any] | None = None
    clv_evidence: dict[str, Any] | None = None
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
                "backtest_parity_evidence": artifact_indicates_pass(self.parity_evidence)[1]
                if self.confirm_backtest_parity
                else None,
                "clv_evidence": artifact_indicates_pass(self.clv_evidence)[1]
                if self.confirm_clv_non_regression
                else None,
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


def artifact_indicates_pass(evidence: dict[str, Any] | None) -> tuple[bool, str]:
    """Whether a parity/CLV evidence artifact indicates a pass.

    Accepts the shapes emitted by the parity tools:
      * ``omega-historical-live-parity`` -> ``{"state": "PASS"|"INCONCLUSIVE"|"FAIL"}``
      * ``omega-backtest-parity`` -> ``{"recommend_promotion": true|false}``
      * a generic CLV/walk-forward report -> ``{"verdict": "PASS"}`` or
        ``{"non_regression": true}``.
    Returns ``(passed, human_detail)``. An empty/unrecognized artifact is NOT a
    pass — the gate fails closed rather than trusting an unverifiable confirm.
    """
    if not evidence:
        return False, "no artifact provided"
    state = str(evidence.get("state", "")).upper()
    if state:
        return state == "PASS", f"state={state}"
    if evidence.get("recommend_promotion") is True:
        return True, "recommend_promotion=true"
    if evidence.get("recommend_promotion") is False:
        return False, "recommend_promotion=false"
    verdict = str(evidence.get("verdict", "")).upper()
    if verdict:
        return verdict == "PASS", f"verdict={verdict}"
    if evidence.get("non_regression") is True:
        return True, "non_regression=true"
    return (
        False,
        "artifact has no recognized pass signal (state/recommend_promotion/verdict/non_regression)",
    )


def _evidence_gate(name: str, confirmed: bool, evidence: dict[str, Any] | None) -> GateResult:
    """Evaluate an evidence-backed operator gate (BACKTEST_PARITY / CLV_NON_REG).

    The operator confirmation flag is necessary but NOT sufficient: it must be
    backed by a referenced artifact that actually indicates a pass. A bare
    confirm with no artifact, or an artifact that is INCONCLUSIVE/FAIL, fails
    closed — this is the enforcement the audit flagged as missing.
    """
    if not confirmed:
        return GateResult(name, False, "not operator-confirmed")
    if not evidence:
        return GateResult(
            name,
            False,
            f"{name} confirmation requires a referenced parity/CLV artifact "
            "(--parity-report / --clv-report); a bare confirm flag is no longer sufficient",
        )
    ok, detail = artifact_indicates_pass(evidence)
    prefix = "artifact PASS: " if ok else "artifact does NOT indicate pass: "
    return GateResult(name, ok, prefix + detail)


def evaluate_promotion_gates(
    candidate: GateCandidate,
    incumbent: GateCandidate | None,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    brier_improvement: float = DEFAULT_BRIER_IMPROVEMENT,
    log_loss_tol: float = DEFAULT_LOG_LOSS_TOL,
    ece_floor: float = DEFAULT_ECE_FLOOR,
    confirm_backtest_parity: bool = False,
    confirm_clv_non_regression: bool = False,
    parity_evidence: dict[str, Any] | None = None,
    clv_evidence: dict[str, Any] | None = None,
) -> GateReport:
    """Evaluate every promotion gate and return a complete GateReport.

    Pure and side-effect free: callers decide what to do with ``report.passed``.

    ``parity_evidence`` / ``clv_evidence`` are the parsed parity/CLV report
    artifacts that back the operator confirmations. A confirmation without a
    pass-indicating artifact fails closed (see :func:`_evidence_gate`).
    """
    results: list[GateResult] = []

    # Gate 1: sample size.
    n = candidate.sample_size
    results.append(
        GateResult(
            "SAMPLE_SIZE", n >= min_samples, f"candidate.sample_size={n}, required>={min_samples}"
        )
    )

    # Gate 2: absolute ECE quality floor on the candidate itself.
    # Prefer the cross-validated ECE when the fitter recorded one (cv_n_folds>0):
    # a single 20% holdout ECE is high-variance + upward-biased at small n, so a
    # well-calibrated profile can fail this floor on split noise. The CV estimate
    # (fit-per-fold, scored out-of-sample, repeated) is the robust measure. Legacy
    # candidates without CV metrics fall back to the single-split calibration_error.
    cv_ece = candidate.metrics.get("cv_calibration_error")
    cv_n_folds = candidate.metrics.get("cv_n_folds", 0)
    use_cv = cv_ece is not None and cv_n_folds > 0
    if use_cv:
        ece = cv_ece
        src = "cv_calibration_error"
    elif cv_n_folds > 0:
        ece = None
        src = "cv_calibration_error"
    else:
        ece = candidate.metrics.get("calibration_error")
        src = "calibration_error"

    if ece is None:
        results.append(
            GateResult("ECE_FLOOR", False, f"candidate has no {src} metric; cannot verify floor")
        )
    else:
        results.append(
            GateResult(
                "ECE_FLOOR",
                ece <= ece_floor,
                f"candidate.{src}={ece:.4f}, required<={ece_floor:.4f}",
            )
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
            results.append(
                GateResult("BRIER_IMPROVES", True, "incumbent has no brier_score — auto-pass")
            )
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
            results.append(
                GateResult("LOG_LOSS_NO_REG", True, "incumbent has no log_loss — auto-pass")
            )
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

    # Gates 5/6: evidence-backed operator confirmations. The confirm flag is
    # necessary but not sufficient — it must be backed by a referenced artifact
    # that indicates a pass (see _evidence_gate).
    results.append(_evidence_gate("BACKTEST_PARITY", confirm_backtest_parity, parity_evidence))
    results.append(_evidence_gate("CLV_NON_REG", confirm_clv_non_regression, clv_evidence))

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
        parity_evidence=parity_evidence,
        clv_evidence=clv_evidence,
        incumbent_id=incumbent.profile_id if incumbent else None,
    )
