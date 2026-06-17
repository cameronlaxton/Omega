"""Single authoritative source for trace calibration/evidence eligibility.

Before this module the same predicate lived in three places that could drift:
``omega/core/contracts/service.py`` (computed ``calibration_eligible`` +
``calibration_exclusion_reasons`` at analyze time), ``PersistableTrace.
calibration_eligibility()`` (a read-side diagnostic), and the
``TraceStore.query_traces`` SQL filter. They are now all expressed here.

Two distinct concepts, deliberately separate (see docs/phase6/evidence-signals.md):

* **Probability-calibration eligibility** — can this trace's predicted
  probability feed the calibration learner? Requires a successful engine run
  with provided context, complete identity, no blocking downgrades, and a
  non-failed QA verdict. **Evidence is NOT required.**
* **Evidence-learning eligibility** — can this trace's structured evidence feed
  retrospective signal scoring? Requires present (or recovered) evidence and a
  non-failed QA verdict. This is the *only* path that cares about evidence.

The canonical persisted gate stays ``trace_quality.calibration_eligible`` inside
the full_trace JSON blob: ``service.py`` writes it via
:func:`calibration_exclusion_reasons` here, and every reader trusts it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# --- Probability-calibration status vocabulary (parallel to reason strings) ---
STATUS_ELIGIBLE = "eligible"
STATUS_PENDING_OUTCOME = "pending_outcome"
STATUS_INELIGIBLE_QA_FAILED = "ineligible_qa_failed"
STATUS_INELIGIBLE_MISSING_PREDICTION = "ineligible_missing_prediction"
STATUS_INELIGIBLE_MISSING_OUTCOME = "ineligible_missing_outcome"
STATUS_INELIGIBLE_INVALID_TRACE = "ineligible_invalid_trace"
STATUS_INELIGIBLE_TRACE_QUALITY = "ineligible_trace_quality"

# --- Evidence-learning status vocabulary ---
EV_ELIGIBLE_ORIGINAL = "eligible_original"
EV_ELIGIBLE_RECOVERED = "eligible_recovered_predecision"
EV_INELIGIBLE_EMPTY = "ineligible_empty_evidence"
EV_INELIGIBLE_UNRECOVERABLE = "ineligible_unrecoverable"
EV_INELIGIBLE_QA_FAILED = "ineligible_qa_failed"
EV_INELIGIBLE_INVALID = "ineligible_invalid_evidence"

# --- Exclusion-reason strings. These are the EXACT strings service.py emitted
# before centralization and are consumed by report_calibration.py / tests, so
# they must not be renamed. ---
REASON_ENGINE_SKIPPED = "engine_skipped"
REASON_BASELINE_CONTEXT = "baseline_default_context"
REASON_LEGACY_MISSING_CONTEXT = "legacy_missing_context_source"
REASON_LEGACY_MISSING_IDENTITY = "legacy_missing_identity"
# New, stable token appended when a trace-scoped QA verdict fails. The verdict
# table carries the human-readable scope/reason; this is the greppable flag that
# flips calibration_eligible off.
REASON_QA_FAILED = "qa_failed"


@dataclass(frozen=True)
class EligibilityResult:
    """Outcome of an eligibility check.

    ``status`` is drawn from the vocabularies above; ``reason`` is a short
    human-readable explanation for audit/logging.
    """

    eligible: bool
    status: str
    reason: str | None = None


def calibration_exclusion_reasons(
    *,
    result_status: str | None,
    context_source: str | None,
    baseline_used: bool,
    identity_status: str | None,
    result_downgrades: Iterable[str] = (),
    result_missing_requirements: Iterable[str] = (),
    caller_exclusion_reasons: Iterable[str] = (),
    qa_verdict: str | None = None,
) -> list[str]:
    """Canonical write-side computation of calibration exclusion reasons.

    This is the single definition of what makes a trace probability-calibration
    *ineligible*. ``service.py`` calls it to populate
    ``trace_quality.calibration_exclusion_reasons`` and derive
    ``calibration_eligible = not reasons``. Returned reasons are sorted+deduped,
    matching the historical service.py behavior exactly. ``qa_verdict == "fail"``
    appends :data:`REASON_QA_FAILED` so QA failures reconcile into the same flag.
    """
    reasons: list[str] = []
    if result_status != "success":
        reasons.append(REASON_ENGINE_SKIPPED)
    if context_source != "provided":
        reasons.append(
            REASON_BASELINE_CONTEXT if baseline_used else REASON_LEGACY_MISSING_CONTEXT
        )
    if identity_status != "complete":
        reasons.append(REASON_LEGACY_MISSING_IDENTITY)
    reasons.extend(str(r) for r in result_downgrades)
    reasons.extend(str(r) for r in result_missing_requirements)
    reasons.extend(str(r) for r in caller_exclusion_reasons)
    if qa_verdict == "fail":
        reasons.append(REASON_QA_FAILED)
    return sorted(set(reasons))


def probability_calibration_eligibility(
    *,
    predictions: object | None,
    trace_quality: dict | None,
    qa_verdict: str | None = None,
) -> EligibilityResult:
    """Read-side authoritative probability-calibration eligibility.

    Trusts the canonical ``trace_quality.calibration_eligible`` flag (which
    ``service.py`` computed via :func:`calibration_exclusion_reasons`) plus the
    one genuinely independent structural prerequisite — model predictions must
    exist — and a non-failed QA verdict. Evidence status is intentionally never
    consulted here.
    """
    tq = trace_quality or {}
    if qa_verdict == "fail":
        return EligibilityResult(
            False, STATUS_INELIGIBLE_QA_FAILED, "trace-scoped QA verdict failed"
        )
    if predictions is None:
        return EligibilityResult(
            False, STATUS_INELIGIBLE_MISSING_PREDICTION, "no model predictions to calibrate"
        )
    if not bool(tq.get("calibration_eligible")):
        return EligibilityResult(
            False, STATUS_INELIGIBLE_TRACE_QUALITY, "trace_quality.calibration_eligible is false"
        )
    if tq.get("context_source") != "provided":
        return EligibilityResult(
            False, STATUS_INELIGIBLE_TRACE_QUALITY, "context_source is not 'provided'"
        )
    if tq.get("identity_status") != "complete":
        return EligibilityResult(
            False, STATUS_INELIGIBLE_TRACE_QUALITY, "identity_status is not 'complete'"
        )
    return EligibilityResult(True, STATUS_ELIGIBLE)


def evidence_learning_eligibility(
    *,
    trace_quality: dict | None,
    qa_verdict: str | None = None,
) -> EligibilityResult:
    """Read-side authoritative evidence-learning eligibility.

    Requires a non-failed QA verdict and present (or recovered-pre-decision)
    evidence. Empty evidence blocks *only* evidence-signal learning — never
    probability calibration.
    """
    tq = trace_quality or {}
    if qa_verdict == "fail":
        return EligibilityResult(
            False, EV_INELIGIBLE_QA_FAILED, "trace-scoped QA verdict failed"
        )
    status = tq.get("evidence_status")
    if status == "present":
        return EligibilityResult(True, EV_ELIGIBLE_ORIGINAL)
    if status == "recovered_predecision":
        return EligibilityResult(True, EV_ELIGIBLE_RECOVERED)
    return EligibilityResult(
        False, EV_INELIGIBLE_EMPTY, "no evidence signals present on this trace"
    )
