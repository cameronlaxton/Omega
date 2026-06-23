"""Graded trace quality — the honest answer to "how much should we trust this trace?".

``omega.trace.eligibility`` answers the *boolean* questions (is this trace
probability-calibration eligible? evidence-learning eligible?). This module
answers the *graded* one: a 0-100 ``aggregate_quality`` with a band, the reasons
behind it, the three learning weights it implies, and the confidence cap it
forces on any recommendation built from the trace.

Before this module ``trace_quality.aggregate_quality`` was a hardcoded ``None``
and "no evidence" was treated as harmless. Both were dishonest. Here:

* zero evidence + empty/baseline context scores in the floor band, zeroes both
  learning weights, and caps confidence at ``Pass`` (no actionable output);
* empty evidence + *provided* context still calibrates probabilities but cannot
  learn from evidence and is capped at ``C``;
* a ``static_identity`` calibration path (no real profile applied) caps at B/C;
* a failed QA verdict zeroes every learning weight and invalidates the trace.

Everything here is a pure function of its keyword inputs, so the same trace
always yields the same quality — a hard requirement for reproducible traces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

# --- Confidence-cap tiers (most → least restrictive). ``None`` == no cap. ------
_TIER_RANK: dict[str, int] = {"Pass": 0, "C": 1, "B": 2, "A": 3}

# --- Quality-reason tokens (greppable, stable). --------------------------------
REASON_ZERO_EVIDENCE_EMPTY_CONTEXT = "zero_evidence_empty_context"
REASON_QA_FAILED = "qa_failed"
REASON_EMPTY_EVIDENCE_PROVIDED_CONTEXT = "empty_evidence_provided_context"
REASON_STATIC_IDENTITY_CALIBRATION = "static_identity_calibration"
REASON_BASELINE_CONTEXT = "baseline_context"
REASON_MISSING_IDENTITY = "missing_identity"
REASON_HIGH_IMPUTATION = "high_imputation"
REASON_NOT_CALIBRATION_ELIGIBLE = "not_calibration_eligible"

# --- Band thresholds (inclusive lower bounds on aggregate_quality). ------------
BAND_STRONG_MIN = 75
BAND_USABLE_MIN = 50
BAND_WEAK_MIN = 20

# Modes in which evidence is recorded but never learned from.
_NON_LEARNING_MODES = frozenset({"disabled", "observe"})


@dataclass(frozen=True)
class TraceQuality:
    """Graded trust for one trace. ``aggregate_quality`` is never ``None``."""

    aggregate_quality: int  # 0-100
    quality_band: str  # strong | usable | weak | invalid
    quality_reasons: list[str] = field(default_factory=list)
    trace_weight: float = 0.0
    probability_calibration_weight: float = 0.0
    evidence_learning_weight: float = 0.0
    confidence_cap: str | None = None  # "A"|"B"|"C"|"Pass"; None = no cap

    def as_dict(self) -> dict:
        """Flat dict for merging into the persisted ``trace_quality`` block."""
        return {
            "aggregate_quality": self.aggregate_quality,
            "quality_band": self.quality_band,
            "quality_reasons": list(self.quality_reasons),
            "trace_weight": self.trace_weight,
            "probability_calibration_weight": self.probability_calibration_weight,
            "evidence_learning_weight": self.evidence_learning_weight,
            "confidence_cap": self.confidence_cap,
        }


def _most_restrictive(caps: Iterable[str | None]) -> str | None:
    """Return the lowest-tier (most restrictive) cap, or None if no real cap."""
    real = [c for c in caps if c]
    if not real:
        return None
    return min(real, key=lambda c: _TIER_RANK.get(c, 3))


def is_zero_evidence_empty_context(*, evidence_count: int, context_source: str | None) -> bool:
    """The dishonest-by-omission case: no evidence AND no real context.

    This is the trace shape that used to sail through as harmless. It cannot
    calibrate (context isn't 'provided'), cannot learn (no evidence), and must
    never produce actionable output.
    """
    return evidence_count <= 0 and context_source != "provided"


def aggregate_quality(
    *,
    evidence_status: str | None,
    evidence_count: int,
    context_source: str | None,
    baseline_used: bool,
    identity_status: str | None,
    calibration_eligible: bool,
    calibration_path: str | None,
    qa_verdict: str | None = None,
    imputed_fraction: float | None = None,
    evidence_mode: str | None = None,
) -> TraceQuality:
    """Compute the graded :class:`TraceQuality` for one trace.

    Deterministic additive score (max 100) followed by hard-rule overrides that
    encode the honesty contract. ``aggregate_quality`` is always an ``int``.
    """
    reasons: list[str] = []
    qa_failed = qa_verdict == "fail"
    empty_evidence = evidence_count <= 0
    provided = context_source == "provided"
    zero_ev_empty = is_zero_evidence_empty_context(
        evidence_count=evidence_count, context_source=context_source
    )

    # --- Additive component score (pre-override). ---
    score = 0
    if provided:
        score += 35
    elif baseline_used:
        score += 5
        reasons.append(REASON_BASELINE_CONTEXT)
    if identity_status == "complete":
        score += 20
    else:
        reasons.append(REASON_MISSING_IDENTITY)
    if not empty_evidence or evidence_status == "present":
        score += 20
    elif evidence_status == "recovered_predecision":
        score += 10
    score += {
        "profile": 15,
        "base_profile_fallback": 10,
        "static_calibrated": 5,
        "static_identity": 0,
    }.get(calibration_path or "", 0)
    if not qa_failed:
        score += 10
    frac = max(0.0, min(1.0, imputed_fraction or 0.0))
    if frac > 0.0:
        score -= round(15 * frac)
        if frac > 0.2:
            reasons.append(REASON_HIGH_IMPUTATION)

    # --- Hard-rule overrides on the aggregate. ---
    aggregate = score
    if qa_failed:
        aggregate = min(aggregate, 15)
        reasons.append(REASON_QA_FAILED)
    if zero_ev_empty:
        aggregate = min(aggregate, 20)
        reasons.append(REASON_ZERO_EVIDENCE_EMPTY_CONTEXT)
    if empty_evidence and provided:
        reasons.append(REASON_EMPTY_EVIDENCE_PROVIDED_CONTEXT)
    if calibration_path == "static_identity":
        reasons.append(REASON_STATIC_IDENTITY_CALIBRATION)
    if not calibration_eligible:
        reasons.append(REASON_NOT_CALIBRATION_ELIGIBLE)
    aggregate = int(max(0, min(100, aggregate)))

    # --- Band. ---
    if qa_failed:
        band = "invalid"
    elif aggregate >= BAND_STRONG_MIN:
        band = "strong"
    elif aggregate >= BAND_USABLE_MIN:
        band = "usable"
    elif aggregate >= BAND_WEAK_MIN:
        band = "weak"
    else:
        band = "invalid"

    # --- Weights. ---
    trace_weight = 0.0 if qa_failed else round(aggregate / 100.0, 4)

    if qa_failed or zero_ev_empty or not calibration_eligible:
        prob_cal_weight = 0.0
    else:
        prob_cal_weight = trace_weight

    if qa_failed or empty_evidence or (evidence_mode in _NON_LEARNING_MODES):
        ev_learn_weight = 0.0
    else:
        ev_learn_weight = trace_weight

    # --- Confidence cap (most restrictive applicable). ---
    # The band itself ceilings confidence: only a 'strong' trace may reach A; a
    # 'usable' trace caps at B; 'weak' at C; 'invalid' forces Pass. This is how
    # "A requires aggregate_quality >= 75" is enforced for every recommendation
    # without threading the score into per-edge confidence assignment.
    band_cap = {"strong": None, "usable": "B", "weak": "C", "invalid": "Pass"}[band]
    caps: list[str | None] = [band_cap]
    if qa_failed:
        caps.append("Pass")
    if zero_ev_empty:
        caps.append("Pass")
    if empty_evidence and provided:
        caps.append("C")
    if calibration_path == "static_identity":
        caps.append("B" if aggregate >= BAND_USABLE_MIN else "C")
    confidence_cap = _most_restrictive(caps)

    # Stable, deduped reason order.
    seen: set[str] = set()
    deduped = [r for r in reasons if not (r in seen or seen.add(r))]

    return TraceQuality(
        aggregate_quality=aggregate,
        quality_band=band,
        quality_reasons=deduped,
        trace_weight=trace_weight,
        probability_calibration_weight=prob_cal_weight,
        evidence_learning_weight=ev_learn_weight,
        confidence_cap=confidence_cap,
    )


# ---------------------------------------------------------------------------
# Session-level zero-evidence blocker (Issue 4)
# ---------------------------------------------------------------------------

ZERO_EVIDENCE_EMPTY_CONTEXT = REASON_ZERO_EVIDENCE_EMPTY_CONTEXT
# More than this many zero-evidence-empty-context traces in one session fails the
# run summary: at that volume the session is reasoning blind, not occasionally.
ZERO_EVIDENCE_EMPTY_CONTEXT_THRESHOLD = 10


@dataclass(frozen=True)
class ZeroEvidenceSummary:
    """Session-level tally of zero-evidence-empty-context traces."""

    count: int
    trace_ids: list[str]
    total: int
    blocked: bool
    diagnostic: str

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "trace_ids": list(self.trace_ids),
            "total": self.total,
            "blocked": self.blocked,
            "diagnostic": self.diagnostic,
        }


def _trace_is_zero_evidence_empty_context(trace: dict) -> bool:
    """True when a persisted trace dict carries the zero-evidence-empty token.

    Trusts the engine-computed ``trace_quality.quality_reasons`` first (the
    authoritative signal from :func:`aggregate_quality`); falls back to deriving
    it from raw fields for legacy traces written before quality_reasons existed.
    """
    tq = trace.get("trace_quality") or {}
    reasons = tq.get("quality_reasons") or []
    if REASON_ZERO_EVIDENCE_EMPTY_CONTEXT in reasons:
        return True
    if "quality_reasons" in tq:
        # Engine wrote reasons and the token is absent -> trust that.
        return False
    # Legacy fallback: derive from evidence_status + context_source.
    evidence_status = tq.get("evidence_status")
    evidence_count = 0 if evidence_status in (None, "empty") else 1
    return is_zero_evidence_empty_context(
        evidence_count=evidence_count, context_source=tq.get("context_source")
    )


def summarize_zero_evidence(traces: Iterable[dict]) -> ZeroEvidenceSummary:
    """Tally zero-evidence-empty-context traces and decide if the run is blocked.

    A run is ``blocked`` when the count exceeds
    :data:`ZERO_EVIDENCE_EMPTY_CONTEXT_THRESHOLD`. The ``diagnostic`` is a
    human-readable block (empty when not blocked) naming the offending traces.
    """
    trace_list = list(traces)
    offenders = [
        str(t.get("trace_id") or "<unknown>")
        for t in trace_list
        if _trace_is_zero_evidence_empty_context(t)
    ]
    count = len(offenders)
    total = len(trace_list)
    blocked = count > ZERO_EVIDENCE_EMPTY_CONTEXT_THRESHOLD
    diagnostic = ""
    if blocked:
        shown = offenders[:20]
        more = f" (+{count - len(shown)} more)" if count > len(shown) else ""
        diagnostic = (
            f"BLOCKER: {count}/{total} traces are zero_evidence_empty_context "
            f"(no structured evidence AND no provided context), exceeding the "
            f"threshold of {ZERO_EVIDENCE_EMPTY_CONTEXT_THRESHOLD}. The session is "
            f"reasoning blind: these traces cannot calibrate, cannot learn, and "
            f"must not produce actionable output. Provide game/player context and "
            f"structured evidence before relying on this run.\n"
            f"Offending traces: {', '.join(shown)}{more}"
        )
    return ZeroEvidenceSummary(
        count=count,
        trace_ids=offenders,
        total=total,
        blocked=blocked,
        diagnostic=diagnostic,
    )
