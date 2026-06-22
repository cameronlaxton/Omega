"""
Qualitative-signal feedback classification (Issue #22, Phase 6).

The closed-loop signal-feedback gate. After the trace-enrichment phases (1-5),
every freshly produced trace carries normalized per-signal evidence applications
(``raw_factor`` / ``family_role`` / ``confidence_defaulted`` / ``final_applied_factor``
on the handler path; ``effective_scalar`` on the Markov path) plus the
trace-level dimensions needed to interpret them. This module reads those traces
and classifies each one across the dimensions the issue requires the report to
distinguish:

    signal present | signal applied | evidence mode | backend path |
    market type | calibration eligibility | outcome resolution status

Crucially it is a *gate*: a trace produced before enrichment lacks the
normalized fields, so it is labeled ``insufficient`` and excluded from every
per-signal aggregate — never scored silently. Traces with no evidence at all
are labeled ``no_evidence`` (an evidence gap, not a defect).

This module is pure (no I/O, no DB). ``omega-report-qualitative-feedback`` owns
the trace query and rendering side; this file owns only the classification and
aggregation. It deliberately does not re-implement directional scoring — that
lives in ``signal_performance.py`` / ``omega-score-evidence-signals``; this
report says *which* signals are ready to feed it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

# A trace was produced by the Issue #22 enriched evidence pipeline when any of
# its applications carries one of these normalized fields. Absent on every
# application => a pre-enrichment trace, which the gate labels ``insufficient``.
_ENRICHMENT_MARKERS: frozenset[str] = frozenset(
    {
        "confidence_defaulted",
        "final_applied_factor",
        "raw_factor",
        "effective_scalar",
        "family_role",
    }
)

# Per-trace feedback status.
SUFFICIENT = "sufficient"
INSUFFICIENT = "insufficient"
NO_EVIDENCE = "no_evidence"

# Version of the rendered report format. Emitted as a machine-detectable marker
# so the report shape can evolve without breaking any future parser.
REPORT_SCHEMA_VERSION = 1


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-trace classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalFeedback:
    """One evidence signal's normalized state on a sufficient trace."""

    signal_type: str
    applied: bool
    target: str
    evidence_mode: str | None
    confidence: float | None
    confidence_defaulted: bool | None
    final_factor: float | None
    family_role: str | None


@dataclass(frozen=True)
class TraceFeedback:
    """One trace classified across the feedback-gate dimensions.

    ``signals`` is populated only for ``SUFFICIENT`` traces; ``INSUFFICIENT``
    and ``NO_EVIDENCE`` traces carry a ``reason`` and no per-signal rows so they
    can be labeled but never silently aggregated.
    """

    trace_id: str
    status: str  # SUFFICIENT | INSUFFICIENT | NO_EVIDENCE
    reason: str | None
    evidence_mode: str | None
    backend_path: str | None
    market_type: str | None
    calibration_eligible: bool
    outcome_resolved: bool
    signals: tuple[SignalFeedback, ...] = ()


def _has_enrichment(apps: list[Any]) -> bool:
    return any(isinstance(a, dict) and (_ENRICHMENT_MARKERS & a.keys()) for a in apps)


def _backend_path(trace: dict[str, Any]) -> str | None:
    """Resolve which simulation path priced the trace.

    Prefers the concrete backend name on the simulation result; falls back to
    the evidence-routing mode (markov vs plane) when the backend is absent.
    """
    simulation = (trace.get("result") or {}).get("simulation") or {}
    backend = simulation.get("simulation_backend")
    if backend:
        return str(backend)
    mode = trace.get("evidence_mode")
    if mode == "markov_transition":
        return "markov"
    return "plane" if mode else None


def _calibration_eligible(trace: dict[str, Any]) -> bool:
    tq = trace.get("trace_quality") or trace.get("quality_gate") or {}
    return bool(tq.get("calibration_eligible"))


def _outcome_resolved(trace: dict[str, Any]) -> bool:
    return bool(trace.get("_outcome") or trace.get("_prop_outcomes"))


def classify_trace(trace: dict[str, Any]) -> TraceFeedback:
    """Classify one trace dict (as returned by ``TraceStore.query_traces``).

    The trace carries the full ``evidence_application`` list from its persisted
    blob plus any attached ``_outcome`` / ``_prop_outcomes``.
    """
    trace_id = str(trace.get("trace_id") or "")
    evidence_mode = trace.get("evidence_mode")
    backend_path = _backend_path(trace)
    market_type = trace.get("kind")
    calibration_eligible = _calibration_eligible(trace)
    outcome_resolved = _outcome_resolved(trace)

    apps = trace.get("evidence_application") or []
    input_evidence = (trace.get("input_snapshot") or {}).get("evidence") or []
    base = dict(
        trace_id=trace_id,
        evidence_mode=evidence_mode,
        backend_path=backend_path,
        market_type=market_type,
        calibration_eligible=calibration_eligible,
        outcome_resolved=outcome_resolved,
    )

    if not apps:
        # A trace that supplied evidence but carries no application records is a
        # readiness gap (pre-enrichment / unrecorded), not a no-evidence trace —
        # surface it as INSUFFICIENT so it is labeled, not silently hidden.
        if input_evidence:
            return TraceFeedback(
                status=INSUFFICIENT,
                reason=(
                    "evidence present in input_snapshot but no evidence_application "
                    "records (pre-enrichment or unrecorded application)"
                ),
                **base,
            )
        return TraceFeedback(
            status=NO_EVIDENCE,
            reason="no evidence supplied and no evidence_application recorded",
            **base,
        )
    if not _has_enrichment(apps):
        return TraceFeedback(
            status=INSUFFICIENT,
            reason=(
                "pre-enrichment trace: evidence_application missing normalized "
                "Issue #22 fields (raw_factor/family_role/confidence_defaulted/...)"
            ),
            **base,
        )

    signals = tuple(
        SignalFeedback(
            signal_type=str(a.get("signal_type") or "unknown"),
            applied=bool(a.get("applied")),
            target=str(a.get("target") or "unknown"),
            evidence_mode=a.get("evidence_mode") or evidence_mode,
            confidence=_as_float_or_none(a.get("confidence")),
            confidence_defaulted=a.get("confidence_defaulted"),
            final_factor=_as_float_or_none(a.get("final_applied_factor", a.get("factor"))),
            family_role=a.get("family_role"),
        )
        for a in apps
        if isinstance(a, dict)
    )
    return TraceFeedback(status=SUFFICIENT, reason=None, signals=signals, **base)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalTypeSummary:
    """Coverage of one signal type across the feedback dimensions.

    Counts are over ``SUFFICIENT`` traces only. ``applied`` counts the signals
    the engine actually applied; the ``by_*`` and ``*_applied`` counters refine
    those by routing/eligibility so the operator can see, e.g., that
    ``def_matchup_weak`` is applied on the Markov path and has 12 resolved
    outcomes ready for scoring.
    """

    signal_type: str
    present: int
    applied: int
    outcome_resolved_applied: int
    calibration_eligible_applied: int
    by_evidence_mode: dict[str, int]
    by_backend_path: dict[str, int]
    by_market_type: dict[str, int]


@dataclass(frozen=True)
class QualitativeFeedbackReport:
    """Aggregate feedback-gate report over a set of traces."""

    total_traces: int
    sufficient: int
    insufficient: int
    no_evidence: int
    insufficient_trace_ids: tuple[str, ...]
    signal_summaries: tuple[SignalTypeSummary, ...]
    classifications: tuple[TraceFeedback, ...] = field(default=())


def build_report(traces: list[dict[str, Any]]) -> QualitativeFeedbackReport:
    """Classify every trace and aggregate per-signal coverage.

    Per-signal aggregates draw exclusively from ``SUFFICIENT`` traces, so an
    older/immature trace is labeled (its id is listed) but never folded into a
    signal's counts.
    """
    classifications = [classify_trace(t) for t in traces]
    sufficient = [c for c in classifications if c.status == SUFFICIENT]
    insufficient = [c for c in classifications if c.status == INSUFFICIENT]
    no_evidence = [c for c in classifications if c.status == NO_EVIDENCE]

    present: Counter[str] = Counter()
    applied: Counter[str] = Counter()
    resolved_applied: Counter[str] = Counter()
    calib_applied: Counter[str] = Counter()
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_backend: dict[str, Counter[str]] = defaultdict(Counter)
    by_market: dict[str, Counter[str]] = defaultdict(Counter)

    for c in sufficient:
        for s in c.signals:
            st = s.signal_type
            present[st] += 1
            if not s.applied:
                continue
            applied[st] += 1
            if c.outcome_resolved:
                resolved_applied[st] += 1
            if c.calibration_eligible:
                calib_applied[st] += 1
            by_mode[st][s.evidence_mode or "unknown"] += 1
            by_backend[st][c.backend_path or "unknown"] += 1
            by_market[st][c.market_type or "unknown"] += 1

    summaries = tuple(
        SignalTypeSummary(
            signal_type=st,
            present=present[st],
            applied=applied[st],
            outcome_resolved_applied=resolved_applied[st],
            calibration_eligible_applied=calib_applied[st],
            by_evidence_mode=dict(by_mode[st]),
            by_backend_path=dict(by_backend[st]),
            by_market_type=dict(by_market[st]),
        )
        for st in sorted(present)
    )

    return QualitativeFeedbackReport(
        total_traces=len(classifications),
        sufficient=len(sufficient),
        insufficient=len(insufficient),
        no_evidence=len(no_evidence),
        insufficient_trace_ids=tuple(c.trace_id for c in insufficient),
        signal_summaries=summaries,
        classifications=tuple(classifications),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_report_markdown(report: QualitativeFeedbackReport) -> str:
    """Render a deterministic Markdown summary of the feedback gate."""
    lines = [
        f"<!-- omega:qualitative_feedback_report schema_version={REPORT_SCHEMA_VERSION} -->",
        "# Qualitative Signal Feedback",
        "",
        "## Trace readiness",
        "",
        f"- Total traces: {report.total_traces}",
        f"- Sufficient (enriched, usable): {report.sufficient}",
        f"- Insufficient (pre-enrichment, labeled — NOT scored): {report.insufficient}",
        f"- No evidence: {report.no_evidence}",
    ]
    if report.insufficient_trace_ids:
        shown = ", ".join(report.insufficient_trace_ids[:10])
        more = (
            f" (+{len(report.insufficient_trace_ids) - 10} more)"
            if len(report.insufficient_trace_ids) > 10
            else ""
        )
        lines += ["", f"Insufficient trace ids: {shown}{more}"]

    lines += ["", "## Signal coverage (sufficient traces only)", ""]
    if not report.signal_summaries:
        lines.append("_No qualitative signals on any sufficient trace._")
    else:
        lines.append(
            f"  {'signal_type':<28} {'present':>7} {'applied':>7} "
            f"{'resolved':>8} {'calib':>6}  routing"
        )
        lines.append(f"  {'-' * 28} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 6}  {'-' * 24}")
        for s in report.signal_summaries:
            routing = ",".join(f"{k}:{v}" for k, v in sorted(s.by_backend_path.items()))
            lines.append(
                f"  {s.signal_type:<28} {s.present:>7} {s.applied:>7} "
                f"{s.outcome_resolved_applied:>8} {s.calibration_eligible_applied:>6}  "
                f"{routing}"
            )
    return "\n".join(lines) + "\n"
