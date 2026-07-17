"""Context-pack builder — the disciplined input handed to the narrative provider.

This is the highest-leverage piece of the enrichment layer: it composes the six
deterministic decision-quality views (Trust Breakdown, Guardrails, Evidence
Auditor, Market Movement, Signal Conflict, Similar Spots) into a clean schema so
the LLM reasons over *computed, audited* values instead of raw JSON. Crucially it
carries only qualitative trust factors and public market data — the engine's
protected numbers (edge / EV / probability / stake) are deliberately omitted, so
the model cannot anchor on, restate, or "improve" them.
"""

from __future__ import annotations

from typing import Any


def build_context_pack(trace_id: str, service: Any) -> dict[str, Any]:
    """Assemble the enrichment context pack from the read-only console service.

    ``service`` is an :class:`omega.ui.service.ConsoleService` (read-only). Raises
    ``ValueError`` when the trace does not exist.
    """
    detail = service.get_trace_detail(trace_id)
    if detail is None:
        raise ValueError(f"trace {trace_id!r} not found")
    d = detail.model_dump()

    rv = d.get("recommendation_view") or {}
    recs = rv.get("recommendations") or []
    primary = next((r for r in recs if r.get("is_primary")), recs[0] if recs else None)
    recommendation = None
    if primary:
        recommendation = {
            "market": _v(primary.get("market")),
            "selection": _v(primary.get("selection")),
            "line": _v(primary.get("line")),
            "odds": _v(primary.get("odds")),  # public market price (not engine-owned)
            "confidence_band": _v(primary.get("display_confidence_band")),
            # Raw edge / probability / EV are deliberately omitted (engine-owned).
        }

    tb = d.get("trust_breakdown") or {}
    gr = d.get("guardrails") or {}
    ea = d.get("evidence_audit") or {}
    mm = d.get("market_movement") or {}
    sc = d.get("signal_conflict") or {}

    similar = service.similar_spots(trace_id)

    return {
        "trace_id": trace_id,
        "league": d.get("league"),
        "kind": d.get("kind"),
        "matchup": d.get("matchup"),
        "recommendation": recommendation,
        "trust": {
            "band": tb.get("quality_band"),
            "headline": tb.get("headline"),
            "positives": tb.get("positives", []),
            "negatives": tb.get("negatives", []),
            "buckets": [
                {"name": b.get("name"), "summary": b.get("summary"), "polarity": b.get("polarity")}
                for b in tb.get("buckets", [])
            ],
        },
        "guardrails": {
            "worst_severity": gr.get("worst_severity"),
            "summary": gr.get("summary"),
            "flags": [
                {"severity": g.get("severity"), "message": g.get("message"),
                 "action": g.get("suggested_action")}
                for g in gr.get("guardrails", [])
            ],
        },
        "market_movement": {
            "direction": mm.get("direction"),
            "interpretation": mm.get("interpretation"),
            "headline": mm.get("headline"),
        },
        "signal_conflict": {
            "level": sc.get("conflict_level"),
            "dominant": sc.get("dominant_conflict"),
            "supporting": sc.get("supporting_count"),
            "opposing": sc.get("opposing_count"),
        },
        "evidence_quality": ea.get("evidence_quality"),
        "missing_context": [
            it.get("label") for it in ea.get("items", []) if not it.get("present")
        ],
        "historical_support": (similar.historical_support if similar else None),
        "historical_cohorts": (
            [{"label": c.label, "sample": c.sample, "hit_rate": c.hit_rate}
             for c in similar.cohorts]
            if similar else []
        ),
    }


def _v(field: Any) -> Any:
    """Pull the scalar ``value`` out of an ExtractedField dict (or pass through)."""
    if isinstance(field, dict):
        return field.get("value")
    return field


# -- decision-support matchup packs (Phase 1) ---------------------------------

# What each focus asks the provider to do. The provider may only filter,
# compare, and explain the brief — the guidance never authorizes recomputing
# engine values or recommending action.
FOCUS_GUIDANCE: dict[str, str] = {
    "breakdown": (
        "Walk through this matchup's verified context, each market's outcome "
        "cases, and the data-quality warnings. Neutral framing; no side is "
        "preferred."
    ),
    "compare_markets": (
        "Compare the markets within this one event on information quality, "
        "disagreement between context and listed lines, and uncertainty. Do "
        "not rank by attractiveness or imply one should be bet."
    ),
    "stress_test": (
        "Only render the engine-persisted deterministic sensitivity artifact. "
        "If none is present, state that a stress test is not available for "
        "this analysis — never approximate one."
    ),
    "decision_changes": (
        "Identify the concrete conditions (lineup news, line moves, weather, "
        "role changes) that would materially change this assessment, from the "
        "brief's scenario triggers, uncertainties, and decision conditions."
    ),
    "counter_case": (
        "Lay out the strongest case AGAINST each outcome's supporting "
        "evidence, using only recorded challenging evidence and data-quality "
        "warnings. Never fabricate counterarguments that are not grounded in "
        "the brief."
    ),
    "source_check": (
        "Audit the brief's sources: which facts carry a URL and retrieval "
        "timestamp, which are missing provenance, which are stale, and what "
        "should be re-verified before relying on this brief."
    ),
}


def build_matchup_context_pack(
    trace_id: str, service: Any, focus: str = "breakdown"
) -> dict[str, Any]:
    """Decision-support follow-up pack: the safe matchup brief plus focus.

    Reads ONLY the allowlisted ``MatchupBriefV1`` projection (every market on
    the trace's event) — no raw trace payload, no recommendation view, and no
    quarantined similar-spots cohorts. Raises ``ValueError`` for an unknown
    focus or a missing trace.
    """
    from omega.enrichment.schemas import MATCHUP_FOCUS_VALUES
    from omega.trace.decision_support import group_key_for_trace

    if focus not in MATCHUP_FOCUS_VALUES:
        raise ValueError(f"focus {focus!r} not in {MATCHUP_FOCUS_VALUES}")
    trace = service.store.get_trace(trace_id)
    if trace is None:
        raise ValueError(f"trace {trace_id!r} not found")
    group_key, _, _ = group_key_for_trace(trace)
    brief = service.matchup_brief(group_key)
    if brief is None:
        raise ValueError(f"matchup brief unavailable for {group_key!r}")
    return {
        "mode": "decision_support",
        "focus": focus,
        "focus_guidance": FOCUS_GUIDANCE[focus],
        "trace_id": trace_id,
        "group_key": group_key,
        "league": brief.get("league"),
        "matchup": brief.get("matchup"),
        "presentation_mode": brief.get("presentation_mode"),
        "brief": brief,
    }
