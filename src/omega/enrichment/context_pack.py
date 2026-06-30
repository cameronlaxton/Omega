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
