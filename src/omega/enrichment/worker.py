"""In-process enrichment worker — the queued→running→completed transition.

Local-first: no Redis/Celery. The (opt-in) API schedules ``run_enrichment`` as a
FastAPI background task; the enrichment store doubles as the status board. The
worker reads the trace through the read-only console service to build the context
pack, calls the provider, and persists the artifact — every write lands in the
enrichment sidecar DB, never the canonical trace store.
"""

from __future__ import annotations

from omega.core.contracts.language import blocked_language
from omega.enrichment import PROMPT_VERSION
from omega.enrichment.context_pack import build_context_pack, build_matchup_context_pack
from omega.enrichment.providers import get_provider
from omega.enrichment.schemas import EnrichmentResult
from omega.enrichment.store import EnrichmentStore

STRESS_TEST_UNAVAILABLE = "not available for this analysis"


def _stress_test_unavailable_result(pack: dict) -> EnrichmentResult:
    """Deterministic stress_test refusal until an engine sensitivity artifact
    exists — presentation code never approximates sensitivity."""
    return EnrichmentResult(
        headline=f"Stress test {STRESS_TEST_UNAVAILABLE}",
        summary=(
            "No deterministic sensitivity artifact is persisted for this "
            "analysis, so a stress test is "
            f"{STRESS_TEST_UNAVAILABLE}. Sensitivity is computed only by the "
            "deterministic engine; this layer never approximates it."
        ),
        missing_context=["engine-owned deterministic sensitivity artifact"],
        operator_notes=[
            "Re-run once the simulation contract persists a sensitivity artifact "
            "(Phase 3)."
        ],
        risk_rating="medium",
        recommendation_type="monitor",
    )


def _brief_has_sensitivity(pack: dict) -> bool:
    markets = (pack.get("brief") or {}).get("markets") or []
    return any(
        (m.get("sensitivity") or {}).get("status") == "available" for m in markets
    )


def render_narrative_md(result: EnrichmentResult) -> str:
    """Render an :class:`EnrichmentResult` into a console-ready markdown artifact."""
    lines: list[str] = [f"## {result.headline}", "", result.summary, ""]

    def _section(title: str, items: list[str]) -> None:
        if items:
            lines.append(f"### {title}")
            lines.extend(f"- {it}" for it in items)
            lines.append("")

    _section("Why Omega likes it", result.model_case)
    lines.append("### Market read")
    lines.append(
        f"- Movement: **{result.market_context.line_movement}** — "
        f"{result.market_context.interpretation}"
    )
    lines.append("")
    _section("Counterarguments", result.counter_case)
    _section("Missing context", result.missing_context)
    _section("Operator notes", result.operator_notes)
    lines.append(f"**Risk:** {result.risk_rating} · **Recommendation:** {result.recommendation_type}")
    lines.append("")
    lines.append("_Narrative only — the deterministic engine owns probability, edge, EV, and stake._")
    return "\n".join(lines)


def render_decision_support_md(result: EnrichmentResult) -> str:
    """Render an :class:`EnrichmentResult` for a decision-support focus.

    Neutral section names and no recommendation-era framing (no "Why Omega
    likes it", no "Recommendation:" label) — this artifact reviews every
    outcome symmetrically and names no pick.
    """
    lines: list[str] = [f"## {result.headline}", "", result.summary, ""]

    def _section(title: str, items: list[str]) -> None:
        if items:
            lines.append(f"### {title}")
            lines.extend(f"- {it}" for it in items)
            lines.append("")

    _section("Market groups reviewed", result.model_case)
    lines.append("### Market read")
    lines.append(
        f"- Movement: **{result.market_context.line_movement}** — "
        f"{result.market_context.interpretation}"
    )
    lines.append("")
    _section("Uncertainties", result.counter_case)
    _section("Missing context", result.missing_context)
    _section("Notes", result.operator_notes)
    lines.append(f"**Risk:** {result.risk_rating}")
    lines.append("")
    lines.append(
        "_Symmetric decision-support review — the deterministic engine owns "
        "probability, edge, EV, and stake; no pick is made here._"
    )
    return "\n".join(lines)


def _blocked_language_in_result(result: EnrichmentResult) -> list[str]:
    """Scan every prose field of a provider result for blocked vocabulary."""
    parts = [
        result.headline,
        result.summary,
        result.confidence_explanation,
        result.market_context.line_movement,
        result.market_context.interpretation,
        *result.model_case,
        *result.counter_case,
        *result.missing_context,
        *result.operator_notes,
    ]
    return blocked_language("\n".join(p for p in parts if p))


def run_enrichment(
    *,
    enrichment_id: str,
    enrich_db: str,
    console_db: str | None = None,
    sessions_dir: str | None = None,
    provider_name: str | None = None,
    model: str | None = None,
    focus: str | None = None,
) -> None:
    """Build the context pack, call the provider, and persist the artifact.

    ``focus`` selects the decision-support follow-up mode (Phase 1): the pack is
    then the safe matchup brief, and ``stress_test`` completes deterministically
    with an explicit not-available artifact unless the engine persisted a
    sensitivity artifact. ``focus=None`` keeps the legacy Deep Dive behavior.

    Never raises: any failure is captured on the enrichment row as ``failed`` so
    the polling UI can surface it honestly.
    """
    from omega.ui.service import open_service  # local import keeps module import light

    store = EnrichmentStore(enrich_db)
    try:
        trace_id = store.trace_id_for(enrichment_id)
        if trace_id is None:
            store.set_failed(enrichment_id, "enrichment row not found")
            return
        store.set_running(enrichment_id)

        service = open_service(db_path=console_db, sessions_dir=sessions_dir)
        try:
            if focus is not None:
                pack = build_matchup_context_pack(trace_id, service, focus)
            else:
                pack = build_context_pack(trace_id, service)
        finally:
            service.close()

        if focus == "stress_test" and not _brief_has_sensitivity(pack):
            unavailable = _stress_test_unavailable_result(pack)
            store.set_completed(
                enrichment_id,
                provider="deterministic",
                model=None,
                prompt_version=PROMPT_VERSION,
                context_pack=pack,
                result=unavailable,
                narrative_md=render_decision_support_md(unavailable),
            )
            return

        provider = get_provider(provider_name, model)
        result = provider.generate_enrichment(pack)

        if focus is not None:
            found = _blocked_language_in_result(result)
            if found:
                store.set_failed(
                    enrichment_id,
                    f"blocked_language_detected: provider output contained {found!r}; "
                    "refusing to persist a decision-support artifact with recommendation "
                    "vocabulary",
                )
                return
            narrative_md = render_decision_support_md(result)
        else:
            narrative_md = render_narrative_md(result)

        store.set_completed(
            enrichment_id,
            provider=provider.name,
            model=getattr(provider, "model", None),
            prompt_version=PROMPT_VERSION,
            context_pack=pack,
            result=result,
            narrative_md=narrative_md,
        )
    except Exception as exc:  # noqa: BLE001 — a worker failure must be recorded, not raised
        store.set_failed(enrichment_id, f"{type(exc).__name__}: {exc}")
    finally:
        store.close()


__all__ = ["render_narrative_md", "render_decision_support_md", "run_enrichment"]
