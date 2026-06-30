"""In-process enrichment worker — the queued→running→completed transition.

Local-first: no Redis/Celery. The (opt-in) API schedules ``run_enrichment`` as a
FastAPI background task; the enrichment store doubles as the status board. The
worker reads the trace through the read-only console service to build the context
pack, calls the provider, and persists the artifact — every write lands in the
enrichment sidecar DB, never the canonical trace store.
"""

from __future__ import annotations

from omega.enrichment import PROMPT_VERSION
from omega.enrichment.context_pack import build_context_pack
from omega.enrichment.providers import get_provider
from omega.enrichment.schemas import EnrichmentResult
from omega.enrichment.store import EnrichmentStore


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


def run_enrichment(
    *,
    enrichment_id: str,
    enrich_db: str,
    console_db: str | None = None,
    sessions_dir: str | None = None,
    provider_name: str | None = None,
    model: str | None = None,
) -> None:
    """Build the context pack, call the provider, and persist the artifact.

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
            pack = build_context_pack(trace_id, service)
        finally:
            service.close()

        provider = get_provider(provider_name, model)
        result = provider.generate_enrichment(pack)
        store.set_completed(
            enrichment_id,
            provider=provider.name,
            model=getattr(provider, "model", None),
            prompt_version=PROMPT_VERSION,
            context_pack=pack,
            result=result,
            narrative_md=render_narrative_md(result),
        )
    except Exception as exc:  # noqa: BLE001 — a worker failure must be recorded, not raised
        store.set_failed(enrichment_id, f"{type(exc).__name__}: {exc}")
    finally:
        store.close()


__all__ = ["run_enrichment", "render_narrative_md"]
