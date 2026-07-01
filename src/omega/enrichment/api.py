"""The enrichment write API — the only mutating HTTP surface in the stack.

A *separate* FastAPI sub-app, mounted at ``/enrich`` beside the read-only console
by the opt-in ``omega-enrich`` entry point. The read-only console never imports
this module; the route-level read-only guard skips mounted sub-apps; and every
write here targets the enrichment sidecar DB only.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException

from omega.enrichment.schemas import EnrichmentFeedback
from omega.enrichment.store import EnrichmentStore
from omega.enrichment.worker import run_enrichment

_WRITE_INTENT_HEADER = "X-Omega-Enrich-Intent"
_WRITE_INTENT_VALUE = "operator-console"


def _require_write_intent(
    value: Annotated[str | None, Header(alias=_WRITE_INTENT_HEADER)] = None,
) -> None:
    if value != _WRITE_INTENT_VALUE:
        raise HTTPException(status_code=403, detail="missing enrichment write intent")


def build_enrichment_app(
    *,
    enrich_db: str,
    console_db: str | None = None,
    sessions_dir: str | None = None,
    provider_name: str | None = None,
    model: str | None = None,
) -> FastAPI:
    """Build the writable enrichment sub-app bound to one enrichment sidecar DB."""
    app = FastAPI(title="Omega Enrichment", version="1")
    app.state.enrich_db = enrich_db
    # Ensure the sidecar DB + schema exist so read-only GETs work before the
    # first generation writes anything.
    EnrichmentStore(enrich_db).close()

    def _store(read_only: bool = False) -> EnrichmentStore:
        return EnrichmentStore(enrich_db, read_only=read_only)

    def _record_payload(record: Any) -> dict[str, Any]:
        return record.model_dump(mode="json")

    @app.post("/traces/{trace_id}")
    def enqueue(
        trace_id: str,
        background_tasks: BackgroundTasks,
        write_intent: Annotated[str | None, Header(alias=_WRITE_INTENT_HEADER)] = None,
    ) -> dict[str, str]:
        _require_write_intent(write_intent)
        store = _store()
        try:
            eid = store.create(
                trace_id=trace_id, trace_type=None, league=None, market=None, depth="deep"
            )
        finally:
            store.close()
        background_tasks.add_task(
            run_enrichment,
            enrichment_id=eid,
            enrich_db=enrich_db,
            console_db=console_db,
            sessions_dir=sessions_dir,
            provider_name=provider_name,
            model=model,
        )
        return {"enrichment_id": eid, "status": "queued"}

    @app.get("/enrichments/{enrichment_id}")
    def get_enrichment(enrichment_id: str) -> dict[str, Any]:
        store = _store(read_only=True)
        try:
            record = store.get(enrichment_id)
        finally:
            store.close()
        if record is None:
            raise HTTPException(status_code=404, detail="enrichment not found")
        return _record_payload(record)

    @app.get("/traces/{trace_id}/enrichments")
    def list_enrichments(trace_id: str) -> dict[str, Any]:
        store = _store(read_only=True)
        try:
            records = store.list_for_trace(trace_id)
        finally:
            store.close()
        return {"trace_id": trace_id, "enrichments": [_record_payload(r) for r in records]}

    @app.get("/traces/{trace_id}/latest")
    def latest(trace_id: str) -> dict[str, Any]:
        store = _store(read_only=True)
        try:
            record = store.latest_for_trace(trace_id)
        finally:
            store.close()
        return {"trace_id": trace_id, "latest": _record_payload(record) if record else None}

    @app.post("/enrichments/{enrichment_id}/feedback")
    def feedback(
        enrichment_id: str,
        body: EnrichmentFeedback,
        write_intent: Annotated[str | None, Header(alias=_WRITE_INTENT_HEADER)] = None,
    ) -> dict[str, str]:
        _require_write_intent(write_intent)
        store = _store()
        try:
            if store.trace_id_for(enrichment_id) is None:
                raise HTTPException(status_code=404, detail="enrichment not found")
            fid = store.add_feedback(enrichment_id, body)
        finally:
            store.close()
        return {"feedback_id": fid, "status": "recorded"}

    return app


__all__ = ["build_enrichment_app"]
