"""B1–B5 — enrichment store, schemas, context pack, providers, worker (offline)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from omega.enrichment.context_pack import build_context_pack
from omega.enrichment.providers import StubProvider, get_provider
from omega.enrichment.schemas import (
    RECOMMENDATION_TYPES,
    EnrichmentFeedback,
    EnrichmentResult,
    sanitize_raw_result,
)
from omega.enrichment.store import EnrichmentStore
from omega.enrichment.worker import render_narrative_md, run_enrichment
from omega.ui.service import open_service

# ---- schemas --------------------------------------------------------------


def test_recommendation_type_enum_is_enforced():
    with pytest.raises(ValidationError):
        EnrichmentResult(headline="h", summary="s", recommendation_type="bet_this_now")


def test_sanitize_strips_forbidden_numeric_keys():
    raw = {"headline": "h", "summary": "s", "edge": 5.0, "kelly_fraction": 0.1,
           "recommendation_type": "lean"}
    clean = sanitize_raw_result(raw)
    assert "edge" not in clean and "kelly_fraction" not in clean
    result = EnrichmentResult.model_validate(clean)
    assert result.recommendation_type == "lean"


# ---- store ----------------------------------------------------------------


def test_store_lifecycle(enrich_db: str):
    store = EnrichmentStore(enrich_db)
    eid = store.create(trace_id="t1", trace_type="game", league="NBA", market="moneyline")
    assert store.get(eid).status == "queued"
    store.set_running(eid)
    assert store.get(eid).status == "running"
    store.set_completed(
        eid, provider="stub", model=None, prompt_version="v1",
        context_pack={"trace_id": "t1"},
        result=EnrichmentResult(headline="H", summary="S"), narrative_md="md body",
    )
    rec = store.get(eid)
    assert rec.status == "completed" and rec.narrative_md == "md body"
    assert rec.result is not None and rec.result.headline == "H"
    store.add_feedback(eid, EnrichmentFeedback(user_rating=1, feedback_text="useful"))
    assert store.latest_for_trace("t1").id == eid
    assert store.trace_id_for(eid) == "t1"
    with pytest.raises(sqlite3.IntegrityError):
        store.add_feedback("missing", EnrichmentFeedback(user_rating=-1))
    store.close()


# ---- providers ------------------------------------------------------------


def test_get_provider_defaults_to_stub():
    assert isinstance(get_provider(None), StubProvider)
    assert isinstance(get_provider("stub"), StubProvider)
    with pytest.raises(ValueError):
        get_provider("stbu")


def test_stub_provider_builds_valid_result_from_pack():
    pack = {
        "league": "NBA", "kind": "game",
        "trust": {"headline": "medium trust", "positives": ["Edge is meaningful."],
                  "negatives": ["Evidence thin."]},
        "guardrails": {"worst_severity": "warn", "summary": "1 warning",
                       "flags": [{"severity": "warn", "message": "stale", "action": "recheck price"}]},
        "market_movement": {"direction": "against", "headline": "moved against"},
        "signal_conflict": {"dominant": "market_conflict"},
        "missing_context": ["No closing line"],
        "historical_support": "mixed",
    }
    result = StubProvider().generate_enrichment(pack)
    assert result.recommendation_type in RECOMMENDATION_TYPES
    assert result.risk_rating == "medium"
    assert "recheck price" in result.operator_notes
    assert result.missing_context == ["No closing line"]
    # market moved against -> conservative monitor
    assert result.recommendation_type == "monitor"


def test_render_narrative_md_has_sections():
    result = EnrichmentResult(headline="Head", summary="Sum", model_case=["a"],
                              counter_case=["b"], operator_notes=["c"])
    md = render_narrative_md(result)
    assert "## Head" in md and "Why Omega likes it" in md and "Operator notes" in md
    assert "engine owns probability" in md


# ---- context pack ---------------------------------------------------------


def test_context_pack_composes_views_without_protected_numbers(traces_db: str, tmp_path: Path):
    service = open_service(db_path=traces_db, sessions_dir=str(tmp_path))
    try:
        pack = build_context_pack("enr-1", service)
    finally:
        service.close()
    assert pack["trace_id"] == "enr-1" and pack["league"] == "NBA"
    for key in ("trust", "guardrails", "market_movement", "signal_conflict",
                "missing_context", "historical_support"):
        assert key in pack
    # Public market fields are allowed; engine-owned numbers must NOT be present anywhere.
    for forbidden in ("edge", "edge_pct", "ev", "probability", "kelly", "stake"):
        assert forbidden not in set(_walk_keys(pack))


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_context_pack_missing_trace_raises(traces_db: str, tmp_path: Path):
    service = open_service(db_path=traces_db, sessions_dir=str(tmp_path))
    try:
        with pytest.raises(ValueError):
            build_context_pack("nope", service)
    finally:
        service.close()


# ---- worker (end-to-end, offline stub) ------------------------------------


def test_run_enrichment_completes_with_stub(traces_db: str, enrich_db: str, tmp_path: Path):
    store = EnrichmentStore(enrich_db)
    eid = store.create(trace_id="enr-1", trace_type=None, league=None, market=None)
    store.close()

    run_enrichment(enrichment_id=eid, enrich_db=enrich_db, console_db=traces_db,
                   sessions_dir=str(tmp_path), provider_name="stub")

    store = EnrichmentStore(enrich_db, read_only=True)
    rec = store.get(eid)
    store.close()
    assert rec.status == "completed"
    assert rec.narrative_md and rec.result is not None
    assert rec.result.recommendation_type in RECOMMENDATION_TYPES
    assert rec.provider == "stub"


def test_run_enrichment_missing_trace_marks_failed(enrich_db: str, traces_db: str, tmp_path: Path):
    store = EnrichmentStore(enrich_db)
    eid = store.create(trace_id="does-not-exist", trace_type=None, league=None, market=None)
    store.close()
    run_enrichment(enrichment_id=eid, enrich_db=enrich_db, console_db=traces_db,
                   sessions_dir=str(tmp_path), provider_name="stub")
    store = EnrichmentStore(enrich_db, read_only=True)
    rec = store.get(eid)
    store.close()
    assert rec.status == "failed" and rec.error
