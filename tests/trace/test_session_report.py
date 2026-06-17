from __future__ import annotations

import hashlib
import inspect
import json
import tempfile

import pytest

from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.session_report.context_bundle import (
    ContextBundleError,
    load_context_bundle,
)
from omega.trace.session_report.extractors import extract_intake_report
from omega.trace.session_report.markdown import render_intake_markdown
from omega.trace.store import TraceStore


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("OMEGA_BET_LEDGER_AUTOLOG", "0")
    monkeypatch.setenv("OMEGA_RUNTIME_DIR", str(tmp_path / "runtime"))
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    s = TraceStore(db_path=tmp.name)
    yield s
    s.close()


def _trace(trace_id: str = "sandbox-intake1") -> dict:
    return {
        "trace_id": trace_id,
        "run_id": "run-intake",
        "timestamp": "2026-06-17T12:00:00Z",
        "session_id": "sess-report",
        "prompt": "daily intake",
        "league": "NBA",
        "matchup": "Aces @ Comets",
        "execution_mode": "native_sim",
        "simulation_seed": 123,
        "kind": "game",
        "result": {
            "status": "success",
            "model_prob": 58.2,
            "best_bet": {
                "selection": "Comets ML",
                "edge_pct": 3.4,
                "recommended_units": 0.5,
                "confidence_tier": "B",
            },
        },
        "recommendations": [
            {
                "selection": "Comets ML",
                "market": "moneyline",
                "edge_pct": 3.4,
                "units": 0.5,
                "confidence_tier": "B",
            }
        ],
        "input_snapshot": {
            "bookmaker": "betmgm",
            "home_context": {"off_rating": 114.0},
            "away_context": {"def_rating": 108.0},
            "game_context": {"rest_days": 2},
            "evidence": [
                {
                    "signal_type": "rest_advantage",
                    "category": "situational",
                    "plane": "game",
                    "source": "schedule",
                    "confidence": 0.8,
                    "window": "last_1",
                    "direction": "home",
                }
            ],
        },
        "trace_quality": {
            "calibration_eligible": True,
            "evidence_status": "present",
            "context_source": "provided",
            "identity_status": "complete",
        },
    }


def _ledger(trace_id: str = "sandbox-intake1") -> LedgerBet:
    return LedgerBet(
        ledger_id="ledger-intake1",
        trace_id=trace_id,
        bet_date="2026-06-17",
        league="NBA",
        sport="basketball",
        matchup="Aces @ Comets",
        market="moneyline",
        bookmaker="betmgm",
        selection="Comets ML",
        selection_descriptor="home_moneyline",
        line=None,
        odds=-115,
        stake_amount=25.0,
        status=LedgerStatus.PENDING,
        provenance=BetProvenance.ENGINE_AUTO,
        decision_timestamp="2026-06-17T12:00:00Z",
    )


def _bundle_payload(session_id: str = "sess-report") -> dict:
    claim = "Starter returned to full participation in pregame notes."
    return {
        "schema_version": 1,
        "bundle_id": "bundle-1",
        "session_id": session_id,
        "generated_at": "2026-06-17T12:30:00Z",
        "generated_by": "test",
        "mode": "persisted+cited",
        "entries": [
            {
                "entry_id": "ctx-1",
                "trace_id": "sandbox-intake1",
                "category": "injury_role",
                "source_type": "official",
                "source_title": "Team notes",
                "source_url": "https://example.com/notes",
                "captured_at": "2026-06-17T12:25:00Z",
                "claim": claim,
                "claim_hash": hashlib.sha256(claim.encode("utf-8")).hexdigest(),
            }
        ],
    }


def test_context_bundle_rejects_prohibited_quant_claim(tmp_path):
    claim = "Model probability improved and edge% is stronger."
    payload = _bundle_payload()
    payload["entries"][0]["claim"] = claim
    payload["entries"][0]["claim_hash"] = hashlib.sha256(claim.encode("utf-8")).hexdigest()
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContextBundleError):
        load_context_bundle(path)


def test_cited_context_rejected_in_replay_mode(tmp_path, monkeypatch):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")

    with pytest.raises(ContextBundleError):
        load_context_bundle(path)


def test_intake_report_joins_trace_ledger_and_context_bundle(store, tmp_path):
    store.persist(_trace())
    store.record_ledger_bet(_ledger())
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    bundle = load_context_bundle(bundle_path)

    data = extract_intake_report(
        store,
        session_id="sess-report",
        context_mode="persisted+cited",
        context_bundle=bundle,
    )

    assert data.trace_count == 1
    assert data.ledger_count == 1
    assert data.context_bundle_id == "bundle-1"
    assert data.cards[0].ledger_view.provenance == "engine_auto"
    assert any(b.source_type == "context_bundle" for b in data.cards[0].context)
    assert any(r.label == "engine_auto" for r in data.ledger_linkage)


def test_mismatched_context_entries_are_ignored(store, tmp_path):
    store.persist(_trace())
    payload = _bundle_payload()
    payload["entries"][0]["trace_id"] = "sandbox-other"
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    data = extract_intake_report(
        store,
        session_id="sess-report",
        context_mode="persisted+cited",
        context_bundle=load_context_bundle(path),
    )

    assert not any(b.source_type == "context_bundle" for b in data.cards[0].context)
    assert data.ignored_context_entries[0].reason == "trace_id mismatch"


def test_renderer_is_pure_and_uses_narrow_tables(store):
    store.persist(_trace())
    store.record_ledger_bet(_ledger())
    data = extract_intake_report(store, session_id="sess-report")

    rendered = render_intake_markdown(data)

    assert "TraceStore" not in inspect.getsource(render_intake_markdown)
    assert "## Persisted Recommendation Cards" in rendered
    for line in rendered.splitlines():
        if line.startswith("|") and line.endswith("|"):
            columns = len([part for part in line.split("|")[1:-1]])
            assert columns <= 5


def test_intake_report_renders_missing_context_when_evidence_absent(store):
    trace = _trace()
    trace["input_snapshot"]["evidence"] = []
    trace["trace_quality"]["evidence_status"] = "empty"
    store.persist(trace)

    rendered = render_intake_markdown(extract_intake_report(store, session_id="sess-report"))

    assert "[missing] not captured: no evidence rows were available" in rendered
