"""A1 — Missing Evidence Auditor (omega.ui.insights.build_evidence_audit).

Unit-tests the deterministic present/missing checklist in isolation plus its
read-only surfacing on the trace-detail API.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from omega.ui.insights import build_evidence_audit
from tests.ui.conftest import make_trace


def _full_trace(kind: str = "prop") -> dict:
    return {
        "kind": kind,
        "odds_snapshot": {"moneyline_home": -150},
        "predictions": {"over_prob": 0.55},
        "input_snapshot": {"evidence": [{"signal_type": "usage_spike"}]},
        "trace_quality": {"evidence_status": "present", "calibration_eligible": True},
    }


def _signals() -> list[dict]:
    return [
        {"signal_type": "usage_spike", "value_json": '{"v": 1}', "confidence": 0.8},
        {"signal_type": "injury_report", "category": "situational", "value_json": "out", "confidence": 0.9},
    ]


def _audit(trace, **kw):
    base = dict(
        evidence_signals=_signals(),
        outcome={"result": "home_win"},
        prop_outcomes=[],
        closing_lines=[{"market": "moneyline", "closing_odds": -140}],
        qa_verdict={"verdict": "pass"},
    )
    base.update(kw)
    return build_evidence_audit(trace=trace, **base)


def test_fully_grounded_trace_is_good_with_no_warnings():
    view = _audit(_full_trace())
    assert view.evidence_quality == "good"
    assert view.present_count == view.total_count
    assert view.warnings == []
    assert all(it.present for it in view.items)


def test_missing_model_prediction_is_poor_and_fails():
    trace = _full_trace()
    trace.pop("predictions")
    view = _audit(trace)
    assert view.evidence_quality == "poor"
    pred = next(it for it in view.items if it.key == "model_prediction")
    assert pred.present is False and pred.critical is True
    codes = {(w.code, w.severity) for w in view.warnings}
    assert ("missing_model_prediction", "fail") in codes


def test_no_evidence_at_all_is_poor():
    trace = _full_trace()
    trace.pop("input_snapshot")
    trace["trace_quality"] = {"calibration_eligible": True}
    view = _audit(trace, evidence_signals=[])
    assert view.evidence_quality == "poor"
    blocks = next(it for it in view.items if it.key == "evidence_blocks")
    assert blocks.present is False
    assert any(w.code == "missing_evidence_blocks" and w.severity == "fail" for w in view.warnings)


def test_injury_is_critical_for_prop_but_not_for_game():
    # Prop with no availability signal -> injury critical + poor.
    prop = _audit(_full_trace("prop"), evidence_signals=[{"signal_type": "usage_spike", "value_json": "1"}])
    injury_prop = next(it for it in prop.items if it.key == "injury_context")
    assert injury_prop.critical is True and injury_prop.present is False
    assert prop.evidence_quality == "poor"

    # Same gap on a game line is only informational.
    game = _audit(_full_trace("game"), evidence_signals=[{"signal_type": "pace_up", "value_json": "1"}])
    injury_game = next(it for it in game.items if it.key == "injury_context")
    assert injury_game.critical is False
    assert game.evidence_quality in {"good", "partial"}


def test_only_noncritical_missing_is_partial():
    # All critical inputs present (incl. injury signal); drop outcome + close + QA.
    view = _audit(
        _full_trace("prop"),
        outcome=None,
        prop_outcomes=[],
        closing_lines=[],
        qa_verdict=None,
    )
    assert view.evidence_quality == "partial"
    assert view.warnings == []  # non-critical gaps never warn


def _client(tmp_path: Path, setup: Callable[[TraceStore], None]) -> TestClient:
    db = str(tmp_path / "a1.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        setup(store)
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def test_trace_detail_api_surfaces_evidence_audit(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(make_trace("ea-1")))
    body = client.get("/api/traces/ea-1").json()
    ea = body["evidence_audit"]
    assert ea is not None
    assert ea["evidence_quality"] in {"good", "partial", "poor"}
    assert {it["key"] for it in ea["items"]} >= {"odds_snapshot", "model_prediction", "evidence_blocks"}
    assert ea["present_count"] <= ea["total_count"] == len(ea["items"])


def test_trace_detail_html_renders_auditor_panel(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(make_trace("ea-2")))
    html = client.get("/traces/ea-2").text
    assert "Evidence Auditor" in html
