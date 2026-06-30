"""A5 — Trust Breakdown / Confidence Decomposer (build_trust_breakdown)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from omega.ui.insights import (
    build_evidence_audit,
    build_market_movement,
    build_signal_conflict,
    build_trust_breakdown,
)
from omega.ui.normalizers import build_trace_recommendation_view
from tests.ui.conftest import make_trace


def _build(trace, *, closing=None, signals=None, application=None, outcome=None, qa=None):
    signals = signals or []
    view = build_trace_recommendation_view(trace, evidence_signals=signals)
    rec = next((r for r in view.recommendations if r.is_primary),
               view.recommendations[0] if view.recommendations else None)
    ea = build_evidence_audit(trace=trace, evidence_signals=signals, outcome=outcome,
                              prop_outcomes=[], closing_lines=closing or [], qa_verdict=qa)
    mm = build_market_movement(rec=rec, closing_lines=closing or [])
    sc = build_signal_conflict(rec=rec, evidence_signals=signals,
                               evidence_application=application, market_movement=mm)
    return build_trust_breakdown(trace_quality=trace.get("trace_quality") or {}, rec=rec,
                                 evidence_audit=ea, market_movement=mm, signal_conflict=sc)


def test_strong_trace_reads_high_trust_with_positive_buckets():
    trace = {
        "trace_id": "t", "kind": "game", "league": "NBA",
        "trace_quality": {"aggregate_quality": 82, "quality_band": "strong",
                          "confidence_cap": None, "trace_weight": 0.82,
                          "calibration_eligible": True, "calibration_path": "profile",
                          "identity_status": "complete", "evidence_status": "present"},
        "recommendations": [{"side": "home", "market": "moneyline", "edge_pct": 5.0,
                             "true_prob": 0.6, "calibrated_prob": 0.62, "odds": 150}],
        "predictions": {"home_win_prob": 0.6}, "result": {},
        "odds_snapshot": {"moneyline_home": -150},
        "input_snapshot": {"evidence": [{"signal_type": "usage"}]},
    }
    signals = [
        {"signal_type": "usage", "direction": "home", "confidence": 0.8, "applied": True},
        {"signal_type": "injury_report", "direction": "home", "confidence": 0.7, "applied": True, "value_json": "out"},
    ]
    closing = [{"market": "moneyline", "selection_descriptor": "home_moneyline",
                "closing_odds": -110, "closing_line": None, "source": "t"}]
    tb = _build(trace, closing=closing, signals=signals)
    assert tb.quality_band == "strong"
    assert tb.headline.startswith("high trust")
    assert "Model edge is meaningful." in tb.positives
    assert "Calibrated by a fitted profile." in tb.positives
    assert {b.name for b in tb.buckets} == {
        "Model strength", "Data quality", "Signal agreement",
        "Market confirmation", "Calibration support", "Volatility",
    }
    market = next(b for b in tb.buckets if b.name == "Market confirmation")
    assert market.polarity == "positive"


def test_qa_fail_and_zero_evidence_emit_fail_warnings():
    trace = {
        "trace_id": "t2", "kind": "game",
        "trace_quality": {"aggregate_quality": 15, "quality_band": "invalid",
                          "confidence_cap": "Pass",
                          "quality_reasons": ["qa_failed", "zero_evidence_empty_context",
                                              "not_calibration_eligible"],
                          "calibration_eligible": False},
        "recommendations": [{"side": "home", "market": "moneyline", "edge_pct": 4.0}],
        "predictions": {}, "result": {},
    }
    tb = _build(trace)
    assert tb.headline == "not trustworthy — confidence capped at Pass"
    codes = {w.code for w in tb.warnings}
    assert "qa_failed" in codes and "zero_evidence_empty_context" in codes
    assert all(w.severity == "fail" for w in tb.warnings)


def test_ineligible_calibration_and_missing_injury_are_negative():
    trace = {
        "trace_id": "t3", "kind": "prop", "league": "NBA",
        "trace_quality": {"aggregate_quality": 45, "quality_band": "weak",
                          "calibration_eligible": False, "identity_status": "complete"},
        "recommendations": [{"selection": "over", "market": "player_prop:pts",
                             "edge_pct": 4.0, "over_prob": 0.55}],
        "predictions": {"over_prob": 0.55}, "result": {"over_prob": 0.55},
        "odds_snapshot": {"odds_over": -110},
        "input_snapshot": {"evidence": [{"signal_type": "usage"}]},
    }
    signals = [{"signal_type": "usage", "direction": "over", "confidence": 0.7, "applied": True, "value_json": "1"}]
    tb = _build(trace, signals=signals)
    assert "Not calibration-eligible." in tb.negatives
    cal = next(b for b in tb.buckets if b.name == "Calibration support")
    assert cal.polarity == "negative"
    vol = next(b for b in tb.buckets if b.name == "Volatility")
    assert any("injury" in c.text.lower() for c in vol.contributions)


def _client(tmp_path: Path, setup: Callable[[TraceStore], None]) -> TestClient:
    db = str(tmp_path / "a5.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        setup(store)
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def test_trace_detail_api_and_html_surface_trust_breakdown(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(make_trace("tb-1")))
    tb = client.get("/api/traces/tb-1").json()["trust_breakdown"]
    assert tb is not None
    assert tb["quality_band"] == "strong"  # derived from 0.85 aggregate
    assert len(tb["buckets"]) == 6
    assert "Trust Breakdown" in client.get("/traces/tb-1").text
