"""A6 — Trace Guardrails / Auto Risk Flags (build_trace_guardrails + row chips)."""

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
    build_trace_guardrails,
    build_trust_breakdown,
)
from omega.ui.normalizers import build_trace_recommendation_view
from omega.ui.service import _row_guardrail
from tests.ui.conftest import make_trace


def _guard(trace, *, closing=None, signals=None, application=None, outcome=None, qa=None, odds_age=None):
    signals = signals or []
    view = build_trace_recommendation_view(trace, evidence_signals=signals)
    rec = next((r for r in view.recommendations if r.is_primary),
               view.recommendations[0] if view.recommendations else None)
    ea = build_evidence_audit(trace=trace, evidence_signals=signals, outcome=outcome,
                              prop_outcomes=[], closing_lines=closing or [], qa_verdict=qa)
    mm = build_market_movement(rec=rec, closing_lines=closing or [])
    sc = build_signal_conflict(rec=rec, evidence_signals=signals,
                               evidence_application=application, market_movement=mm)
    tb = build_trust_breakdown(trace_quality=trace.get("trace_quality") or {}, rec=rec,
                               evidence_audit=ea, market_movement=mm, signal_conflict=sc)
    return build_trace_guardrails(trace_quality=trace.get("trace_quality") or {}, rec=rec,
                                  evidence_audit=ea, market_movement=mm, signal_conflict=sc,
                                  trust_breakdown=tb, odds_age_seconds=odds_age)


def _grounded(**tq) -> dict:
    """A fully-grounded game trace; pass trace_quality overrides via kwargs."""
    return {
        "trace_id": "g", "kind": "game", "league": "NBA",
        "trace_quality": {"aggregate_quality": 82, "quality_band": "strong",
                          "confidence_cap": None, "calibration_eligible": True,
                          "calibration_path": "profile", "identity_status": "complete",
                          "evidence_status": "present", **tq},
        "recommendations": [{"side": "home", "market": "moneyline", "edge_pct": 5.0,
                             "true_prob": 0.6, "calibrated_prob": 0.62, "odds": -150}],
        "predictions": {"home_win_prob": 0.6}, "result": {},
        "odds_snapshot": {"moneyline_home": -150},
        "input_snapshot": {"evidence": [{"signal_type": "usage"}]},
    }


_SIGNALS = [
    {"signal_type": "usage", "direction": "home", "confidence": 0.8, "applied": True, "value_json": "1"},
    {"signal_type": "injury_report", "direction": "home", "confidence": 0.7, "applied": True, "value_json": "out"},
]
_CLOSE_TOWARD = [{"market": "moneyline", "selection_descriptor": "home_moneyline",
                  "closing_odds": -200, "closing_line": None, "source": "t"}]
_CLOSE_AGAINST = [{"market": "moneyline", "selection_descriptor": "home_moneyline",
                   "closing_odds": 120, "closing_line": None, "source": "t"}]


def _full(trace, closing):
    return _guard(trace, closing=closing, signals=_SIGNALS,
                  outcome={"result": "home_win"}, qa={"verdict": "pass"})


def test_clean_trace_has_no_guardrails():
    gr = _full(_grounded(), _CLOSE_TOWARD)
    assert gr.worst_severity == "ok"
    assert gr.guardrails == []


def test_qa_fail_and_pass_cap_are_blockers():
    gr = _guard(_grounded(quality_reasons=["qa_failed"], confidence_cap="Pass", quality_band="invalid"))
    codes = {g.code: g.severity for g in gr.guardrails}
    assert codes.get("qa_failed") == "fail"
    assert codes.get("confidence_capped_pass") == "fail"
    assert gr.worst_severity == "fail" and gr.blocker_count >= 2
    # Blockers sort ahead of lower-severity flags.
    assert gr.guardrails[0].severity == "fail"


def test_market_against_is_a_warning_with_action():
    gr = _full(_grounded(), _CLOSE_AGAINST)
    g = next((g for g in gr.guardrails if g.code == "market_moved_against"), None)
    assert g is not None and g.severity == "warn"
    assert g.suggested_action  # actionable copy attached


def test_stale_odds_warns():
    gr = _guard(_grounded(), closing=_CLOSE_TOWARD, signals=_SIGNALS,
                outcome={"result": "home_win"}, qa={"verdict": "pass"}, odds_age=7200)
    assert any(g.code == "stale_odds" and g.severity == "warn" for g in gr.guardrails)


def test_missing_evidence_blocks_is_a_blocker():
    # No evidence at all on a trace with otherwise-fine quality.
    trace = {
        "trace_id": "g2", "kind": "game", "league": "NBA",
        "trace_quality": {"aggregate_quality": 60, "quality_band": "usable"},
        "recommendations": [{"side": "home", "market": "moneyline", "edge_pct": 4.0}],
        "predictions": {"home_win_prob": 0.6}, "odds_snapshot": {"moneyline_home": -150},
    }
    gr = _guard(trace)
    assert any(g.code == "missing_evidence_blocks" and g.severity == "fail" for g in gr.guardrails)
    assert gr.worst_severity == "fail"


def test_row_guardrail_token():
    assert _row_guardrail({"trace_quality": {"quality_reasons": ["qa_failed"]}}) == "fail"
    assert _row_guardrail({"trace_quality": {"confidence_cap": "Pass"}}) == "fail"
    assert _row_guardrail({"trace_quality": {"quality_band": "weak"}}) == "warn"
    assert _row_guardrail({"trace_quality": {"quality_band": "usable"}}) == "info"
    assert _row_guardrail({"trace_quality": {"quality_band": "strong"}}) == "ok"
    assert _row_guardrail({"trace_quality": {"aggregate_quality": 0.85}}) == "ok"
    assert _row_guardrail({"trace_quality": {"aggregate_quality": 0.3}}) == "warn"
    assert _row_guardrail({"trace_quality": {"aggregate_quality": "fail"}}) == "fail"
    assert _row_guardrail({"trace_quality": {"aggregate_quality": "weak"}}) == "warn"


def _client(tmp_path: Path, setup: Callable[[TraceStore], None]) -> TestClient:
    db = str(tmp_path / "a6.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        setup(store)
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def test_guardrails_surface_on_detail_and_lists(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(make_trace("gr-1")))
    detail = client.get("/api/traces/gr-1").json()
    assert detail["guardrails"] is not None
    assert detail["guardrails"]["worst_severity"] in {"ok", "info", "warn", "fail"}

    row = client.get("/api/traces").json()["rows"][0]
    assert row["guardrail"] in {"ok", "info", "warn", "fail"}

    scan = client.get("/api/scanner").json()["rows"]
    assert all(r["guardrail"] in {"ok", "info", "warn", "fail"} for r in scan)
    assert "Trace Guardrails" in client.get("/traces/gr-1").text
    assert "gr-dot-" in client.get("/traces").text
    assert "gr-dot-" in client.get("/scanner").text
