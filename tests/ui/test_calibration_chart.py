"""Calibration Cockpit chart — single-unit, server-computed geometry.

Pins the doctrine: the chart declares ONE explicit unit, its SVG geometry is
computed server-side (the template only drops coordinates), and an empty source
renders an honest empty state rather than a fabricated line.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore
from omega.ui.service import ConsoleService
from tests.ui.conftest import make_trace

_BETS = [
    ("bet-1", "led-1", "2026-03-20", -150, -200),
    ("bet-2", "led-2", "2026-03-21", -120, -110),
]


def _seed_clv(db: str) -> None:
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        for tid, led, date, odds, close in _BETS:
            store.persist(make_trace(tid, kind="game", timestamp=date + "T12:00:00Z", matchup="C @ D"))
            store.record_ledger_bet(
                LedgerBet(
                    ledger_id=led,
                    trace_id=tid,
                    bet_date=date,
                    league="NBA",
                    sport="basketball",
                    matchup="C @ D",
                    market="moneyline",
                    bookmaker="dk",
                    selection="C ML",
                    selection_descriptor="home_moneyline",
                    odds=float(odds),
                    stake_amount=25.0,
                    status=LedgerStatus.WON,
                    provenance=BetProvenance.USER_CONFIRMED,
                    decision_timestamp=date + "T12:00:00Z",
                )
            )
            store.attach_closing_line(
                tid, "moneyline", "home_moneyline", float(close), None, date + "T19:00:00Z", "dk"
            )
    store.close()


def _svc(db: str, sessions: Path) -> ConsoleService:
    return ConsoleService(TraceStore(db_path=db, read_only=True), sessions_dir=str(sessions))


def test_chart_is_single_unit_with_server_geometry(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _seed_clv(db)
    svc = _svc(db, sessions)
    try:
        chart = svc.calibration_chart()
    finally:
        svc.close()
    assert chart.mode == "implied_prob_model_vs_market"
    assert chart.unit == "implied probability (%)"  # explicit, single unit
    assert len(chart.points) == 2
    assert all(p.model_value is not None and p.market_value is not None for p in chart.points)
    # Geometry computed server-side (no client math).
    assert chart.model_polyline and chart.market_polyline
    assert len(chart.dots) == 2
    assert chart.y_min < chart.y_max


def test_chart_empty_state_is_honest(tmp_path: Path):
    db = str(tmp_path / "empty.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    TraceStore(db_path=db).close()
    svc = _svc(db, sessions)
    try:
        chart = svc.calibration_chart()
    finally:
        svc.close()
    assert chart.points == []
    assert chart.model_polyline == ""
    assert any(w.code == "no_clv_data" for w in chart.warnings)


def test_calibration_page_renders_chart(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _seed_clv(db)
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    html = client.get("/calibration").text
    assert "Model vs Market" in html
    assert "<svg" in html and "<polyline" in html
    assert "unit: implied probability" in html


def test_command_center_attaches_chart(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _seed_clv(db)
    svc = _svc(db, sessions)
    try:
        cc = svc.command_center()
    finally:
        svc.close()
    assert cc.calibration_chart is not None
    assert len(cc.calibration_chart.points) == 2


def test_chart_api_is_read_only(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _seed_clv(db)
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    resp = client.get("/api/calibration-chart")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "implied_prob_model_vs_market"
    assert client.post("/api/calibration-chart", json={}).status_code == 405
