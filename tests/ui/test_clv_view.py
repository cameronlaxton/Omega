"""Milestone B.3 — Market Movement / CLV: closing-line-value read view + helper."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore
from omega.ui.clv import closing_line_value
from tests.ui.conftest import make_trace

# -- pure helper -------------------------------------------------------------


def test_closing_line_value_beat_close():
    r = closing_line_value(-150, -200)  # took 0.600, closed 0.667
    assert r.taken_implied == pytest.approx(0.60)
    assert r.closing_implied == pytest.approx(0.6667, abs=1e-4)
    assert r.clv_points == pytest.approx(0.0667, abs=1e-4)
    assert r.beat_close is True


def test_closing_line_value_missed_close():
    r = closing_line_value(-200, -150)  # took 0.667, closed 0.600
    assert r.clv_points == pytest.approx(-0.0667, abs=1e-4)
    assert r.beat_close is False


def test_closing_line_value_missing_odds():
    assert closing_line_value(-110, None).clv_points is None
    assert closing_line_value(None, -110).beat_close is None
    assert closing_line_value(0, -110).clv_points is None  # 0 is not valid American


# -- service / API -----------------------------------------------------------


def _client(tmp_path: Path, *, setup=None) -> TestClient:
    db = str(tmp_path / "b3.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        if setup:
            setup(store)
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def _seed_bet(store, ledger_id, trace_id, odds, *, descriptor="home_moneyline", closing=None):
    store.persist(make_trace(trace_id, kind="game", matchup="C @ D"))
    store.record_ledger_bet(LedgerBet(
        ledger_id=ledger_id, trace_id=trace_id, bet_date="2026-03-21", league="NBA",
        sport="basketball", matchup="C @ D", market="moneyline", bookmaker="dk",
        selection="C ML", selection_descriptor=descriptor, odds=float(odds),
        stake_amount=25.0, status=LedgerStatus.WON, provenance=BetProvenance.USER_CONFIRMED,
        decision_timestamp="2026-03-21T12:00:00Z",
    ))
    if closing is not None:
        store.attach_closing_line(trace_id, "moneyline", descriptor, float(closing), None,
                                  "2026-03-21T19:00:00Z", "dk")


def test_clv_row_computed_for_bet_with_closing_line(tmp_path):
    client = _client(tmp_path, setup=lambda s: _seed_bet(s, "led-clv", "bet-1", -150, closing=-200))
    body = client.get("/api/clv").json()
    assert body["summary"]["with_closing_line"] == 1
    assert body["summary"]["beat_close"] == 1
    row = body["rows"][0]
    assert row["ledger_id"] == "led-clv"
    assert row["taken_odds"] == -150 and row["closing_odds"] == -200
    assert row["clv_points"] == pytest.approx(0.0667, abs=1e-4)
    assert row["beat_close"] is True
    assert row["closing_source"] == "dk"
    assert row["field_sources"]["closing_odds"] == "closing_lines"
    assert row["field_sources"]["clv_points"].startswith("computed:")


def test_clv_bet_without_closing_line_excluded_but_counted(tmp_path):
    def setup(store):
        _seed_bet(store, "led-with", "bet-with", -150, closing=-200)
        _seed_bet(store, "led-without", "bet-without", -110, closing=None)

    body = _client(tmp_path, setup=setup).get("/api/clv").json()
    assert body["summary"]["bets_scanned"] == 2
    assert body["summary"]["with_closing_line"] == 1
    assert {r["ledger_id"] for r in body["rows"]} == {"led-with"}


def test_clv_empty_state(tmp_path):
    body = _client(tmp_path).get("/api/clv").json()
    assert body["rows"] == []
    assert body["summary"]["bets_scanned"] == 0
    assert body["summary"]["avg_clv_points"] is None


def test_clv_page_renders(tmp_path):
    client = _client(tmp_path, setup=lambda s: _seed_bet(s, "led-clv", "bet-1", -150, closing=-200))
    html = client.get("/clv").text
    assert "<h1>Closing-Line Value</h1>" in html
    assert "led-clv" in html
    assert "beat the close" in html  # summary tile
