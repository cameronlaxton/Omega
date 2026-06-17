"""Tests for the portfolio-plane + game_context MCP tools."""

from __future__ import annotations

import tempfile

import pytest

import omega.integrations.game_context as gc_mod
from omega.mcp.server import (
    omega_get_game_context,
    omega_get_portfolio_summary,
    omega_record_flat_bet,
)
from omega.trace.store import TraceStore


@pytest.fixture
def db_path(monkeypatch):
    # Keep the ledger empty until the tools write to it.
    monkeypatch.setenv("OMEGA_BET_LEDGER_AUTOLOG", "0")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = TraceStore(db_path=tmp.name)
    store.persist(
        {
            "trace_id": "g1",
            "run_id": "r",
            "timestamp": "2026-05-01T00:00:00Z",
            "prompt": "p",
            "league": "NBA",
            "matchup": "Pacers @ Celtics",
            "execution_mode": "native_sim",
            "kind": "game",
            "result": {"status": "success", "edges": [], "best_bet": None},
        }
    )
    store.close()
    return tmp.name


def test_record_flat_bet_writes_pending_ledger_row(db_path):
    res = omega_record_flat_bet(
        trace_id="g1",
        market="moneyline",
        side="home",
        odds=-110,
        db_path=db_path,
    )
    assert res["status"] == "success"
    assert res["already_existed"] is False
    bet = res["bet"]
    assert bet["selection_descriptor"] == "home_moneyline"
    assert bet["status"] == "pending"
    assert bet["stake_amount"] == 25.0
    assert bet["provenance"] == "user_confirmed"
    assert bet["bookmaker"] == "betmgm"


def test_record_flat_bet_is_idempotent(db_path):
    first = omega_record_flat_bet(
        trace_id="g1", market="moneyline", side="home", odds=-110, db_path=db_path
    )
    second = omega_record_flat_bet(
        trace_id="g1", market="moneyline", side="home", odds=-110, db_path=db_path
    )
    assert second["already_existed"] is True
    assert second["ledger_id"] == first["ledger_id"]


def test_record_flat_bet_missing_trace_errors(db_path):
    res = omega_record_flat_bet(
        trace_id="does-not-exist",
        market="moneyline",
        side="home",
        odds=-110,
        db_path=db_path,
    )
    assert res["status"] == "error"
    assert res["error_code"] == "trace_not_found"


def test_record_flat_bet_rejects_bad_side(db_path):
    res = omega_record_flat_bet(
        trace_id="g1", market="moneyline", side="middle", odds=-110, db_path=db_path
    )
    assert res["status"] == "error"
    assert res["error_code"] == "invalid_request"


def test_portfolio_summary_reflects_pending_then_graded(db_path):
    omega_record_flat_bet(
        trace_id="g1", market="moneyline", side="home", odds=-110, db_path=db_path
    )

    pending = omega_get_portfolio_summary(db_path=db_path)
    assert pending["status"] == "success"
    s = pending["summary"]
    assert s["total_bets"] == 1
    assert s["pending_count"] == 1
    assert s["pending_stake"] == 25.0
    assert s["current_bankroll"] == 1000.0
    assert s["net_pnl"] == 0.0

    # Settle the bet: attach a home win + settle the pending ledger row.
    from omega.trace.ledger_settlement import settle_pending_ledger

    store = TraceStore(db_path=db_path)
    store.attach_outcome(trace_id="g1", home_score=110, away_score=104)
    settle_pending_ledger(store, apply=True)
    store.close()

    graded = omega_get_portfolio_summary(db_path=db_path)["summary"]
    assert graded["won"] == 1
    assert graded["pending_count"] == 0
    assert graded["net_pnl"] > 0
    assert graded["current_bankroll"] > 1000.0
    assert graded["roi_pct"] > 0


def test_game_context_tool_delegates(monkeypatch):
    captured = {}

    def fake_resolve(**kwargs):
        captured.update(kwargs)
        return {"status": "success", "game_context": {"rest_days": 2}}

    monkeypatch.setattr(gc_mod, "resolve_game_context", fake_resolve)

    res = omega_get_game_context(
        league="NBA",
        home_team="Boston Celtics",
        away_team="Indiana Pacers",
        game_date="2026-05-10",
    )
    assert res["status"] == "success"
    assert res["result"]["game_context"]["rest_days"] == 2
    assert captured["league"] == "NBA"


def test_game_context_tool_validation_error():
    res = omega_get_game_context(
        league="NBA", home_team="", away_team="", game_date="2026-05-10", lookback_days=99
    )
    assert res["status"] == "error"
    assert res["error_code"] == "invalid_request"
