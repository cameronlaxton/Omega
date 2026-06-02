"""Tests for pending-only bet_ledger settlement orchestration."""

from __future__ import annotations

import tempfile

import pytest

from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.ledger_settlement import settle_pending_ledger
from omega.trace.store import TraceStore


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("OMEGA_BET_LEDGER_AUTOLOG", "0")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    s = TraceStore(db_path=tmp.name)
    yield s
    s.close()


def _persist_trace(store: TraceStore, trace_id: str, *, league: str = "NBA") -> None:
    store.persist(
        {
            "trace_id": trace_id,
            "run_id": "r",
            "timestamp": "2026-05-01T00:00:00Z",
            "prompt": "p",
            "league": league,
            "matchup": "A @ B",
            "execution_mode": "native_sim",
            "kind": "game",
            "result": {"status": "success"},
        }
    )


def _bet(
    trace_id: str,
    *,
    ledger_id: str,
    market: str = "moneyline",
    descriptor: str = "home_moneyline",
    line: float | None = None,
    odds: float = -110,
    provenance: BetProvenance = BetProvenance.USER_CONFIRMED,
    status: LedgerStatus = LedgerStatus.PENDING,
) -> LedgerBet:
    return LedgerBet(
        ledger_id=ledger_id,
        trace_id=trace_id,
        bet_date="2026-05-01",
        league="NBA",
        sport="basketball",
        matchup="A @ B",
        market=market,
        bookmaker="betmgm",
        selection=descriptor,
        selection_descriptor=descriptor,
        line=line,
        odds=odds,
        status=status,
        provenance=provenance,
        decision_timestamp="2026-05-01T00:00:00Z",
    )


def test_settles_only_pending_rows_at_query_level(store):
    _persist_trace(store, "pending")
    _persist_trace(store, "already-won")
    store.record_ledger_bet(_bet("pending", ledger_id="p1"))
    won = _bet("already-won", ledger_id="w1", status=LedgerStatus.WON)
    won.payout_amount = 47.73
    won.net_pnl = 22.73
    store.record_ledger_bet(won)
    store.attach_outcome("pending", home_score=110, away_score=104)
    store.attach_outcome("already-won", home_score=100, away_score=120)

    summary = settle_pending_ledger(store, apply=True)

    assert summary.pending_scanned == 1
    assert summary.settled["won"] == 1
    assert store.get_ledger_bets("pending")[0]["status"] == "won"
    already = store.get_ledger_bets("already-won")[0]
    assert already["status"] == "won"
    assert already["net_pnl"] == 22.73


def test_default_provenance_settles_user_confirmed_only(store):
    _persist_trace(store, "user")
    _persist_trace(store, "auto")
    store.record_ledger_bet(
        _bet("user", ledger_id="u1", provenance=BetProvenance.USER_CONFIRMED)
    )
    store.record_ledger_bet(
        _bet("auto", ledger_id="a1", provenance=BetProvenance.ENGINE_AUTO)
    )
    store.attach_outcome("user", home_score=110, away_score=104)
    store.attach_outcome("auto", home_score=110, away_score=104)

    summary = settle_pending_ledger(store, apply=True)

    assert summary.pending_scanned == 1
    assert store.get_ledger_bets("user")[0]["status"] == "won"
    assert store.get_ledger_bets("auto")[0]["status"] == "pending"


def test_all_provenance_can_settle_engine_auto(store):
    _persist_trace(store, "auto")
    store.record_ledger_bet(
        _bet("auto", ledger_id="a1", provenance=BetProvenance.ENGINE_AUTO)
    )
    store.attach_outcome("auto", home_score=110, away_score=104)

    summary = settle_pending_ledger(store, apply=True, provenance=None)

    assert summary.pending_scanned == 1
    assert store.get_ledger_bets("auto")[0]["status"] == "won"


def test_prop_and_push_settlement(store):
    _persist_trace(store, "prop")
    store.record_ledger_bet(
        _bet(
            "prop",
            ledger_id="prop1",
            market="player_prop:points",
            descriptor="jayson_tatum_over_27.5_points",
            line=27.5,
            odds=-115,
        )
    )
    store.attach_prop_outcome(
        "prop",
        player_name="Jayson Tatum",
        stat_type="points",
        stat_value=27.5,
        line=27.5,
        side="over",
    )

    summary = settle_pending_ledger(store, apply=True)

    assert summary.settled["push"] == 1
    row = store.get_ledger_bets("prop")[0]
    assert row["status"] == "push"
    assert row["net_pnl"] == 0.0


def test_ungradeable_rows_remain_pending(store):
    _persist_trace(store, "bad")
    store.record_ledger_bet(_bet("bad", ledger_id="bad1", descriptor="best_bet"))
    store.attach_outcome("bad", home_score=110, away_score=104)

    summary = settle_pending_ledger(store, apply=True)

    assert summary.pending_scanned == 1
    assert summary.ungradeable == 1
    assert store.get_ledger_bets("bad")[0]["status"] == "pending"
