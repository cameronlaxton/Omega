"""Tests for pending-only bet_ledger settlement orchestration."""

from __future__ import annotations

import datetime
import tempfile

import pytest

from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.ledger_settlement import auto_void_aged_pending, settle_pending_ledger
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
    decision_timestamp: str = "2026-05-01T00:00:00Z",
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
        decision_timestamp=decision_timestamp,
    )


def _days_ago(n: int) -> str:
    return (
        (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=n))
        .isoformat()
        .replace("+00:00", "Z")
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
    store.record_ledger_bet(_bet("user", ledger_id="u1", provenance=BetProvenance.USER_CONFIRMED))
    store.record_ledger_bet(_bet("auto", ledger_id="a1", provenance=BetProvenance.ENGINE_AUTO))
    store.attach_outcome("user", home_score=110, away_score=104)
    store.attach_outcome("auto", home_score=110, away_score=104)

    summary = settle_pending_ledger(store, apply=True)

    assert summary.pending_scanned == 1
    assert store.get_ledger_bets("user")[0]["status"] == "won"
    assert store.get_ledger_bets("auto")[0]["status"] == "pending"


def test_all_provenance_can_settle_engine_auto(store):
    _persist_trace(store, "auto")
    store.record_ledger_bet(_bet("auto", ledger_id="a1", provenance=BetProvenance.ENGINE_AUTO))
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


def test_dnp_void_prop_settles_as_void(store):
    """A DNP void prop outcome settles the ledger bet as VOID (stake returned,
    net 0) instead of mis-grading as a loss or remaining ungradeable."""
    _persist_trace(store, "dnp")
    store.record_ledger_bet(
        _bet(
            "dnp",
            ledger_id="dnp1",
            market="player_prop:hits",
            descriptor="aaron_judge_over_0.5_hits",
            line=0.5,
            odds=-150,
        )
    )
    store.attach_prop_outcome(
        "dnp",
        player_name="Aaron Judge",
        stat_type="hits",
        stat_value=0.0,
        line=0.0,
        side="over",
        void=True,
    )

    summary = settle_pending_ledger(store, apply=True)

    assert summary.settled["void"] == 1
    assert summary.ungradeable == 0
    row = store.get_ledger_bets("dnp")[0]
    assert row["status"] == "void"
    assert row["net_pnl"] == 0.0
    assert row["payout_amount"] == 25.0  # default stake returned


def test_ungradeable_rows_remain_pending(store):
    _persist_trace(store, "bad")
    store.record_ledger_bet(_bet("bad", ledger_id="bad1", descriptor="best_bet"))
    store.attach_outcome("bad", home_score=110, away_score=104)

    summary = settle_pending_ledger(store, apply=True)

    assert summary.pending_scanned == 1
    assert summary.ungradeable == 1
    assert store.get_ledger_bets("bad")[0]["status"] == "pending"


class TestAutoVoidAgedPending:
    def test_old_ungradeable_row_is_voided(self, store):
        _persist_trace(store, "old-stuck")
        store.record_ledger_bet(
            _bet("old-stuck", ledger_id="v1", decision_timestamp=_days_ago(30))
        )
        # No outcome attached -- genuinely ungradeable.

        summary = auto_void_aged_pending(store, older_than_days=14, apply=True)

        assert summary.settled["void"] == 1
        row = store.get_ledger_bets("old-stuck")[0]
        assert row["status"] == "void"
        assert row["net_pnl"] == 0.0
        assert row["payout_amount"] == row["stake_amount"]

    def test_recent_ungradeable_row_is_not_voided(self, store):
        _persist_trace(store, "recent-stuck")
        store.record_ledger_bet(
            _bet("recent-stuck", ledger_id="v2", decision_timestamp=_days_ago(2))
        )

        summary = auto_void_aged_pending(store, older_than_days=14, apply=True)

        assert summary.settled.get("void", 0) == 0
        assert store.get_ledger_bets("recent-stuck")[0]["status"] == "pending"

    def test_old_gradeable_row_is_settled_not_voided(self, store):
        """A row that CAN be graded (outcome exists) is settle_pending_ledger's
        job even if it's old -- auto_void_aged_pending must not steal it."""
        _persist_trace(store, "old-gradeable")
        store.record_ledger_bet(
            _bet("old-gradeable", ledger_id="v3", decision_timestamp=_days_ago(30))
        )
        store.attach_outcome("old-gradeable", home_score=110, away_score=104)

        void_summary = auto_void_aged_pending(store, older_than_days=14, apply=True)

        assert void_summary.settled.get("void", 0) == 0
        assert store.get_ledger_bets("old-gradeable")[0]["status"] == "pending"

        settle_summary = settle_pending_ledger(store, apply=True)
        assert settle_summary.settled["won"] == 1
        assert store.get_ledger_bets("old-gradeable")[0]["status"] == "won"

    def test_dry_run_does_not_write(self, store):
        _persist_trace(store, "dry-old")
        store.record_ledger_bet(_bet("dry-old", ledger_id="v4", decision_timestamp=_days_ago(30)))

        summary = auto_void_aged_pending(store, older_than_days=14, apply=False)

        assert summary.settled["void"] == 1  # reported
        assert store.get_ledger_bets("dry-old")[0]["status"] == "pending"  # not written
