"""Schema + CRUD tests for the V13 bet_ledger table."""

from __future__ import annotations

import tempfile

from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.schema import SCHEMA_V2
from omega.trace.store import TraceStore


def _tmp_store() -> TraceStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return TraceStore(db_path=tmp.name)


def _persist_trace(store: TraceStore, trace_id: str = "t-1") -> None:
    store.persist(
        {
            "trace_id": trace_id,
            "run_id": "r",
            "timestamp": "2026-05-01T00:00:00Z",
            "prompt": "p",
            "league": "NBA",
            "matchup": "A @ B",
            "execution_mode": "native_sim",
            "kind": "game",
            "result": {"status": "success"},
        }
    )


def _ledger_bet(trace_id: str = "t-1", descriptor: str = "home_spread_-3.5") -> LedgerBet:
    return LedgerBet(
        ledger_id="led-1",
        trace_id=trace_id,
        bet_date="2026-05-01",
        league="NBA",
        sport="basketball",
        matchup="A @ B",
        market="spread",
        selection="B -3.5",
        selection_descriptor=descriptor,
        line=-3.5,
        odds=-110,
        provenance=BetProvenance.BACKFILL,
        decision_timestamp="2026-05-01T00:00:00Z",
    )


class TestMigration:
    def test_table_and_view_exist_at_v13(self):
        store = _tmp_store()
        try:
            tables = {
                r[0]
                for r in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            views = {
                r[0]
                for r in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'"
                ).fetchall()
            }
            assert "bet_ledger" in tables
            assert "v_bet_ledger_dashboard" in views
            version = store.conn.execute("SELECT MAX(version) FROM schema_versions").fetchone()[0]
            assert version >= 13
        finally:
            store.close()

    def test_reopen_is_idempotent(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        s1 = TraceStore(db_path=tmp.name)
        s1.close()
        s2 = TraceStore(db_path=tmp.name)
        try:
            rows = s2.conn.execute(
                "SELECT COUNT(*) FROM schema_versions WHERE version = 13"
            ).fetchone()[0]
            assert rows == 1
        finally:
            s2.close()

    def test_consolidation_migrates_legacy_bet_records_with_data(self):
        """The one irreversible step: a pre-existing bet_records row with real
        data must land in bet_ledger as provenance='user_confirmed', with its
        units stake converted to dollars and a settled status priced out, and
        bet_records must be gone afterwards. Fresh stores skip this path (they
        jump straight to V14 with no rows to migrate), so build the legacy state
        by hand and call the consolidation directly."""
        store = _tmp_store()
        try:
            _persist_trace(store)  # trace 't-1', league NBA
            # Re-create the legacy table the V14 drop removed and seed one
            # WON wager: 2 units on a $1000 bankroll => $20 stake; -110 odds.
            store.conn.executescript(SCHEMA_V2)
            # Any real pre-V14 DB reached V7 first, which adds session_id; the
            # consolidation query reads it, so reproduce that realistic shape.
            store.conn.execute("ALTER TABLE bet_records ADD COLUMN session_id TEXT")
            store.conn.execute(
                """INSERT INTO bet_records
                   (bet_id, trace_id, book, market, selection, selection_descriptor,
                    line_taken, odds_taken, stake_units, decision_timestamp, status,
                    session_id)
                   VALUES ('legacy-1', 't-1', 'betmgm', 'spread', 'B -3.5',
                           'home_spread_-3.5', -3.5, -110, 2.0,
                           '2026-05-01T00:00:00Z', 'won', 'sess-legacy')""",
            )
            store.conn.commit()

            migrated = store._consolidate_legacy_bet_records()
            assert migrated == 1

            rows = store.query_ledger(provenance="user_confirmed")
            assert len(rows) == 1
            row = rows[0]
            assert row["ledger_id"] == "legacy-1"  # bet_id preserved as ledger_id
            assert row["bookmaker"] == "betmgm"
            assert row["stake_amount"] == 20.0  # 2 units * (1000/100)
            assert row["status"] == "won"
            # WON at -110 on $20 => payout $38.18, net +$18.18.
            assert row["payout_amount"] == 38.18
            assert row["net_pnl"] == 18.18

            # bet_records is dropped, and re-running is a no-op (idempotent).
            gone = store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='bet_records'"
            ).fetchone()
            assert gone is None
            assert store._consolidate_legacy_bet_records() == 0
        finally:
            store.close()


class TestLedgerCrud:
    def test_record_and_get(self):
        store = _tmp_store()
        try:
            _persist_trace(store)
            store.record_ledger_bet(_ledger_bet())
            rows = store.get_ledger_bets("t-1")
            assert len(rows) == 1
            assert rows[0]["selection_descriptor"] == "home_spread_-3.5"
            assert rows[0]["status"] == "pending"
            assert rows[0]["provenance"] == "backfill"
        finally:
            store.close()

    def test_record_is_idempotent_on_unique_key(self):
        store = _tmp_store()
        try:
            _persist_trace(store)
            store.record_ledger_bet(_ledger_bet())
            dupe = _ledger_bet()
            dupe.ledger_id = "led-2"  # different id, same (trace, market, descriptor)
            store.record_ledger_bet(dupe)
            assert len(store.get_ledger_bets("t-1")) == 1
        finally:
            store.close()

    def test_grade_writes_money(self):
        store = _tmp_store()
        try:
            _persist_trace(store)
            store.record_ledger_bet(_ledger_bet())
            store.grade_ledger_bet("led-1", LedgerStatus.WON, 47.73, 22.73)
            row = store.get_ledger_bets("t-1")[0]
            assert row["status"] == "won"
            assert row["net_pnl"] == 22.73
            assert row["graded_at"] is not None
        finally:
            store.close()

    def test_record_requires_existing_trace(self):
        store = _tmp_store()
        try:
            try:
                store.record_ledger_bet(_ledger_bet(trace_id="missing"))
                assert False, "expected ValueError"
            except ValueError:
                pass
        finally:
            store.close()

    def test_query_ledger_filters(self):
        store = _tmp_store()
        try:
            _persist_trace(store)
            store.record_ledger_bet(_ledger_bet())
            assert len(store.query_ledger(league="NBA")) == 1
            assert len(store.query_ledger(league="MLB")) == 0
            assert len(store.query_ledger(provenance="backfill")) == 1
        finally:
            store.close()


def _bet_with(
    provenance: BetProvenance,
    *,
    odds: float = -110,
    bookmaker: str = "consensus",
    ledger_id: str = "led-x",
) -> LedgerBet:
    bet = _ledger_bet()
    bet.ledger_id = ledger_id
    bet.provenance = provenance
    bet.odds = odds
    bet.bookmaker = bookmaker
    return bet


class TestProvenanceUpgrade:
    """A user confirmation must take over the engine's auto-logged row for the
    same selection (they share the idempotency key) so the CLV pipeline, which
    reads provenance='user_confirmed', sees the real wager."""

    def test_user_confirmed_upgrades_pending_engine_auto_in_place(self):
        store = _tmp_store()
        try:
            _persist_trace(store)
            store.record_ledger_bet(
                _bet_with(
                    BetProvenance.ENGINE_AUTO, odds=-110, bookmaker="consensus", ledger_id="auto-1"
                )
            )
            # User confirms the same selection at a different price/book.
            store.record_ledger_bet(
                _bet_with(
                    BetProvenance.USER_CONFIRMED,
                    odds=-105,
                    bookmaker="draftkings",
                    ledger_id="user-1",
                )
            )
            rows = store.get_ledger_bets("t-1")
            assert len(rows) == 1  # still one row — upgraded in place
            row = rows[0]
            assert row["ledger_id"] == "auto-1"  # original row id is preserved
            assert row["provenance"] == "user_confirmed"
            assert row["odds"] == -105
            assert row["bookmaker"] == "draftkings"
            # And it is now visible to user_confirmed-only consumers.
            assert len(store.query_ledger(provenance="user_confirmed")) == 1
        finally:
            store.close()

    def test_engine_auto_does_not_downgrade_user_confirmed(self):
        store = _tmp_store()
        try:
            _persist_trace(store)
            store.record_ledger_bet(
                _bet_with(
                    BetProvenance.USER_CONFIRMED,
                    odds=-105,
                    bookmaker="draftkings",
                    ledger_id="user-1",
                )
            )
            store.record_ledger_bet(
                _bet_with(
                    BetProvenance.ENGINE_AUTO, odds=-110, bookmaker="consensus", ledger_id="auto-1"
                )
            )
            row = store.get_ledger_bets("t-1")[0]
            assert row["provenance"] == "user_confirmed"
            assert row["odds"] == -105  # untouched
            assert row["bookmaker"] == "draftkings"
        finally:
            store.close()

    def test_graded_row_is_not_clobbered_by_upgrade(self):
        store = _tmp_store()
        try:
            _persist_trace(store)
            store.record_ledger_bet(
                _bet_with(BetProvenance.ENGINE_AUTO, odds=-110, ledger_id="auto-1")
            )
            store.grade_ledger_bet("auto-1", LedgerStatus.WON, 47.73, 22.73)
            # A late user confirmation must NOT silently invalidate a settled grade.
            store.record_ledger_bet(
                _bet_with(BetProvenance.USER_CONFIRMED, odds=-105, ledger_id="user-1")
            )
            row = store.get_ledger_bets("t-1")[0]
            assert row["status"] == "won"
            assert row["odds"] == -110  # guard held: settled row untouched
            assert row["provenance"] == "engine_auto"
        finally:
            store.close()
