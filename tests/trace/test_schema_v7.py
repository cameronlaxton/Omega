"""
Tests for V7 schema migration:
- bet_records.session_id column added and indexed
- backfill from traces.session_id via trace_id join
- record_bet() populates session_id from the linked trace
- BUG-3 cleanup removes binary 1/0 game-outcome rows attached to prop-kind traces
- migration is idempotent (safe to re-run)
"""

from __future__ import annotations

import tempfile
import uuid

from omega.trace.bet_record import BetRecord, BetStatus
from omega.trace.schema import CURRENT_VERSION, apply_v7_migration, apply_v8_migration
from omega.trace.store import TraceStore


def _tmp_db_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _persist_trace(
    store: TraceStore, trace_id: str, *, session_id: str | None, kind: str = "prop"
) -> None:
    blob = {
        "trace_id": trace_id,
        "run_id": "r-" + trace_id,
        "timestamp": "2026-05-17T12:00:00Z",
        "prompt": "x",
        "league": "NBA",
        "matchup": f"home @ away ({trace_id})",
        "execution_mode": "native_sim",
        "session_id": session_id,
        "kind": kind,
    }
    store.persist(blob)


def _make_bet(trace_id: str, descriptor: str) -> BetRecord:
    return BetRecord(
        bet_id=uuid.uuid4().hex[:12],
        trace_id=trace_id,
        book="DraftKings",
        market="player_prop:pts",
        selection=f"Test selection {descriptor}",
        selection_descriptor=descriptor,
        line_taken=20.5,
        odds_taken=-110,
        stake_units=1.0,
        decision_timestamp="2026-05-17T15:00:00Z",
        status=BetStatus.PENDING,
    )


class TestSchemaVersionAndLedger:
    def test_current_version_is_eighteen(self):
        store = TraceStore(db_path=_tmp_db_path())
        assert CURRENT_VERSION == 18
        assert store.schema_version() == 18
        store.close()

    def test_bet_records_table_dropped_at_v14(self):
        store = TraceStore(db_path=_tmp_db_path())
        tbl = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bet_records'"
        ).fetchone()
        assert tbl is None  # consolidated into bet_ledger
        store.close()

    def test_bet_ledger_has_session_id_column(self):
        store = TraceStore(db_path=_tmp_db_path())
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(bet_ledger)").fetchall()}
        assert "session_id" in cols
        store.close()


class TestRecordBetPopulatesSessionId:
    """record_bet() now writes a user_confirmed row into bet_ledger; session_id
    is still sourced from the linked trace."""

    def test_session_id_pulled_from_trace(self):
        store = TraceStore(db_path=_tmp_db_path())
        _persist_trace(store, "t-sess-1", session_id="sess-20260517-abcd")
        store.record_bet(_make_bet("t-sess-1", "tatum_over_27.5_pts"))

        row = store.conn.execute(
            "SELECT session_id FROM bet_ledger WHERE trace_id = 't-sess-1' "
            "AND provenance = 'user_confirmed'"
        ).fetchone()
        assert row["session_id"] == "sess-20260517-abcd"
        store.close()

    def test_session_id_null_when_trace_has_none(self):
        store = TraceStore(db_path=_tmp_db_path())
        _persist_trace(store, "t-null-sess", session_id=None)
        store.record_bet(_make_bet("t-null-sess", "x_over_1.5"))

        row = store.conn.execute(
            "SELECT session_id FROM bet_ledger WHERE trace_id = 't-null-sess' "
            "AND provenance = 'user_confirmed'"
        ).fetchone()
        assert row["session_id"] is None
        store.close()


class TestBug3Cleanup:
    def test_removes_binary_outcomes_on_prop_traces(self):
        store = TraceStore(db_path=_tmp_db_path())
        _persist_trace(store, "t-prop-bad", session_id=None, kind="prop")
        _persist_trace(store, "t-game-ok", session_id=None, kind="game")

        # Bad: 1-0 placeholder attached to a prop-kind trace
        store.attach_outcome("t-prop-bad", home_score=1, away_score=0, source="manual:bad")
        # Legit: a real 1-0 baseball game on a game-kind trace must NOT be touched
        store.attach_outcome("t-game-ok", home_score=1, away_score=0, source="manual:legit")

        deleted = apply_v7_migration(store.conn)
        assert deleted == 1

        # Bad row gone
        remaining_bad = store.conn.execute(
            "SELECT COUNT(*) AS c FROM outcomes WHERE trace_id = 't-prop-bad'"
        ).fetchone()["c"]
        assert remaining_bad == 0

        # Game row preserved
        remaining_game = store.conn.execute(
            "SELECT COUNT(*) AS c FROM outcomes WHERE trace_id = 't-game-ok'"
        ).fetchone()["c"]
        assert remaining_game == 1

        store.close()

    def test_preserves_real_scores_on_prop_traces(self):
        """A prop trace with a real score (5-3) is NOT a binary placeholder;
        cleanup must not touch it even though it's incorrectly attached."""
        store = TraceStore(db_path=_tmp_db_path())
        _persist_trace(store, "t-prop-real-score", session_id=None, kind="prop")
        store.attach_outcome("t-prop-real-score", home_score=5, away_score=3, source="manual:legit")

        deleted = apply_v7_migration(store.conn)
        assert deleted == 0

        remaining = store.conn.execute(
            "SELECT COUNT(*) AS c FROM outcomes WHERE trace_id = 't-prop-real-score'"
        ).fetchone()["c"]
        assert remaining == 1
        store.close()


class TestIdempotency:
    def test_v7_migration_safe_to_rerun(self):
        store = TraceStore(db_path=_tmp_db_path())
        _persist_trace(store, "t-idem", session_id="sess-20260517-xxxx")
        store.record_bet(_make_bet("t-idem", "x_over_1.5"))

        # Running V7 again should not error (bet_records is gone; only the
        # BUG-3 outcome cleanup runs) and should not double-count.
        for _ in range(3):
            apply_v7_migration(store.conn)

        row = store.conn.execute(
            "SELECT session_id FROM bet_ledger WHERE trace_id = 't-idem' "
            "AND provenance = 'user_confirmed'"
        ).fetchone()
        assert row["session_id"] == "sess-20260517-xxxx"

        # Schema version stays current
        assert store.schema_version() == CURRENT_VERSION
        store.close()


class TestV8OutcomeUniqueness:
    def test_duplicate_game_outcomes_are_collapsed_before_unique_index(self):
        store = TraceStore(db_path=_tmp_db_path())
        _persist_trace(store, "t-dup-outcome", session_id=None, kind="game")

        store.conn.execute("DROP INDEX IF EXISTS idx_outcomes_trace_id_unique")
        store.conn.execute(
            """INSERT INTO outcomes
               (outcome_id, trace_id, home_score, away_score, result, source)
               VALUES
               ('o-first', 't-dup-outcome', 100, 90, 'home_win', 'test'),
               ('o-second', 't-dup-outcome', 80, 95, 'away_win', 'test')"""
        )
        store.conn.commit()

        deleted = apply_v8_migration(store.conn)

        assert deleted == 1
        rows = store.conn.execute(
            "SELECT outcome_id, result FROM outcomes WHERE trace_id = 't-dup-outcome'"
        ).fetchall()
        assert [dict(row) for row in rows] == [{"outcome_id": "o-first", "result": "home_win"}]
        idx = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_outcomes_trace_id_unique'"
        ).fetchone()
        assert idx is not None
        store.close()

    def test_reopening_store_is_no_op(self):
        path = _tmp_db_path()
        store_a = TraceStore(db_path=path)
        _persist_trace(store_a, "t-reopen", session_id="sess-x")
        store_a.close()

        store_b = TraceStore(db_path=path)
        assert store_b.schema_version() == CURRENT_VERSION
        store_b.close()
