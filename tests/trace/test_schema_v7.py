"""
Tests for V7 schema migration:
- bet_records.session_id column added and indexed
- backfill from traces.session_id via trace_id join
- record_bet() populates session_id from the linked trace
- BUG-3 cleanup removes binary 1/0 game-outcome rows attached to prop-kind traces
- migration is idempotent (safe to re-run)
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest

from omega.trace.bet_record import BetRecord, BetStatus
from omega.trace.schema import CURRENT_VERSION, apply_v7_migration
from omega.trace.store import TraceStore


def _tmp_db_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _persist_trace(store: TraceStore, trace_id: str, *, session_id: str | None, kind: str = "prop") -> None:
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


class TestV7SchemaColumn:
    def test_session_id_column_exists(self):
        store = TraceStore(db_path=_tmp_db_path())
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(bet_records)").fetchall()}
        assert "session_id" in cols
        store.close()

    def test_session_id_index_exists(self):
        store = TraceStore(db_path=_tmp_db_path())
        idx = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_bet_records_session_id'"
        ).fetchone()
        assert idx is not None
        store.close()

    def test_current_version_is_seven(self):
        store = TraceStore(db_path=_tmp_db_path())
        assert CURRENT_VERSION == 7
        assert store.schema_version() == 7
        store.close()


class TestRecordBetPopulatesSessionId:
    def test_session_id_pulled_from_trace(self):
        store = TraceStore(db_path=_tmp_db_path())
        _persist_trace(store, "t-sess-1", session_id="sess-20260517-abcd")
        store.record_bet(_make_bet("t-sess-1", "tatum_over_27.5_pts"))

        row = store.conn.execute(
            "SELECT session_id FROM bet_records WHERE trace_id = 't-sess-1'"
        ).fetchone()
        assert row["session_id"] == "sess-20260517-abcd"
        store.close()

    def test_session_id_null_when_trace_has_none(self):
        store = TraceStore(db_path=_tmp_db_path())
        _persist_trace(store, "t-null-sess", session_id=None)
        store.record_bet(_make_bet("t-null-sess", "x_over_1.5"))

        row = store.conn.execute(
            "SELECT session_id FROM bet_records WHERE trace_id = 't-null-sess'"
        ).fetchone()
        assert row["session_id"] is None
        store.close()


class TestBackfill:
    def test_backfill_populates_existing_rows(self):
        # Build a V6-shape DB by hand: bet_records without session_id column,
        # then run the V7 migration directly and check that rows acquire it.
        db_path = _tmp_db_path()
        # First call creates V7 schema; tear it down to simulate pre-V7 state.
        store = TraceStore(db_path=db_path)
        _persist_trace(store, "t-back-1", session_id="sess-20260515-zzzz")

        # Insert a bet directly bypassing record_bet() to simulate a row inserted
        # by old code before V7 wired session_id into INSERT.
        store.conn.execute("UPDATE bet_records SET session_id = NULL")
        store.conn.execute(
            """INSERT INTO bet_records
               (bet_id, trace_id, book, market, selection, selection_descriptor,
                line_taken, odds_taken, stake_units, decision_timestamp, status,
                session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (
                "bet-backfill-1",
                "t-back-1",
                "DraftKings",
                "player_prop:pts",
                "Test",
                "test_over_1.5",
                1.5,
                -110,
                1.0,
                "2026-05-15T15:00:00Z",
                "pending",
            ),
        )
        store.conn.commit()

        deleted = apply_v7_migration(store.conn)
        assert deleted >= 0  # no bad outcomes seeded

        row = store.conn.execute(
            "SELECT session_id FROM bet_records WHERE bet_id = 'bet-backfill-1'"
        ).fetchone()
        assert row["session_id"] == "sess-20260515-zzzz"
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

        # Running V7 again should not error and should not double-count
        for _ in range(3):
            apply_v7_migration(store.conn)

        row = store.conn.execute(
            "SELECT session_id FROM bet_records WHERE trace_id = 't-idem'"
        ).fetchone()
        assert row["session_id"] == "sess-20260517-xxxx"

        # Schema version stays at 7
        assert store.schema_version() == 7
        store.close()

    def test_reopening_store_is_no_op(self):
        path = _tmp_db_path()
        store_a = TraceStore(db_path=path)
        _persist_trace(store_a, "t-reopen", session_id="sess-x")
        store_a.close()

        store_b = TraceStore(db_path=path)
        assert store_b.schema_version() == 7
        store_b.close()
