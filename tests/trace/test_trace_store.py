"""
Tests for omega.trace.store — TraceStore SQLite persistence.

Covers:
- Schema creation and versioning
- Persist: round-trip, idempotent, required field validation
- Outcome attachment: link, result derivation, missing trace rejection
- Query: by league, time range, outcome status, execution mode
- Graded traces: join traces with outcomes for calibration
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from omega.trace.schema import CURRENT_VERSION
from omega.trace.store import TraceStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_store() -> TraceStore:
    """Create a TraceStore backed by a temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return TraceStore(db_path=tmp.name)


def _make_trace(
    trace_id: str = "t-001",
    run_id: str = "r-001",
    league: str = "NBA",
    matchup: str = "Celtics @ Lakers",
    execution_mode: str = "native_sim",
    simulation_seed: int = 12345,
    aggregate_quality: float = 0.85,
    prompt: str = "Lakers vs Celtics NBA",
    **overrides: Any,
) -> dict[str, Any]:
    base = {
        "trace_id": trace_id,
        "run_id": run_id,
        "timestamp": "2026-03-21T12:00:00Z",
        "prompt": prompt,
        "league": league,
        "matchup": matchup,
        "execution_mode": execution_mode,
        "simulation_seed": simulation_seed,
        "aggregate_quality": aggregate_quality,
        "predictions": {"home_win_prob": 0.58, "away_win_prob": 0.42},
        "recommendations": [{"side": "home", "edge_pct": 4.2, "units": 1.5}],
        "odds_snapshot": {"moneyline_home": -150, "moneyline_away": 130},
        "downgrades": [],
        "understanding": {"subjects": ["game"]},
        "stage_timings": {"understanding": 12.5, "execution": 450.2},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_version_recorded(self):
        store = _tmp_store()
        assert store.schema_version() == CURRENT_VERSION
        store.close()

    def test_schema_idempotent(self):
        """Calling _ensure_schema twice doesn't crash."""
        store = _tmp_store()
        store._ensure_schema()
        assert store.schema_version() == CURRENT_VERSION
        store.close()


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


class TestPersist:
    def test_round_trip(self):
        store = _tmp_store()
        trace = _make_trace()
        tid = store.persist(trace)
        assert tid == "t-001"

        retrieved = store.get_trace("t-001")
        assert retrieved is not None
        assert retrieved["trace_id"] == "t-001"
        assert retrieved["league"] == "NBA"
        assert retrieved["predictions"]["home_win_prob"] == 0.58
        assert retrieved["stage_timings"]["execution"] == 450.2
        store.close()

    def test_idempotent_persist(self):
        """Persisting the same trace_id twice does not error or duplicate."""
        store = _tmp_store()
        trace = _make_trace()
        store.persist(trace)
        store.persist(trace)  # should not raise
        assert store.count() == 1
        store.close()

    def test_missing_required_fields_raises(self):
        store = _tmp_store()
        with pytest.raises(ValueError, match="trace_id"):
            store.persist({"prompt": "test"})
        store.close()

    def test_persist_with_null_optional_fields(self):
        """Traces with null predictions/odds/etc persist correctly."""
        store = _tmp_store()
        trace = _make_trace(
            predictions=None,
            odds_snapshot=None,
            recommendations=[],
            simulation_seed=None,
        )
        store.persist(trace)
        retrieved = store.get_trace("t-001")
        assert retrieved["predictions"] is None
        assert retrieved["odds_snapshot"] is None
        store.close()

    def test_count(self):
        store = _tmp_store()
        assert store.count() == 0
        store.persist(_make_trace("t-001"))
        store.persist(_make_trace("t-002"))
        assert store.count() == 2
        store.close()


# ---------------------------------------------------------------------------
# Outcome attachment
# ---------------------------------------------------------------------------


class TestOutcomeAttachment:
    def test_attach_and_retrieve(self):
        store = _tmp_store()
        store.persist(_make_trace())
        oid = store.attach_outcome("t-001", home_score=112, away_score=105)
        assert oid  # non-empty

        graded = store.get_graded_traces()
        assert len(graded) == 1
        assert graded[0]["_outcome"]["home_score"] == 112
        assert graded[0]["_outcome"]["away_score"] == 105
        assert graded[0]["_outcome"]["result"] == "home_win"
        store.close()

    def test_away_win_result(self):
        store = _tmp_store()
        store.persist(_make_trace())
        store.attach_outcome("t-001", home_score=98, away_score=110)
        graded = store.get_graded_traces()
        assert graded[0]["_outcome"]["result"] == "away_win"
        store.close()

    def test_draw_result(self):
        store = _tmp_store()
        store.persist(_make_trace())
        store.attach_outcome("t-001", home_score=1, away_score=1)
        graded = store.get_graded_traces()
        assert graded[0]["_outcome"]["result"] == "draw"
        store.close()

    def test_attach_to_missing_trace_raises(self):
        store = _tmp_store()
        with pytest.raises(ValueError, match="No trace found"):
            store.attach_outcome("nonexistent", home_score=100, away_score=90)
        store.close()

    def test_second_game_outcome_for_same_trace_is_rejected(self):
        store = _tmp_store()
        store.persist(_make_trace())
        store.attach_outcome("t-001", home_score=112, away_score=105)

        with pytest.raises(ValueError, match="Outcome already attached"):
            store.attach_outcome("t-001", home_score=98, away_score=110)

        graded = store.get_graded_traces()
        assert len(graded) == 1
        assert graded[0]["_outcome"]["result"] == "home_win"
        store.close()

    def test_outcome_source(self):
        store = _tmp_store()
        store.persist(_make_trace())
        store.attach_outcome("t-001", home_score=100, away_score=95, source="api")
        # Source is stored but not exposed in the simple _outcome dict;
        # verify it doesn't crash with non-default source
        store.close()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:
    def _seed_store(self) -> TraceStore:
        store = _tmp_store()
        store.persist(
            _make_trace(
                "t-nba-1",
                league="NBA",
                execution_mode="native_sim",
                matchup="Celtics @ Lakers",
                timestamp="2026-03-20T10:00:00Z",
            )
        )
        store.persist(
            _make_trace(
                "t-nba-2",
                league="NBA",
                execution_mode="research",
                matchup="Heat @ Bucks",
                timestamp="2026-03-21T10:00:00Z",
            )
        )
        store.persist(
            _make_trace(
                "t-nfl-1",
                league="NFL",
                execution_mode="native_sim",
                matchup="Chiefs @ Bills",
                timestamp="2026-03-21T15:00:00Z",
            )
        )
        return store

    def test_query_all(self):
        store = self._seed_store()
        results = store.query_traces()
        assert len(results) == 3
        store.close()

    def test_query_by_league(self):
        store = self._seed_store()
        results = store.query_traces(league="NBA")
        assert len(results) == 2
        assert all(t["league"] == "NBA" for t in results)
        store.close()

    def test_query_by_execution_mode(self):
        store = self._seed_store()
        results = store.query_traces(execution_mode="native_sim")
        assert len(results) == 2
        store.close()

    def test_query_by_time_range(self):
        store = self._seed_store()
        results = store.query_traces(start="2026-03-21T00:00:00Z")
        assert len(results) == 2
        store.close()

    def test_query_has_outcome_filter(self):
        store = self._seed_store()
        store.attach_outcome("t-nba-1", home_score=110, away_score=102)

        graded = store.query_traces(has_outcome=True)
        assert len(graded) == 1
        assert graded[0]["trace_id"] == "t-nba-1"

        ungraded = store.query_traces(has_outcome=False)
        assert len(ungraded) == 2
        store.close()

    def test_query_limit(self):
        store = self._seed_store()
        results = store.query_traces(limit=1)
        assert len(results) == 1
        store.close()

    def test_get_nonexistent_trace(self):
        store = _tmp_store()
        assert store.get_trace("nonexistent") is None
        store.close()


# ---------------------------------------------------------------------------
# Graded traces (calibration input)
# ---------------------------------------------------------------------------


class TestGradedTraces:
    def test_graded_traces_joins_outcome(self):
        store = _tmp_store()
        store.persist(_make_trace("t-001"))
        store.persist(_make_trace("t-002"))
        store.attach_outcome("t-001", home_score=108, away_score=101)

        graded = store.get_graded_traces()
        assert len(graded) == 1
        assert graded[0]["trace_id"] == "t-001"
        assert "_outcome" in graded[0]
        store.close()

    def test_graded_traces_by_league(self):
        store = _tmp_store()
        store.persist(_make_trace("t-nba", league="NBA"))
        store.persist(_make_trace("t-nfl", league="NFL"))
        store.attach_outcome("t-nba", home_score=100, away_score=90)
        store.attach_outcome("t-nfl", home_score=21, away_score=17)

        nba_graded = store.get_graded_traces(league="NBA")
        assert len(nba_graded) == 1
        assert nba_graded[0]["league"] == "NBA"
        store.close()

    def test_empty_graded(self):
        store = _tmp_store()
        store.persist(_make_trace())
        assert store.get_graded_traces() == []
        store.close()


class TestSessionSummary:
    def test_prop_outcomes_count_as_graded_in_session_summary(self):
        store = _tmp_store()
        store.persist(
            _make_trace(
                trace_id="t-prop-session",
                execution_mode="sandbox_prop",
                predictions={"over_prob": 0.62, "under_prob": 0.38},
                session_id="sess-props",
                kind="prop",
            )
        )
        store.attach_prop_outcome(
            "t-prop-session",
            player_name="Test Player",
            stat_type="pts",
            stat_value=26,
            line=24.5,
            side="over",
        )

        summary = store.get_session_summary()

        assert summary == [
            {
                "session_id": "sess-props",
                "trace_count": 1,
                "graded_count": 1,
                "first_ts": "2026-03-21T12:00:00Z",
                "last_ts": "2026-03-21T12:00:00Z",
            }
        ]
        store.close()


# ---------------------------------------------------------------------------
# Schema V4 — session_id column
# ---------------------------------------------------------------------------


class TestSchemaV4SessionId:
    def test_session_id_column_present(self):
        store = _tmp_store()
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(traces)").fetchall()}
        assert "session_id" in cols
        store.close()

    def test_session_id_index_present(self):
        store = _tmp_store()
        idx_names = {
            row[0]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='traces'"
            ).fetchall()
        }
        assert "idx_traces_session_id" in idx_names
        store.close()

    def test_persist_with_session_id(self):
        store = _tmp_store()
        trace = _make_trace(trace_id="t-sess-1", session_id="sess-20260515-abc")
        store.persist(trace)
        row = store.conn.execute(
            "SELECT session_id FROM traces WHERE trace_id = ?",
            ("t-sess-1",),
        ).fetchone()
        assert row["session_id"] == "sess-20260515-abc"
        store.close()

    def test_persist_without_session_id_is_null(self):
        store = _tmp_store()
        store.persist(_make_trace(trace_id="t-no-sess"))
        row = store.conn.execute(
            "SELECT session_id FROM traces WHERE trace_id = ?", ("t-no-sess",)
        ).fetchone()
        assert row["session_id"] is None
        store.close()

    def test_v3_to_v4_migration_idempotent(self):
        """Simulate an existing V3 DB (no session_id column), then run V4 migration
        twice and confirm: column exists, legacy rows have NULL session_id, second
        run is a no-op."""
        import sqlite3
        import tempfile

        from omega.trace.schema import (
            SCHEMA_V1,
            SCHEMA_V2,
            SCHEMA_V3,
            apply_v4_migration,
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_V1)
        conn.executescript(SCHEMA_V2)
        conn.executescript(SCHEMA_V3)
        # Insert a legacy V3 row (no session_id column yet)
        conn.execute(
            """INSERT INTO traces (trace_id, run_id, timestamp, prompt, full_trace)
               VALUES (?, ?, ?, ?, ?)""",
            ("legacy-1", "run-legacy", "2026-01-01T00:00:00Z", "legacy", "{}"),
        )
        conn.commit()
        cols_before = {r[1] for r in conn.execute("PRAGMA table_info(traces)").fetchall()}
        assert "session_id" not in cols_before

        apply_v4_migration(conn)
        cols_after = {r[1] for r in conn.execute("PRAGMA table_info(traces)").fetchall()}
        assert "session_id" in cols_after

        # Legacy row carries NULL session_id
        row = conn.execute(
            "SELECT session_id FROM traces WHERE trace_id = ?", ("legacy-1",)
        ).fetchone()
        assert row["session_id"] is None

        # Re-running is a no-op (no error)
        apply_v4_migration(conn)
        conn.close()

    def test_get_session_summary(self):
        store = _tmp_store()
        store.persist(_make_trace(trace_id="t-a", session_id="sess-1"))
        store.persist(_make_trace(trace_id="t-b", session_id="sess-1"))
        store.persist(_make_trace(trace_id="t-c", session_id="sess-2"))
        store.persist(_make_trace(trace_id="t-d"))  # no session_id, excluded

        summary = store.get_session_summary()
        by_sess = {row["session_id"]: row for row in summary}
        assert set(by_sess) == {"sess-1", "sess-2"}
        assert by_sess["sess-1"]["trace_count"] == 2
        assert by_sess["sess-2"]["trace_count"] == 1
        store.close()
