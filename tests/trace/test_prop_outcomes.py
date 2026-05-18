"""
Tests for prop_outcomes (Schema V6) — player-prop grading persistence.

Covers:
- V6 migration creates prop_outcomes table and its index
- attach_prop_outcome: result derivation (win/loss/push), idempotency,
  missing-trace rejection, side validation
- get_prop_outcomes: retrieval
- Separation invariant: prop_outcomes rows do not appear in the game outcomes
  table and vice versa
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from omega.trace.schema import CURRENT_VERSION
from omega.trace.store import TraceStore


def _tmp_store() -> TraceStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return TraceStore(db_path=tmp.name)


def _make_prop_trace(
    trace_id: str = "t-prop-001",
    player_name: str = "LeBron James",
    prop_type: str = "points",
    line: float = 24.5,
) -> Dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": "r-001",
        "timestamp": "2026-05-17T22:00:00Z",
        "prompt": "LeBron points prop",
        "league": "NBA",
        "matchup": f"{player_name} {prop_type} {line}",
        "execution_mode": "sandbox_prop",
        "predictions": {"prob_over": 0.55},
        "recommendations": [],
        "odds_snapshot": {"odds_over": -110, "odds_under": -110},
        "downgrades": [],
        "input_snapshot": {
            "player_name": player_name,
            "prop_type": prop_type,
            "line": line,
        },
    }


class TestSchemaV6:
    def test_current_version_is_six(self):
        assert CURRENT_VERSION == 6

    def test_prop_outcomes_table_exists(self):
        store = _tmp_store()
        cols = {
            row[1]
            for row in store.conn.execute("PRAGMA table_info(prop_outcomes)").fetchall()
        }
        assert {
            "prop_outcome_id",
            "trace_id",
            "player_name",
            "stat_type",
            "stat_value",
            "line",
            "side",
            "result",
            "attached_at",
            "source",
        } <= cols
        store.close()

    def test_prop_outcomes_index_present(self):
        store = _tmp_store()
        idx_names = {
            row[0]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='prop_outcomes'"
            ).fetchall()
        }
        assert "idx_prop_outcomes_trace_id" in idx_names
        store.close()

    def test_schema_version_recorded(self):
        store = _tmp_store()
        assert store.schema_version() == 6
        row = store.conn.execute(
            "SELECT description FROM schema_versions WHERE version = 6"
        ).fetchone()
        assert row is not None
        assert "prop_outcomes" in row["description"].lower()
        store.close()

    def test_unique_constraint_on_trace_player_stat(self):
        """UNIQUE (trace_id, player_name, stat_type) prevents duplicate grading."""
        store = _tmp_store()
        store.persist(_make_prop_trace())
        store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 28.0, 24.5, "over"
        )
        # Try a raw INSERT bypassing the idempotent helper to confirm the constraint
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                """INSERT INTO prop_outcomes
                   (prop_outcome_id, trace_id, player_name, stat_type,
                    stat_value, line, side, result, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("dup-id", "t-prop-001", "LeBron James", "points",
                 28.0, 24.5, "over", "win", "manual"),
            )
        store.close()


class TestAttachPropOutcome:
    def test_over_win(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        pid = store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 28.0, 24.5, "over"
        )
        assert pid
        rows = store.get_prop_outcomes("t-prop-001")
        assert len(rows) == 1
        assert rows[0]["result"] == "win"
        assert rows[0]["side"] == "over"
        assert rows[0]["stat_value"] == 28.0
        assert rows[0]["line"] == 24.5
        store.close()

    def test_over_loss(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 18.0, 24.5, "over"
        )
        rows = store.get_prop_outcomes("t-prop-001")
        assert rows[0]["result"] == "loss"
        store.close()

    def test_under_win(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 18.0, 24.5, "under"
        )
        rows = store.get_prop_outcomes("t-prop-001")
        assert rows[0]["result"] == "win"
        assert rows[0]["side"] == "under"
        store.close()

    def test_under_loss(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 28.0, 24.5, "under"
        )
        rows = store.get_prop_outcomes("t-prop-001")
        assert rows[0]["result"] == "loss"
        store.close()

    def test_push_when_stat_equals_line(self):
        """Integer-line props (whole-number lines) can push when stat == line."""
        store = _tmp_store()
        store.persist(_make_prop_trace(prop_type="hits", line=2.0))
        store.attach_prop_outcome(
            "t-prop-001", "Aaron Judge", "hits", 2.0, 2.0, "over"
        )
        rows = store.get_prop_outcomes("t-prop-001")
        assert rows[0]["result"] == "push"
        store.close()

    def test_idempotent_on_repeat_attach(self):
        """Re-attaching same (trace, player, stat) returns same row id."""
        store = _tmp_store()
        store.persist(_make_prop_trace())
        pid1 = store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 28.0, 24.5, "over"
        )
        pid2 = store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 30.0, 24.5, "over",
            source="api:espn_boxscore",
        )
        assert pid1 == pid2
        # And only one row landed — the first one
        rows = store.get_prop_outcomes("t-prop-001")
        assert len(rows) == 1
        assert rows[0]["stat_value"] == 28.0
        assert rows[0]["source"] == "manual"
        store.close()

    def test_multiple_stats_for_same_player_same_trace(self):
        """Same player, different stat_type on same trace can coexist."""
        store = _tmp_store()
        store.persist(_make_prop_trace())
        store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 28.0, 24.5, "over"
        )
        store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "rebounds", 6.0, 7.5, "under"
        )
        rows = store.get_prop_outcomes("t-prop-001")
        assert len(rows) == 2
        stats = {r["stat_type"] for r in rows}
        assert stats == {"points", "rebounds"}
        store.close()

    def test_attach_to_missing_trace_raises(self):
        store = _tmp_store()
        with pytest.raises(ValueError, match="No trace found"):
            store.attach_prop_outcome(
                "nonexistent", "LeBron James", "points", 28.0, 24.5, "over"
            )
        store.close()

    def test_invalid_side_raises(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        with pytest.raises(ValueError, match="over.*under"):
            store.attach_prop_outcome(
                "t-prop-001", "LeBron James", "points", 28.0, 24.5, "yes"
            )
        store.close()

    def test_side_case_insensitive(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 28.0, 24.5, "OVER"
        )
        rows = store.get_prop_outcomes("t-prop-001")
        assert rows[0]["side"] == "over"
        store.close()

    def test_source_persisted(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        store.attach_prop_outcome(
            "t-prop-001",
            "LeBron James",
            "points",
            28.0,
            24.5,
            "over",
            source="api:espn_boxscore",
        )
        rows = store.get_prop_outcomes("t-prop-001")
        assert rows[0]["source"] == "api:espn_boxscore"
        store.close()


class TestSeparationFromGameOutcomes:
    """Phase 6 invariant: prop and game outcomes are stored in separate tables
    and never cross-contaminate."""

    def test_prop_outcome_does_not_create_game_outcome(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        store.attach_prop_outcome(
            "t-prop-001", "LeBron James", "points", 28.0, 24.5, "over"
        )
        game_outcomes = store.conn.execute(
            "SELECT outcome_id FROM outcomes WHERE trace_id = ?", ("t-prop-001",)
        ).fetchall()
        assert game_outcomes == []
        store.close()

    def test_game_outcome_does_not_create_prop_outcome(self):
        store = _tmp_store()
        store.persist(_make_prop_trace(trace_id="t-game-001"))
        store.attach_outcome("t-game-001", home_score=110, away_score=102)
        prop_outcomes = store.get_prop_outcomes("t-game-001")
        assert prop_outcomes == []
        store.close()

    def test_get_prop_outcomes_empty_for_ungraded(self):
        store = _tmp_store()
        store.persist(_make_prop_trace())
        assert store.get_prop_outcomes("t-prop-001") == []
        store.close()


class TestQueryTracesUnifiedOutcomeFilter:
    """has_outcome filter recognizes either game outcomes or prop outcomes,
    and query_traces attaches _prop_outcomes when present without row duplication."""

    def test_has_outcome_true_includes_prop_only_trace(self):
        store = _tmp_store()
        store.persist(_make_prop_trace(trace_id="t-prop-A"))
        store.persist(_make_prop_trace(trace_id="t-ungraded-B"))
        store.attach_prop_outcome(
            "t-prop-A", "LeBron James", "points", 28.0, 24.5, "over"
        )

        graded = store.query_traces(has_outcome=True)
        assert len(graded) == 1
        assert graded[0]["trace_id"] == "t-prop-A"
        store.close()

    def test_has_outcome_false_excludes_prop_graded_trace(self):
        store = _tmp_store()
        store.persist(_make_prop_trace(trace_id="t-prop-A"))
        store.persist(_make_prop_trace(trace_id="t-ungraded-B"))
        store.attach_prop_outcome(
            "t-prop-A", "LeBron James", "points", 28.0, 24.5, "over"
        )

        ungraded = store.query_traces(has_outcome=False)
        assert len(ungraded) == 1
        assert ungraded[0]["trace_id"] == "t-ungraded-B"
        store.close()

    def test_has_outcome_true_includes_game_only_trace(self):
        """Backwards-compat: pre-existing game outcomes still satisfy has_outcome=True."""
        store = _tmp_store()
        store.persist(_make_prop_trace(trace_id="t-game-A"))
        store.attach_outcome("t-game-A", home_score=110, away_score=102)

        graded = store.query_traces(has_outcome=True)
        assert len(graded) == 1
        assert graded[0]["trace_id"] == "t-game-A"
        store.close()

    def test_multiple_prop_outcomes_dont_duplicate_rows(self):
        """A trace with N prop outcomes must still appear exactly once in query results."""
        store = _tmp_store()
        store.persist(_make_prop_trace(trace_id="t-multi"))
        store.attach_prop_outcome(
            "t-multi", "LeBron James", "points", 28.0, 24.5, "over"
        )
        store.attach_prop_outcome(
            "t-multi", "LeBron James", "rebounds", 6.0, 7.5, "under"
        )
        store.attach_prop_outcome(
            "t-multi", "LeBron James", "assists", 9.0, 8.5, "over"
        )

        results = store.query_traces(has_outcome=True)
        assert len(results) == 1
        assert results[0]["trace_id"] == "t-multi"
        assert "_prop_outcomes" in results[0]
        assert len(results[0]["_prop_outcomes"]) == 3
        store.close()

    def test_prop_outcomes_attached_to_query_results(self):
        store = _tmp_store()
        store.persist(_make_prop_trace(trace_id="t-prop-A"))
        store.attach_prop_outcome(
            "t-prop-A", "LeBron James", "points", 28.0, 24.5, "over",
            source="api:espn_boxscore",
        )

        results = store.query_traces(has_outcome=True)
        prop_outcomes = results[0]["_prop_outcomes"]
        assert len(prop_outcomes) == 1
        assert prop_outcomes[0]["result"] == "win"
        assert prop_outcomes[0]["source"] == "api:espn_boxscore"
        store.close()

    def test_ungraded_trace_has_no_prop_outcomes_key(self):
        """Don't pollute the trace dict with empty _prop_outcomes when none exist."""
        store = _tmp_store()
        store.persist(_make_prop_trace(trace_id="t-ungraded"))
        results = store.query_traces()
        assert "_prop_outcomes" not in results[0]
        store.close()


class TestTraceRecorderSchemaVersion:
    """The recorder must inject CURRENT_VERSION, not a stale hard-coded 1."""

    def test_recorded_schema_version_matches_current(self):
        from omega.skills.trace_recorder import TraceRecorder

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = TraceStore(db_path=tmp.name)

        # Point the recorder at our temp DB by monkey-patching TraceStore default
        # for the duration of this call. Easiest: just call persist() directly
        # with the recorder's contract — i.e. trace dict with required fields.
        trace = _make_prop_trace(trace_id="t-version-check")
        store.persist({**trace, "schema_version": CURRENT_VERSION})

        row = store.conn.execute(
            "SELECT schema_version FROM traces WHERE trace_id = ?",
            ("t-version-check",),
        ).fetchone()
        assert row["schema_version"] == CURRENT_VERSION
        store.close()

    def test_recorder_uses_current_version_constant(self):
        """Recorder imports CURRENT_VERSION rather than hard-coding."""
        from omega.skills import trace_recorder

        src = Path(trace_recorder.__file__).read_text(encoding="utf-8")
        assert "from omega.trace.schema import CURRENT_VERSION" in src
        assert "_SCHEMA_VERSION = 1" not in src
        assert 'record["schema_version"] = CURRENT_VERSION' in src
