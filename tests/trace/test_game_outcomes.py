"""
Unit tests for TraceStore.attach_outcome — game-plane grading.

Parallels tests/trace/test_prop_outcomes.py but for game (moneyline/spread)
outcomes, including the 3-way draw result used by soccer/regulation-draw sports.

Covers:
- home_win / away_win / draw result derivation from scores
- outcome row is created post-persistence (separate table)
- re-attaching the same trace is rejected (must delete first)
- attaching to a non-existent trace raises ValueError
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.trace.store import TraceStore  # noqa: E402


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _game_trace(trace_id: str, league: str = "EPL") -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-17T16:00:00Z",
        "prompt": "game",
        "league": league,
        "matchup": "Chelsea @ Arsenal",
        "execution_mode": "sandbox_game",
        "kind": "game",
        "predictions": None,
        "recommendations": None,
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {"home_team": "Arsenal", "away_team": "Chelsea", "league": league},
        "result": {"status": "success"},
    }


def _row(store: TraceStore, trace_id: str):
    return store.conn.execute(
        "SELECT home_score, away_score, result, source FROM outcomes WHERE trace_id = ?",
        (trace_id,),
    ).fetchone()


class TestAttachOutcomeResult:
    def test_home_win(self):
        store = TraceStore(db_path=_tmp_db())
        store.persist(_game_trace("g-home"))
        store.attach_outcome("g-home", home_score=2, away_score=1)
        assert _row(store, "g-home")["result"] == "home_win"
        store.close()

    def test_away_win(self):
        store = TraceStore(db_path=_tmp_db())
        store.persist(_game_trace("g-away"))
        store.attach_outcome("g-away", home_score=0, away_score=3)
        assert _row(store, "g-away")["result"] == "away_win"
        store.close()

    def test_draw(self):
        store = TraceStore(db_path=_tmp_db())
        store.persist(_game_trace("g-draw"))
        store.attach_outcome("g-draw", home_score=1, away_score=1)
        row = _row(store, "g-draw")
        assert row["result"] == "draw"
        assert row["home_score"] == row["away_score"] == 1
        store.close()

    def test_source_recorded(self):
        store = TraceStore(db_path=_tmp_db())
        store.persist(_game_trace("g-src"))
        store.attach_outcome("g-src", home_score=2, away_score=1, source="api:espn")
        assert _row(store, "g-src")["source"] == "api:espn"
        store.close()


class TestAttachOutcomeGuards:
    def test_reattach_rejected(self):
        store = TraceStore(db_path=_tmp_db())
        store.persist(_game_trace("g-dup"))
        store.attach_outcome("g-dup", home_score=2, away_score=1)
        with pytest.raises(ValueError):
            store.attach_outcome("g-dup", home_score=3, away_score=0)
        store.close()

    def test_missing_trace_rejected(self):
        store = TraceStore(db_path=_tmp_db())
        with pytest.raises(ValueError):
            store.attach_outcome("does-not-exist", home_score=1, away_score=0)
        store.close()

    def test_outcome_is_separate_from_trace(self):
        # Outcome attachment must not mutate the source trace row; it lives in
        # its own table and is queryable independently.
        store = TraceStore(db_path=_tmp_db())
        store.persist(_game_trace("g-sep"))
        assert _row(store, "g-sep") is None  # no outcome before attach
        store.attach_outcome("g-sep", home_score=2, away_score=1)
        assert _row(store, "g-sep") is not None
        store.close()
