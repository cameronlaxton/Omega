"""
End-to-end tests for omega-fetch-outcomes-nba â€” NBA game-outcome grading.

Hits a real SQLite DB but stubs the ESPN NBA scoreboard via the script's
``scoreboard_fetcher`` injection point â€” no network.

Covers: happy path, dry-run, prop traces untouched, idempotency, unmapped team.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch_outcomes_nba  # type: ignore  # noqa: E402

from omega.integrations.espn_nba import FinalGame  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _game_trace(
    trace_id: str,
    *,
    home_team: str = "Boston Celtics",
    away_team: str = "Miami Heat",
    timestamp: str = "2026-05-17T19:00:00Z",
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": timestamp,
        "prompt": "game",
        "league": "NBA",
        "matchup": f"{away_team} @ {home_team}",
        "execution_mode": "sandbox_game",
        "kind": "game",
        "predictions": None,
        "recommendations": None,
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {"home_team": home_team, "away_team": away_team, "league": "NBA"},
        "result": {"status": "success"},
    }


def _prop_trace(trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-17T19:00:00Z",
        "prompt": "prop",
        "league": "NBA",
        "matchup": "Miami Heat @ Boston Celtics",
        "execution_mode": "sandbox_prop",
        "kind": "prop",
        "predictions": {"over_prob": 0.55, "under_prob": 0.45},
        "recommendations": [],
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {
            "player_name": "Jayson Tatum",
            "league": "NBA",
            "prop_type": "pts",
            "line": 27.5,
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "game_date": "2026-05-17",
        },
        "result": {"status": "success", "recommendation": "over"},
    }


def _sb(games_by_date: dict[str, list[FinalGame]]):
    def _fetch(iso_date: str):
        return games_by_date.get(iso_date, [])

    return _fetch


def _final(home, away, hs, as_, d="2026-05-17") -> FinalGame:
    return FinalGame(
        event_id="NEV-1",
        date=d,
        home_team=home,
        away_team=away,
        home_score=hs,
        away_score=as_,
        status="final",
    )


def _game_outcomes(store: TraceStore, trace_id: str) -> list[Any]:
    return store.conn.execute(
        "SELECT home_score, away_score, result FROM outcomes WHERE trace_id = ?",
        (trace_id,),
    ).fetchall()


class TestFetchOutcomesNBA:
    def test_grades_a_game_trace(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-nba-1"))
        store.close()

        sb = _sb({"2026-05-17": [_final("Boston Celtics", "Miami Heat", 110, 102)]})
        rc = fetch_outcomes_nba.main(["--since", "2026-05-17", "--db", db], scoreboard_fetcher=sb)
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = _game_outcomes(store, "sandbox-nba-1")
        assert len(rows) == 1
        assert (rows[0]["home_score"], rows[0]["away_score"]) == (110, 102)
        assert rows[0]["result"] == "home_win"
        store.close()

    def test_dry_run_attaches_nothing(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-nba-dry"))
        store.close()

        sb = _sb({"2026-05-17": [_final("Boston Celtics", "Miami Heat", 110, 102)]})
        rc = fetch_outcomes_nba.main(
            ["--since", "2026-05-17", "--db", db, "--dry-run"], scoreboard_fetcher=sb
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-nba-dry") == []
        store.close()

    def test_prop_traces_not_touched(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_prop_trace("sandbox-nba-prop"))
        store.close()

        sb = _sb({"2026-05-17": [_final("Boston Celtics", "Miami Heat", 110, 102)]})
        rc = fetch_outcomes_nba.main(["--since", "2026-05-17", "--db", db], scoreboard_fetcher=sb)
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-nba-prop") == []
        store.close()

    def test_rerun_is_idempotent(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-nba-idem"))
        store.close()

        sb = _sb({"2026-05-17": [_final("Boston Celtics", "Miami Heat", 110, 102)]})
        for _ in range(2):
            rc = fetch_outcomes_nba.main(["--since", "2026-05-17", "--db", db], scoreboard_fetcher=sb)
            assert rc == 0

        store = TraceStore(db_path=db)
        assert len(_game_outcomes(store, "sandbox-nba-idem")) == 1
        store.close()

    def test_unmapped_team_not_attached(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-nba-unmapped", home_team="Springfield Spinners"))
        store.close()

        sb = _sb({"2026-05-17": [_final("Boston Celtics", "Miami Heat", 110, 102)]})
        rc = fetch_outcomes_nba.main(["--since", "2026-05-17", "--db", db], scoreboard_fetcher=sb)
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-nba-unmapped") == []
        store.close()

