"""
End-to-end tests for scripts/fetch_outcomes_mlb.py â€” MLB game-outcome grading.

Hits a real SQLite DB but stubs the ESPN MLB scoreboard via the script's
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

import fetch_outcomes_mlb  # type: ignore  # noqa: E402

from omega.integrations.espn_mlb import FinalGame  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _game_trace(
    trace_id: str,
    *,
    home_team: str = "New York Yankees",
    away_team: str = "Boston Red Sox",
    timestamp: str = "2026-05-17T19:00:00Z",
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": timestamp,
        "prompt": "game",
        "league": "MLB",
        "matchup": f"{away_team} @ {home_team}",
        "execution_mode": "sandbox_game",
        "kind": "game",
        "predictions": None,
        "recommendations": None,
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {"home_team": home_team, "away_team": away_team, "league": "MLB"},
        "result": {"status": "success"},
    }


def _prop_trace(trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-17T19:00:00Z",
        "prompt": "prop",
        "league": "MLB",
        "matchup": "Boston Red Sox @ New York Yankees",
        "execution_mode": "sandbox_prop",
        "kind": "prop",
        "predictions": {"over_prob": 0.55, "under_prob": 0.45},
        "recommendations": [],
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {
            "player_name": "Aaron Judge",
            "league": "MLB",
            "prop_type": "tb",
            "line": 1.5,
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
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
        event_id="MEV-1",
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


class TestFetchOutcomesMLB:
    def test_grades_a_game_trace(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-mlb-1"))
        store.close()

        sb = _sb({"2026-05-17": [_final("New York Yankees", "Boston Red Sox", 5, 3)]})
        rc = fetch_outcomes_mlb.main(["--since", "2026-05-17", "--db", db], scoreboard_fetcher=sb)
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = _game_outcomes(store, "sandbox-mlb-1")
        assert len(rows) == 1
        assert (rows[0]["home_score"], rows[0]["away_score"]) == (5, 3)
        assert rows[0]["result"] == "home_win"
        store.close()

    def test_dry_run_attaches_nothing(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-mlb-dry"))
        store.close()

        sb = _sb({"2026-05-17": [_final("New York Yankees", "Boston Red Sox", 5, 3)]})
        rc = fetch_outcomes_mlb.main(
            ["--since", "2026-05-17", "--db", db, "--dry-run"], scoreboard_fetcher=sb
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-mlb-dry") == []
        store.close()

    def test_prop_traces_not_touched(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_prop_trace("sandbox-mlb-prop"))
        store.close()

        sb = _sb({"2026-05-17": [_final("New York Yankees", "Boston Red Sox", 5, 3)]})
        rc = fetch_outcomes_mlb.main(["--since", "2026-05-17", "--db", db], scoreboard_fetcher=sb)
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-mlb-prop") == []
        store.close()

    def test_rerun_is_idempotent(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-mlb-idem"))
        store.close()

        sb = _sb({"2026-05-17": [_final("New York Yankees", "Boston Red Sox", 5, 3)]})
        for _ in range(2):
            rc = fetch_outcomes_mlb.main(["--since", "2026-05-17", "--db", db], scoreboard_fetcher=sb)
            assert rc == 0

        store = TraceStore(db_path=db)
        assert len(_game_outcomes(store, "sandbox-mlb-idem")) == 1
        store.close()

    def test_unmapped_team_not_attached(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-mlb-unmapped", home_team="Springfield Isotopes"))
        store.close()

        sb = _sb({"2026-05-17": [_final("New York Yankees", "Boston Red Sox", 5, 3)]})
        rc = fetch_outcomes_mlb.main(["--since", "2026-05-17", "--db", db], scoreboard_fetcher=sb)
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-mlb-unmapped") == []
        store.close()

