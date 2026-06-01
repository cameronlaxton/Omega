"""
End-to-end tests for scripts/fetch_outcomes_soccer.py â€” soccer game-outcome grading.

Hits a real SQLite DB but stubs the ESPN soccer scoreboard via the script's
``scoreboard_fetcher`` injection point (signature: (iso_date, league)) â€” no network.

Covers:
- Home-win grading
- Draw grading (3-way result â†’ result == "draw")
- Dry-run attaches nothing
- Prop traces are NOT touched
- Re-run is idempotent
- --leagues filters which competitions are fetched
- A trace with no matching final is left ungraded
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

import fetch_outcomes_soccer  # type: ignore  # noqa: E402

from omega.integrations.espn_soccer import FinalGame  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _game_trace(
    trace_id: str,
    *,
    league: str = "EPL",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    timestamp: str = "2026-05-17T16:00:00Z",
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": timestamp,
        "prompt": "game",
        "league": league,
        "matchup": f"{away_team} @ {home_team}",
        "execution_mode": "sandbox_game",
        "kind": "game",
        "predictions": None,
        "recommendations": None,
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {"home_team": home_team, "away_team": away_team, "league": league},
        "result": {"status": "success"},
    }


def _prop_trace(trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-17T16:00:00Z",
        "prompt": "prop",
        "league": "EPL",
        "matchup": "Chelsea @ Arsenal",
        "execution_mode": "sandbox_prop",
        "kind": "prop",
        "predictions": {"over_prob": 0.55, "under_prob": 0.45},
        "recommendations": [],
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {
            "player_name": "Bukayo Saka",
            "league": "EPL",
            "prop_type": "shots",
            "line": 2.5,
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "game_date": "2026-05-17",
        },
        "result": {"status": "success", "recommendation": "over"},
    }


def _final(home, away, hs, as_, league="EPL", d="2026-05-17") -> FinalGame:
    return FinalGame(
        event_id="SOC-1",
        date=d,
        home_team=home,
        away_team=away,
        home_score=hs,
        away_score=as_,
        status="final",
        league=league,
    )


def _sb(games_by_key: dict[tuple[str, str], list[FinalGame]]):
    """key = (iso_date, league)."""

    def _fetch(iso_date: str, league: str):
        return games_by_key.get((iso_date, league), [])

    return _fetch


def _game_outcomes(store: TraceStore, trace_id: str) -> list[Any]:
    return store.conn.execute(
        "SELECT home_score, away_score, result FROM outcomes WHERE trace_id = ?",
        (trace_id,),
    ).fetchall()


class TestFetchOutcomesSoccer:
    def test_grades_a_home_win(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-soccer-1"))
        store.close()

        sb = _sb({("2026-05-17", "EPL"): [_final("Arsenal", "Chelsea", 2, 1)]})
        rc = fetch_outcomes_soccer.main(
            ["--since", "2026-05-17", "--leagues", "EPL", "--db", db],
            scoreboard_fetcher=sb,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = _game_outcomes(store, "sandbox-soccer-1")
        assert len(rows) == 1
        assert (rows[0]["home_score"], rows[0]["away_score"]) == (2, 1)
        assert rows[0]["result"] == "home_win"
        store.close()

    def test_grades_a_draw(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-soccer-draw"))
        store.close()

        sb = _sb({("2026-05-17", "EPL"): [_final("Arsenal", "Chelsea", 1, 1)]})
        rc = fetch_outcomes_soccer.main(
            ["--since", "2026-05-17", "--leagues", "EPL", "--db", db],
            scoreboard_fetcher=sb,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = _game_outcomes(store, "sandbox-soccer-draw")
        assert len(rows) == 1
        assert rows[0]["result"] == "draw"
        store.close()

    def test_dry_run_attaches_nothing(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-soccer-dry"))
        store.close()

        sb = _sb({("2026-05-17", "EPL"): [_final("Arsenal", "Chelsea", 2, 1)]})
        rc = fetch_outcomes_soccer.main(
            ["--since", "2026-05-17", "--leagues", "EPL", "--db", db, "--dry-run"],
            scoreboard_fetcher=sb,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-soccer-dry") == []
        store.close()

    def test_prop_traces_not_touched(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_prop_trace("sandbox-soccer-prop"))
        store.close()

        sb = _sb({("2026-05-17", "EPL"): [_final("Arsenal", "Chelsea", 2, 1)]})
        rc = fetch_outcomes_soccer.main(
            ["--since", "2026-05-17", "--leagues", "EPL", "--db", db],
            scoreboard_fetcher=sb,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-soccer-prop") == []
        store.close()

    def test_rerun_is_idempotent(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-soccer-idem"))
        store.close()

        sb = _sb({("2026-05-17", "EPL"): [_final("Arsenal", "Chelsea", 2, 1)]})
        for _ in range(2):
            rc = fetch_outcomes_soccer.main(
                ["--since", "2026-05-17", "--leagues", "EPL", "--db", db],
                scoreboard_fetcher=sb,
            )
            assert rc == 0

        store = TraceStore(db_path=db)
        rows = _game_outcomes(store, "sandbox-soccer-idem")
        assert len(rows) == 1
        store.close()

    def test_league_filter_skips_other_competitions(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-laliga", league="LA_LIGA", home_team="Barcelona", away_team="Real Madrid"))
        store.close()

        # Provide a La Liga final, but only ask for EPL â†’ nothing should attach.
        sb = _sb({("2026-05-17", "LA_LIGA"): [_final("Barcelona", "Real Madrid", 3, 0, league="LA_LIGA")]})
        rc = fetch_outcomes_soccer.main(
            ["--since", "2026-05-17", "--leagues", "EPL", "--db", db],
            scoreboard_fetcher=sb,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-laliga") == []
        store.close()

    def test_no_matching_final_left_ungraded(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_game_trace("sandbox-soccer-nomatch", home_team="Brentford", away_team="Fulham"))
        store.close()

        sb = _sb({("2026-05-17", "EPL"): [_final("Arsenal", "Chelsea", 2, 1)]})
        rc = fetch_outcomes_soccer.main(
            ["--since", "2026-05-17", "--leagues", "EPL", "--db", db],
            scoreboard_fetcher=sb,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert _game_outcomes(store, "sandbox-soccer-nomatch") == []
        store.close()

