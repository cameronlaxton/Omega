"""
Tests for omega-backfill-outcomes-manual â€” interactive backlog grading.

Hits a real SQLite DB but stubs ESPN scoreboard and box-score fetches.
Confirmation prompts are stubbed via the `confirm=` callable.

Covers:
- backfill_date: game-kind trace -> attach_outcome
- backfill_date: prop-kind trace -> attach_prop_outcome
- backfill_date: skip ('n') leaves trace ungraded
- backfill_date: quit ('q') stops the walk
- backfill_single_trace: legacy prop trace without game identity gets graded
  when the operator supplies it
- source label format: manual:espn_boxscore_YYYYMMDD
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import backfill_outcomes_manual as backfill  # type: ignore  # noqa: E402

from omega.integrations.espn_nba import FinalGame  # noqa: E402
from omega.integrations.espn_soccer import FinalGame as SoccerFinalGame  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _make_game_trace(
    trace_id: str = "sandbox-game-1",
    league: str = "NBA",
    home_team: str = "Miami Heat",
    away_team: str = "Boston Celtics",
    timestamp: str = "2026-05-17T19:00:00Z",
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
        "input_snapshot": {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
        },
        "result": {},
    }


def _make_prop_trace(
    trace_id: str = "sandbox-prop-1",
    *,
    league: str = "NBA",
    home_team: str = "Miami Heat",
    away_team: str = "Boston Celtics",
    game_date: str = "2026-05-17",
    include_game_fields: bool = True,
    player_name: str = "Jayson Tatum",
    prop_type: str = "pts",
    line: float = 24.5,
    recommendation: str = "over",
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "player_name": player_name,
        "league": league,
        "prop_type": prop_type,
        "line": line,
    }
    if include_game_fields:
        snap["home_team"] = home_team
        snap["away_team"] = away_team
        snap["game_date"] = game_date
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-17T19:00:00Z",
        "prompt": "prop",
        "league": league,
        "matchup": f"{away_team} @ {home_team}"
        if include_game_fields
        else f"{player_name} {prop_type} {line}",
        "execution_mode": "sandbox_prop",
        "kind": "prop",
        "predictions": {"over_prob": 0.55, "under_prob": 0.45},
        "recommendations": [],
        "odds_snapshot": {"odds_over": -110, "odds_under": -110},
        "downgrades": [],
        "input_snapshot": snap,
        "result": {
            "player_name": player_name,
            "prop_type": prop_type,
            "line": line,
            "recommendation": recommendation,
        },
    }


def _final_game() -> FinalGame:
    return FinalGame(
        event_id="EV-1",
        date="2026-05-17",
        home_team="Miami Heat",
        away_team="Boston Celtics",
        home_score=109,
        away_score=114,
        status="final",
    )


def _nba_box_score(player: str, pts: float) -> dict[str, Any]:
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": "Boston Celtics"},
                    "statistics": [
                        {
                            "name": "starters",
                            "keys": ["MIN", "PTS", "REB", "AST"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": player},
                                    "stats": ["38", str(pts), "9", "6"],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }


def _soccer_final_game() -> SoccerFinalGame:
    return SoccerFinalGame(
        event_id="SOC-1",
        date="2026-05-17",
        home_team="Arsenal",
        away_team="Chelsea",
        home_score=2,
        away_score=1,
        status="final",
        league="EPL",
    )


def _soccer_box_score(player: str, shots: float) -> dict[str, Any]:
    return {
        "rosters": [
            {
                "team": {"displayName": "Arsenal"},
                "roster": [
                    {
                        "athlete": {"displayName": player},
                        "stats": [
                            {"name": "totalGoals", "value": "1"},
                            {"name": "goalAssists", "value": "0"},
                            {"name": "totalShots", "value": str(shots)},
                            {"name": "shotsOnTarget", "value": "2"},
                            {"name": "yellowCards", "value": "0"},
                            {"name": "redCards", "value": "0"},
                        ],
                    }
                ],
            }
        ]
    }


class TestBackfillDate:
    def test_attaches_game_outcome_when_confirmed(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_make_game_trace())

        counts = backfill.backfill_date(
            store,
            league="NBA",
            d=date(2026, 5, 17),
            confirm=lambda _desc: "y",
            scoreboard_fetcher=lambda _l, _d: [_final_game()],
            box_score_fetcher=lambda _l, _e: {},
        )
        assert counts["game_attached"] == 1
        assert counts["prop_attached"] == 0
        # Source label format
        row = store.conn.execute(
            "SELECT source, home_score, away_score FROM outcomes WHERE trace_id = ?",
            ("sandbox-game-1",),
        ).fetchone()
        assert row["source"] == "manual:espn_boxscore_20260517"
        assert row["home_score"] == 109
        assert row["away_score"] == 114
        store.close()

    def test_attaches_prop_outcome_when_confirmed(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace())

        counts = backfill.backfill_date(
            store,
            league="NBA",
            d=date(2026, 5, 17),
            confirm=lambda _desc: "y",
            scoreboard_fetcher=lambda _l, _d: [_final_game()],
            box_score_fetcher=lambda _l, _e: _nba_box_score("Jayson Tatum", 31),
        )
        assert counts["prop_attached"] == 1
        rows = store.get_prop_outcomes("sandbox-prop-1")
        assert len(rows) == 1
        assert rows[0]["stat_value"] == 31.0
        assert rows[0]["result"] == "win"
        assert rows[0]["source"] == "manual:espn_boxscore_20260517"
        store.close()

    def test_attaches_soccer_game_outcome_when_confirmed(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(
            _make_game_trace(
                trace_id="sandbox-soccer-game-1",
                home_team="Arsenal",
                away_team="Chelsea",
                league="EPL",
            )
        )

        counts = backfill.backfill_date(
            store,
            league="EPL",
            d=date(2026, 5, 17),
            confirm=lambda _desc: "y",
            scoreboard_fetcher=lambda _l, _d: [_soccer_final_game()],
            box_score_fetcher=lambda _l, _e: {},
        )
        assert counts["game_attached"] == 1
        row = store.conn.execute(
            "SELECT source, home_score, away_score, result FROM outcomes WHERE trace_id = ?",
            ("sandbox-soccer-game-1",),
        ).fetchone()
        assert row["source"] == "manual:espn_boxscore_20260517"
        assert row["home_score"] == 2
        assert row["away_score"] == 1
        assert row["result"] == "home_win"
        store.close()

    def test_attaches_soccer_prop_outcome_when_confirmed(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(
            _make_prop_trace(
                trace_id="sandbox-soccer-prop-1",
                home_team="Arsenal",
                away_team="Chelsea",
                league="EPL",
                player_name="Bukayo Saka",
                prop_type="shots",
                line=2.5,
                recommendation="over",
            )
        )

        counts = backfill.backfill_date(
            store,
            league="EPL",
            d=date(2026, 5, 17),
            confirm=lambda _desc: "y",
            scoreboard_fetcher=lambda _l, _d: [_soccer_final_game()],
            box_score_fetcher=lambda _l, _e: _soccer_box_score("Bukayo Saka", 4),
        )
        assert counts["prop_attached"] == 1
        rows = store.get_prop_outcomes("sandbox-soccer-prop-1")
        assert len(rows) == 1
        assert rows[0]["stat_type"] == "shots"
        assert rows[0]["stat_value"] == 4.0
        assert rows[0]["result"] == "win"
        assert rows[0]["source"] == "manual:espn_boxscore_20260517"
        store.close()

    def test_skip_response_leaves_trace_ungraded(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_make_game_trace())

        counts = backfill.backfill_date(
            store,
            league="NBA",
            d=date(2026, 5, 17),
            confirm=lambda _desc: "n",
            scoreboard_fetcher=lambda _l, _d: [_final_game()],
            box_score_fetcher=lambda _l, _e: {},
        )
        assert counts["skipped"] == 1
        assert counts["game_attached"] == 0
        ungraded = store.query_traces(league="NBA", has_outcome=False)
        assert any(t["trace_id"] == "sandbox-game-1" for t in ungraded)
        store.close()

    def test_quit_response_stops_walk(self):
        """First trace gets 'q' â€” second trace should not be touched."""
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_make_game_trace("sandbox-game-A"))
        store.persist(_make_game_trace("sandbox-game-B"))

        # Quit on first prompt
        counts = backfill.backfill_date(
            store,
            league="NBA",
            d=date(2026, 5, 17),
            confirm=lambda _desc: "q",
            scoreboard_fetcher=lambda _l, _d: [_final_game()],
            box_score_fetcher=lambda _l, _e: {},
        )
        assert counts["quit"] == 1
        assert counts["game_attached"] == 0
        # No outcomes anywhere
        n = store.conn.execute("SELECT COUNT(*) AS c FROM outcomes").fetchone()["c"]
        assert n == 0
        store.close()

    def test_dry_run_increments_count_but_writes_nothing(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace())

        counts = backfill.backfill_date(
            store,
            league="NBA",
            d=date(2026, 5, 17),
            confirm=lambda _desc: "y",
            scoreboard_fetcher=lambda _l, _d: [_final_game()],
            box_score_fetcher=lambda _l, _e: _nba_box_score("Jayson Tatum", 31),
            dry_run=True,
        )
        assert counts["prop_attached"] == 1
        assert store.get_prop_outcomes("sandbox-prop-1") == []
        store.close()

class TestBackfillSingleTrace:
    def test_legacy_prop_without_game_identity_can_be_pinned(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        # Legacy prop trace: no home_team/away_team/game_date on input_snapshot
        store.persist(
            _make_prop_trace(
                "sandbox-prop-legacy",
                include_game_fields=False,
            )
        )

        counts = backfill.backfill_single_trace(
            store,
            trace_id="sandbox-prop-legacy",
            league="NBA",
            game_date=date(2026, 5, 17),
            home_team="Miami Heat",
            away_team="Boston Celtics",
            confirm=lambda _desc: "y",
            scoreboard_fetcher=lambda _l, _d: [_final_game()],
            box_score_fetcher=lambda _l, _e: _nba_box_score("Jayson Tatum", 31),
        )
        assert counts["prop_attached"] == 1
        rows = store.get_prop_outcomes("sandbox-prop-legacy")
        assert len(rows) == 1
        assert rows[0]["source"] == "manual:espn_boxscore_20260517"
        store.close()

    def test_missing_trace_id_returns_unmatched(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)

        counts = backfill.backfill_single_trace(
            store,
            trace_id="does-not-exist",
            league="NBA",
            game_date=date(2026, 5, 17),
            home_team="Miami Heat",
            away_team="Boston Celtics",
            confirm=lambda _desc: "y",
            scoreboard_fetcher=lambda _l, _d: [],
            box_score_fetcher=lambda _l, _e: {},
        )
        assert counts["unmatched"] == 1
        assert counts["prop_attached"] == 0
        assert counts["game_attached"] == 0
        store.close()
