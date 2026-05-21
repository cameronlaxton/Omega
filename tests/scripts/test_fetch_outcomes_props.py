"""
End-to-end tests for scripts/fetch_outcomes_props.py.

Hits a real SQLite DB but stubs both ESPN endpoints via the script's
``scoreboard_fetcher`` / ``box_score_fetcher`` injection points — no network.

Covers:
- Happy path: a prop trace with game identity is graded and a prop_outcomes row lands
- Unsupported prop_type is skipped (no row, logged)
- Missing game identity is skipped (no row, logged)
- Game-kind traces are not touched (handled by fetch_outcomes_nba/mlb)
- Re-run is idempotent (existing prop_outcome is preserved)
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
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch_outcomes_props  # type: ignore  # noqa: E402

from omega.integrations.espn_nba import FinalGame  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tmp_store_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _make_prop_trace(
    trace_id: str,
    *,
    league: str = "NBA",
    player_name: str = "Jayson Tatum",
    prop_type: str = "pts",
    line: float = 24.5,
    home_team: str = "Miami Heat",
    away_team: str = "Boston Celtics",
    game_date: str = "2026-05-17",
    timestamp: str = "2026-05-17T19:00:00Z",
    recommendation: str = "over",
    **snap_overrides: Any,
) -> dict[str, Any]:
    snap = {
        "player_name": player_name,
        "league": league,
        "prop_type": prop_type,
        "line": line,
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date,
    }
    snap.update(snap_overrides)
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": timestamp,
        "prompt": f"{league} {player_name} {prop_type} {line}",
        "league": league,
        "matchup": f"{away_team} @ {home_team}",
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
            "status": "success",
            "over_prob": 0.55,
            "under_prob": 0.45,
            "recommendation": recommendation,
        },
    }


def _make_game_trace(trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-17T19:00:00Z",
        "prompt": "game",
        "league": "NBA",
        "matchup": "Boston Celtics @ Miami Heat",
        "execution_mode": "sandbox_game",
        "kind": "game",
        "predictions": None,
        "recommendations": None,
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {
            "home_team": "Miami Heat",
            "away_team": "Boston Celtics",
            "league": "NBA",
        },
        "result": {},
    }


def _fake_scoreboard_factory(games_by_league_date: dict[tuple, list[FinalGame]]):
    """Return a (league, date) → games callable using the supplied map."""

    def _fetch(league: str, d: date):
        return games_by_league_date.get((league, d.isoformat()), [])

    return _fetch


def _fake_box_score_factory(payloads_by_event: dict[str, dict[str, Any]]):
    def _fetch(league: str, event_id: str) -> dict[str, Any]:
        if event_id not in payloads_by_event:
            raise RuntimeError(f"no fixture for event {event_id}")
        return payloads_by_event[event_id]

    return _fetch


def _nba_pts_payload(player: str, pts: float) -> dict[str, Any]:
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
                                    "stats": ["35", str(pts), "5", "4"],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchOutcomesPropsHappyPath:
    def test_grades_a_prop_trace(self):
        db = _tmp_store_path()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace("sandbox-prop-1"))
        store.close()

        games = [
            FinalGame(
                event_id="EV-1",
                date="2026-05-17",
                home_team="Miami Heat",
                away_team="Boston Celtics",
                home_score=0,
                away_score=0,
                status="final",
            )
        ]
        sb = _fake_scoreboard_factory({("NBA", "2026-05-17"): games})
        bs = _fake_box_score_factory({"EV-1": _nba_pts_payload("Jayson Tatum", 31)})

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=sb,
            box_score_fetcher=bs,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = store.get_prop_outcomes("sandbox-prop-1")
        assert len(rows) == 1
        assert rows[0]["stat_type"] == "pts"
        assert rows[0]["stat_value"] == 31.0
        assert rows[0]["line"] == 24.5
        assert rows[0]["side"] == "over"
        assert rows[0]["result"] == "win"
        assert rows[0]["source"] == "api:espn_boxscore"
        store.close()

    def test_dry_run_attaches_nothing(self):
        db = _tmp_store_path()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace("sandbox-prop-dry"))
        store.close()

        games = [
            FinalGame(
                event_id="EV-1",
                date="2026-05-17",
                home_team="Miami Heat",
                away_team="Boston Celtics",
                home_score=0,
                away_score=0,
                status="final",
            )
        ]
        sb = _fake_scoreboard_factory({("NBA", "2026-05-17"): games})
        bs = _fake_box_score_factory({"EV-1": _nba_pts_payload("Jayson Tatum", 31)})

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db, "--dry-run"],
            scoreboard_fetcher=sb,
            box_score_fetcher=bs,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert store.get_prop_outcomes("sandbox-prop-dry") == []
        store.close()

    def test_rerun_is_idempotent(self):
        db = _tmp_store_path()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace("sandbox-prop-idem"))
        store.close()

        games = [
            FinalGame(
                event_id="EV-1",
                date="2026-05-17",
                home_team="Miami Heat",
                away_team="Boston Celtics",
                home_score=0,
                away_score=0,
                status="final",
            )
        ]
        sb = _fake_scoreboard_factory({("NBA", "2026-05-17"): games})
        bs = _fake_box_score_factory({"EV-1": _nba_pts_payload("Jayson Tatum", 31)})

        for _ in range(2):
            rc = fetch_outcomes_props.main(
                ["--since", "2026-05-17", "--league", "NBA", "--db", db],
                scoreboard_fetcher=sb,
                box_score_fetcher=bs,
            )
            assert rc == 0

        store = TraceStore(db_path=db)
        rows = store.get_prop_outcomes("sandbox-prop-idem")
        assert len(rows) == 1
        # On rerun, query_traces(has_outcome=False) should no longer surface it
        ungraded = store.query_traces(league="NBA", has_outcome=False)
        assert all(t["trace_id"] != "sandbox-prop-idem" for t in ungraded)
        store.close()


class TestFetchOutcomesPropsSkips:
    def test_missing_game_identity_is_skipped(self):
        """Prop trace lacking home_team/away_team/game_date must NOT be graded."""
        db = _tmp_store_path()
        store = TraceStore(db_path=db)
        # Build the trace with no game fields on input_snapshot
        trace = _make_prop_trace("sandbox-prop-nogame")
        trace["input_snapshot"].pop("home_team")
        trace["input_snapshot"].pop("away_team")
        trace["input_snapshot"].pop("game_date")
        # Without game fields the matchup denormalization will fall back to
        # the prop descriptor form, which is fine for this test.
        trace["matchup"] = "Jayson Tatum pts 24.5"
        store.persist(trace)
        store.close()

        sb = _fake_scoreboard_factory({})
        bs = _fake_box_score_factory({})

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=sb,
            box_score_fetcher=bs,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert store.get_prop_outcomes("sandbox-prop-nogame") == []
        store.close()

    def test_unsupported_prop_type_is_skipped(self):
        db = _tmp_store_path()
        store = TraceStore(db_path=db)
        store.persist(
            _make_prop_trace(
                "sandbox-prop-unsup",
                prop_type="double_double",
                line=0.5,
            )
        )
        store.close()

        sb = _fake_scoreboard_factory({})  # never reached
        bs = _fake_box_score_factory({})

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=sb,
            box_score_fetcher=bs,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert store.get_prop_outcomes("sandbox-prop-unsup") == []
        store.close()

    def test_game_kind_traces_are_not_touched(self):
        """fetch_outcomes_props.py owns props only; it must never attach to game traces."""
        db = _tmp_store_path()
        store = TraceStore(db_path=db)
        store.persist(_make_game_trace("sandbox-game-1"))
        store.close()

        sb = _fake_scoreboard_factory({})
        bs = _fake_box_score_factory({})

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=sb,
            box_score_fetcher=bs,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        # No prop outcomes
        assert store.get_prop_outcomes("sandbox-game-1") == []
        # And no game outcomes either (this script never attaches game outcomes)
        game_outcomes = store.conn.execute(
            "SELECT outcome_id FROM outcomes WHERE trace_id = ?",
            ("sandbox-game-1",),
        ).fetchall()
        assert game_outcomes == []
        store.close()

    def test_under_loss_resolved(self):
        """Side='under' with stat_value > line should resolve to 'loss'."""
        db = _tmp_store_path()
        store = TraceStore(db_path=db)
        store.persist(
            _make_prop_trace(
                "sandbox-prop-under",
                recommendation="under",
            )
        )
        store.close()

        games = [
            FinalGame(
                event_id="EV-1",
                date="2026-05-17",
                home_team="Miami Heat",
                away_team="Boston Celtics",
                home_score=0,
                away_score=0,
                status="final",
            )
        ]
        sb = _fake_scoreboard_factory({("NBA", "2026-05-17"): games})
        bs = _fake_box_score_factory({"EV-1": _nba_pts_payload("Jayson Tatum", 31)})

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=sb,
            box_score_fetcher=bs,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = store.get_prop_outcomes("sandbox-prop-under")
        assert rows[0]["side"] == "under"
        assert rows[0]["result"] == "loss"
        store.close()

    def test_player_name_normalization_matches_accents(self):
        """Trace says 'Luka Doncic', ESPN box score says 'Luka Dončić' — should still grade."""
        db = _tmp_store_path()
        store = TraceStore(db_path=db)
        store.persist(
            _make_prop_trace(
                "sandbox-prop-luka",
                player_name="Luka Doncic",
                home_team="Miami Heat",
                away_team="Boston Celtics",
            )
        )
        store.close()

        games = [
            FinalGame(
                event_id="EV-1",
                date="2026-05-17",
                home_team="Miami Heat",
                away_team="Boston Celtics",
                home_score=0,
                away_score=0,
                status="final",
            )
        ]
        sb = _fake_scoreboard_factory({("NBA", "2026-05-17"): games})
        bs = _fake_box_score_factory({"EV-1": _nba_pts_payload("Luka Dončić", 30)})

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=sb,
            box_score_fetcher=bs,
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = store.get_prop_outcomes("sandbox-prop-luka")
        assert len(rows) == 1
        assert rows[0]["stat_value"] == 30.0
        # player_name on the row matches the trace's spelling, not ESPN's
        assert rows[0]["player_name"] == "Luka Doncic"
        store.close()
