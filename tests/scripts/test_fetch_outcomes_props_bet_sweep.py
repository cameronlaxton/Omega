"""
Tests for the bet_records sweep in scripts/fetch_outcomes_props.py.

BUG-2 defense: when the agent minted a separate bet-confirmation trace,
the bet's trace_id and the analysis trace's id ended up disjoint, so
prop_outcomes attached to the analysis trace never reached the bet.
The sweep ensures bet-trace candidates flow through the same grading path
and prop_outcomes land under the bet's trace_id (which report tooling joins on).
"""

from __future__ import annotations

import sys
import tempfile
import uuid
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
from omega.trace.bet_record import BetRecord, BetStatus  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _tmp_db_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _make_prop_trace(
    trace_id: str,
    *,
    player: str = "Jayson Tatum",
    prop_type: str = "pts",
    line: float = 24.5,
    timestamp: str = "2026-05-17T19:00:00Z",
    include_identity: bool = True,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "player_name": player,
        "league": "NBA",
        "prop_type": prop_type,
        "line": line,
    }
    if include_identity:
        snap["home_team"] = "Miami Heat"
        snap["away_team"] = "Boston Celtics"
        snap["game_date"] = "2026-05-17"
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": timestamp,
        "prompt": f"NBA {player} {prop_type} {line}",
        "league": "NBA",
        "matchup": "Boston Celtics @ Miami Heat"
        if include_identity
        else f"{player} {prop_type} {line}",
        "execution_mode": "sandbox_prop",
        "kind": "prop",
        "predictions": {"over_prob": 0.55, "under_prob": 0.45},
        "odds_snapshot": {"odds_over": -110, "odds_under": -110},
        "downgrades": [],
        "input_snapshot": snap,
        "result": {
            "player_name": player,
            "prop_type": prop_type,
            "line": line,
            "status": "success",
            "over_prob": 0.55,
            "under_prob": 0.45,
            "recommendation": "over",
        },
    }


def _make_bet(trace_id: str, descriptor: str = "tatum_over_24.5_pts") -> BetRecord:
    return BetRecord(
        bet_id=uuid.uuid4().hex[:12],
        trace_id=trace_id,
        book="DraftKings",
        market="player_prop:pts",
        selection="Jayson Tatum Over 24.5 pts",
        selection_descriptor=descriptor,
        line_taken=24.5,
        odds_taken=-110,
        stake_units=1.0,
        decision_timestamp="2026-05-17T19:30:00Z",
        status=BetStatus.PENDING,
    )


def _scoreboard():
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

    def _fetch(league: str, d: date):
        return games

    return _fetch


def _box_score():
    payload = {
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
                                    "athlete": {"displayName": "Jayson Tatum"},
                                    "stats": ["35", "31", "5", "4"],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }

    def _fetch(league: str, event_id: str) -> dict[str, Any]:
        return payload

    return _fetch


class TestBetSweep:
    def test_sweep_grades_via_bet_trace_id(self):
        """The bet's trace is distinct from any analysis trace but carries
        identity. The sweep grades through it and prop_outcome lands under
        the bet's trace_id."""
        db = _tmp_db_path()
        store = TraceStore(db_path=db)
        # Bet trace has full identity (post-Cowork-policy state, or operator-
        # composed single-trace export). No separate analysis trace exists.
        store.persist(_make_prop_trace("sandbox-bet-1", include_identity=True))
        store.record_bet(_make_bet("sandbox-bet-1"))
        store.close()

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=_scoreboard(),
            box_score_fetcher=_box_score(),
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = store.get_prop_outcomes("sandbox-bet-1")
        assert len(rows) == 1
        assert rows[0]["stat_value"] == 31.0
        assert rows[0]["result"] == "win"
        store.close()

    def test_sweep_does_not_duplicate_when_already_in_traces_path(self):
        """A trace pulled by both query_traces() and the bet sweep must be
        graded once (idempotent via UNIQUE constraint, and no double-attach)."""
        db = _tmp_db_path()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace("sandbox-bet-dup"))
        store.record_bet(_make_bet("sandbox-bet-dup"))
        store.close()

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=_scoreboard(),
            box_score_fetcher=_box_score(),
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        rows = store.get_prop_outcomes("sandbox-bet-dup")
        assert len(rows) == 1  # not duplicated
        store.close()

    def test_sweep_skips_when_bet_trace_lacks_identity(self):
        """A bet whose trace lacks home/away/date is the legacy BUG-2/4 case;
        it stays manual. The sweep must not crash, just skip it."""
        db = _tmp_db_path()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace("sandbox-bet-noid", include_identity=False))
        store.record_bet(_make_bet("sandbox-bet-noid"))
        store.close()

        rc = fetch_outcomes_props.main(
            ["--since", "2026-05-17", "--league", "NBA", "--db", db],
            scoreboard_fetcher=_scoreboard(),
            box_score_fetcher=_box_score(),
        )
        assert rc == 0

        store = TraceStore(db_path=db)
        assert store.get_prop_outcomes("sandbox-bet-noid") == []
        store.close()

    def test_sweep_ignores_already_graded_bet(self):
        """A bet whose trace already has a prop_outcome should not appear in
        the sweep candidate list (NOT EXISTS clause filters it out)."""
        db = _tmp_db_path()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace("sandbox-bet-graded"))
        store.record_bet(_make_bet("sandbox-bet-graded"))
        # Pre-attach an outcome
        store.attach_prop_outcome(
            trace_id="sandbox-bet-graded",
            player_name="Jayson Tatum",
            stat_type="pts",
            stat_value=31.0,
            line=24.5,
            side="over",
            source="manual:test",
        )
        candidates = store.query_ungraded_prop_bet_traces(league="NBA")
        assert all(t["trace_id"] != "sandbox-bet-graded" for t in candidates)
        store.close()

    def test_sweep_ignores_non_pending_bets(self):
        db = _tmp_db_path()
        store = TraceStore(db_path=db)
        store.persist(_make_prop_trace("sandbox-bet-won"))
        bet = _make_bet("sandbox-bet-won")
        store.record_bet(bet)
        store.update_bet_status(bet.bet_id, "won")

        candidates = store.query_ungraded_prop_bet_traces(league="NBA")
        assert all(t["trace_id"] != "sandbox-bet-won" for t in candidates)
        store.close()
