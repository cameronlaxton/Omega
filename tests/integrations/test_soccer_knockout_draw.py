"""Knockout draw-grading correctness (audit remediation C7).

In 3-way (1X2) soccer betting the draw market settles on 90' regulation. A
single-leg knockout that reaches extra time or penalties WAS level at 90', so it
settles as a draw even though the final/ET score shows a winner. ESPN drops the
ET/penalty status once a match is "final"; we now capture it and grade the 1X2
result as a regulation draw for single-leg international competitions.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.integrations import espn_soccer  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _event(home, away, hs, as_, *, name="STATUS_FULL_TIME"):
    return {
        "events": [
            {
                "id": "SOC-KO-1",
                "date": "2026-07-05T19:00Z",
                "competitions": [
                    {
                        "status": {"type": {"completed": True, "state": "post", "name": name}},
                        "competitors": [
                            {"homeAway": "home", "score": hs, "team": {"displayName": home}},
                            {"homeAway": "away", "score": as_, "team": {"displayName": away}},
                        ],
                    }
                ],
            }
        ]
    }


def test_full_time_match_not_flagged():
    games = espn_soccer.parse_scoreboard(_event("Brazil", "Croatia", 2, 1), league="WORLD_CUP")
    assert len(games) == 1
    assert games[0].decided_after_regulation is False


def test_extra_time_flagged():
    games = espn_soccer.parse_scoreboard(
        _event("Argentina", "Netherlands", 2, 1, name="STATUS_FULL_TIME_ET"), league="WORLD_CUP"
    )
    assert games[0].decided_after_regulation is True
    assert games[0].status == "final"  # still normalized for uniform filtering


def test_penalties_flagged():
    games = espn_soccer.parse_scoreboard(
        _event("France", "Italy", 1, 1, name="STATUS_PENALTIES"), league="EURO"
    )
    assert games[0].decided_after_regulation is True


def _make_game_trace(trace_id: str, league: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-07-05T19:00:00Z",
        "prompt": "game",
        "league": league,
        "matchup": "Netherlands @ Argentina",
        "execution_mode": "sandbox_game",
        "kind": "game",
        "predictions": None,
        "recommendations": None,
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {"home_team": "Argentina", "away_team": "Netherlands", "league": league},
        "result": {},
    }


def test_result_override_grades_draw_despite_unequal_scores():
    """attach_outcome(result_override='draw') stores result=draw while keeping the
    actual (ET) scores as provenance — the C7 grading for single-leg knockouts."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = TraceStore(db_path=tmp.name)
    try:
        store.persist(_make_game_trace("sandbox-ko-1", "WORLD_CUP"))
        store.attach_outcome(
            trace_id="sandbox-ko-1",
            home_score=2,
            away_score=1,
            source="api:espn:aet",
            result_override="draw",
        )
        outcome = store.get_outcome("sandbox-ko-1")
        assert outcome["result"] == "draw"  # 1X2 settles on the 90' draw
        assert outcome["home_score"] == 2 and outcome["away_score"] == 1  # ET score kept
    finally:
        store.close()


def test_no_override_derives_from_scores():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = TraceStore(db_path=tmp.name)
    try:
        store.persist(_make_game_trace("sandbox-ko-2", "WORLD_CUP"))
        store.attach_outcome(trace_id="sandbox-ko-2", home_score=2, away_score=1)
        assert store.get_outcome("sandbox-ko-2")["result"] == "home_win"
    finally:
        store.close()
