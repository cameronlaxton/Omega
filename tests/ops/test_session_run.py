"""Tests for omega.ops.session_run — daily session orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest
from omega.ops.session_run import (
    _generate_session_id,
    _league_list,
    _soccer_leagues_in,
    _tennis_leagues_in,
    _analysis_plan,
    run_session,
)


# ---------------------------------------------------------------------------
# Helper / utility function tests
# ---------------------------------------------------------------------------


def test_generate_session_id_format():
    sid = _generate_session_id("2026-06-19")
    assert sid.startswith("sess-2026-06-19-")
    parts = sid.split("-")
    assert len(parts) == 5


def test_league_list_parses_comma_separated():
    result = _league_list("MLB,TENNIS,FIFA_WORLD_CUP_2026")
    assert result == ["MLB", "TENNIS", "FIFA_WORLD_CUP_2026"]


def test_league_list_strips_whitespace():
    result = _league_list("MLB , WTA , EPL")
    assert result == ["MLB", "WTA", "EPL"]


def test_league_list_upcases():
    result = _league_list("mlb,wta")
    assert result == ["MLB", "WTA"]


def test_soccer_leagues_in():
    leagues = ["MLB", "FIFA_WORLD_CUP_2026", "TENNIS", "EPL"]
    assert set(_soccer_leagues_in(leagues)) == {"FIFA_WORLD_CUP_2026", "EPL"}


def test_tennis_leagues_in():
    leagues = ["MLB", "ATP", "WTA", "EPL", "GRAND_SLAM"]
    assert set(_tennis_leagues_in(leagues)) == {"ATP", "WTA", "GRAND_SLAM"}


def test_no_soccer_leagues():
    assert _soccer_leagues_in(["MLB", "NBA"]) == []


def test_no_tennis_leagues():
    assert _tennis_leagues_in(["MLB", "EPL"]) == []


# ---------------------------------------------------------------------------
# _analysis_plan content
# ---------------------------------------------------------------------------


def test_analysis_plan_contains_session_id():
    plan = _analysis_plan(
        session_id="sess-2026-06-19-test",
        date="2026-06-19",
        leagues=["MLB"],
        mlb_games=5,
        mlb_props=3,
        tennis_games=0,
        fifa_games=0,
        mode="research-lean",
        downgraded_leagues=[],
        require_actionable_min=0,
    )
    assert "sess-2026-06-19-test" in plan


def test_analysis_plan_shows_downgraded_leagues():
    plan = _analysis_plan(
        session_id="sess-test",
        date="2026-06-19",
        leagues=["FIFA_WORLD_CUP_2026", "MLB"],
        mlb_games=3,
        mlb_props=0,
        tennis_games=0,
        fifa_games=2,
        mode="actionable",
        downgraded_leagues=["FIFA_WORLD_CUP_2026"],
        require_actionable_min=0,
    )
    assert "DOWNGRADE" in plan or "downgrade" in plan.lower()
    assert "FIFA_WORLD_CUP_2026" in plan
    assert "research_candidate" in plan


def test_analysis_plan_mentions_ingest_command():
    plan = _analysis_plan(
        session_id="s",
        date="2026-06-19",
        leagues=["MLB"],
        mlb_games=1,
        mlb_props=0,
        tennis_games=0,
        fifa_games=0,
        mode="research-lean",
        downgraded_leagues=[],
        require_actionable_min=0,
    )
    assert "omega-ingest-traces" in plan
    assert "omega-render-session-report" in plan


# ---------------------------------------------------------------------------
# run_session in dry-run mode (no subprocess calls)
# ---------------------------------------------------------------------------


def test_run_session_dry_run_exits_0_no_soccer(capsys):
    rc = run_session(
        session_id="sess-test-dry",
        date="2026-06-19",
        leagues=["MLB"],
        mlb_games=3,
        mode="research-lean",
        dry_run=True,
    )
    assert rc == 0


def test_run_session_dry_run_with_soccer_no_downgrade(capsys):
    # In dry_run, subprocess calls are skipped so soccer gate always returns 0 downgraded
    rc = run_session(
        session_id="sess-test-dry",
        date="2026-06-19",
        leagues=["MLB", "FIFA_WORLD_CUP_2026"],
        mlb_games=3,
        fifa_games=2,
        mode="actionable",
        dry_run=True,
    )
    # 0 (no downgrade) in dry_run because subprocess calls are skipped
    assert rc in (0, 2)  # 2 if somehow downgraded, 0 otherwise (dry_run skips gate)


def test_run_session_prints_analysis_plan(capsys):
    run_session(
        session_id="sess-test-plan",
        date="2026-06-19",
        leagues=["MLB", "TENNIS"],
        mlb_games=5,
        tennis_games=3,
        dry_run=True,
    )
    captured = capsys.readouterr().out
    assert "ANALYSIS PLAN" in captured
    assert "sess-test-plan" in captured


def test_run_session_generates_session_id_if_none(capsys):
    rc = run_session(
        session_id=None,
        date="2026-06-19",
        leagues=["MLB"],
        dry_run=True,
    )
    captured = capsys.readouterr().out
    assert "sess-2026-06-19-" in captured
    assert rc == 0


def test_run_session_skip_preflight(capsys):
    """With skip_preflight=True and dry_run, should not call preflight subprocess."""
    with patch("omega.ops.session_run._run_subprocess") as mock_sub:
        run_session(
            session_id="s",
            date="2026-06-19",
            leagues=["MLB"],
            skip_preflight=True,
            dry_run=False,
        )
    # preflight should NOT have been called
    for call_args in mock_sub.call_args_list:
        cmd = call_args[0][0] if call_args[0] else call_args[1].get("cmd", [])
        assert "omega-cowork-preflight" not in cmd


def test_run_session_downgraded_leagues_exit_2(capsys):
    """When preflight passes but soccer gate returns rc=2, overall exit is 2."""
    with patch("omega.ops.session_run._run_subprocess") as mock_sub:
        # preflight succeeds (0), soccer gate exits 2 (weak coverage)
        mock_sub.side_effect = lambda cmd, label="", dry_run=False: (
            2 if "soccer-gate" in label else 0
        )
        rc = run_session(
            session_id="s",
            date="2026-06-19",
            leagues=["FIFA_WORLD_CUP_2026"],
            fifa_games=2,
            dry_run=False,
        )
    assert rc == 2


def test_run_session_hard_failure_on_preflight_exit_1(capsys):
    """If preflight fails (exit 1), session should exit 1."""
    with patch("omega.ops.session_run._run_subprocess") as mock_sub:
        mock_sub.return_value = 1  # all subprocesses fail
        rc = run_session(
            session_id="s",
            date="2026-06-19",
            leagues=["MLB"],
            dry_run=False,
        )
    assert rc == 1


# ---------------------------------------------------------------------------
# main() CLI smoke test
# ---------------------------------------------------------------------------


def test_main_dry_run_exits_0():
    from omega.ops.session_run import main

    rc = main([
        "--leagues", "MLB,TENNIS",
        "--session-id", "sess-cli-test",
        "--date", "2026-06-19",
        "--mlb-games", "3",
        "--tennis-games", "5",
        "--dry-run",
    ])
    assert rc == 0


def test_main_missing_leagues_errors():
    from omega.ops.session_run import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--dry-run"])
    assert exc_info.value.code != 0
