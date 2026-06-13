from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_action_plan  # type: ignore  # noqa: E402


def _cmd_by_type(plan: dict) -> dict[str, list[str]]:
    return {atype: cmd for atype, cmd in run_action_plan._validate_all(plan)}


def _script_name(cmd: list[str]) -> str:
    if len(cmd) >= 3 and cmd[1] == "-m":
        return cmd[2]
    return Path(cmd[1]).name


def test_new_deterministic_actions_validate_to_expected_commands():
    plan = {
        "session_id": "test",
        "actions": [
            {"type": "ingest_traces", "args": {"verbose": True}},
            {"type": "fetch_closing_lines", "args": {"league": "nba", "verbose": True}},
            {
                "type": "score_evidence_signals",
                "args": {"league": "nba", "window_days": 14, "verbose": True},
            },
            {
                "type": "fit_adjustment_policy",
                "args": {"league": "nba", "mode": "shadow", "min_samples": 50},
            },
            {
                "type": "fit_calibration",
                "args": {"league": "nba", "plane": "prop", "method": "both", "dry_run": True},
            },
            {
                "type": "settle_bets",
                "args": {"league": "nba", "provenance": "user_confirmed"},
            },
            {"type": "fetch_outcomes", "args": {"leagues": ["nba", "nhl"]}},
        ],
    }

    cmds = _cmd_by_type(plan)

    assert _script_name(cmds["ingest_traces"]) == "omega.ops.ingest_traces"
    assert "--verbose" in cmds["ingest_traces"]
    assert _script_name(cmds["fetch_closing_lines"]) == "omega.ops.fetch_closing_lines"
    assert cmds["fetch_closing_lines"][-3:] == ["--league", "NBA", "--verbose"]
    assert _script_name(cmds["score_evidence_signals"]) == "omega.ops.score_evidence_signals"
    assert "--window-days" in cmds["score_evidence_signals"]
    assert _script_name(cmds["fit_adjustment_policy"]) == "omega.ops.fit_adjustment_policy"
    assert "--mode" in cmds["fit_adjustment_policy"]
    assert "shadow" in cmds["fit_adjustment_policy"]
    assert _script_name(cmds["fit_calibration"]) == "omega.ops.fit_calibration"
    assert "--plane" in cmds["fit_calibration"]
    assert "prop" in cmds["fit_calibration"]
    assert "--dry-run" in cmds["fit_calibration"]
    assert _script_name(cmds["settle_bets"]) == "omega.ops.settle_bets"
    assert cmds["settle_bets"][-5:] == [
        "--provenance",
        "user_confirmed",
        "--league",
        "NBA",
        "--apply",
    ]
    assert _script_name(cmds["fetch_outcomes"]) == "omega.ops.fetch_outcomes_all"
    assert cmds["fetch_outcomes"][-3:] == ["--leagues", "nba", "nhl"]


@pytest.mark.parametrize(
    "action",
    [
        {"type": "promote_adjustment_policy", "args": {"go_live": True}},
        {"type": "fit_adjustment_policy", "args": {"league": "NBA", "mode": "live"}},
        {"type": "fit_adjustment_policy", "args": {"league": "NBA", "go_live": True}},
        {"type": "fetch_outcomes", "args": {"leagues": ["nba", "kbo"]}},
        {"type": "ingest_traces", "args": {"verbose": "yes"}},
        {"type": "fetch_closing_lines", "args": {"league": "KBO"}},
        {"type": "fit_calibration", "args": {"league": "NBA", "plane": "all"}},
        {"type": "settle_bets", "args": {"provenance": "phantom"}},
        {"type": "settle_bets", "args": {"apply": True}},
    ],
)
def test_action_plan_validation_rejects_unsafe_or_invalid_args(action: dict):
    with pytest.raises(ValueError):
        run_action_plan._validate_all({"actions": [action]})


def test_action_plan_templates_dry_run_and_do_not_go_live():
    template_dir = _REPO_ROOT / "fixtures" / "action_plans"
    templates = sorted(template_dir.glob("*.json"))
    assert templates, "fixtures/action_plans must contain tracked action-plan templates"

    for template in templates:
        payload = json.loads(template.read_text(encoding="utf-8"))
        for action in payload["actions"]:
            assert action["type"] != "promote_adjustment_policy"
            assert action["args"].get("go_live") is None
            if action["type"] == "fit_adjustment_policy":
                assert action["args"].get("mode", "shadow") == "shadow"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "omega.ops.run_action_plan",
                str(template),
                "--dry-run",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
