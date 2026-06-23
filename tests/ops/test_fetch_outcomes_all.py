from __future__ import annotations

import subprocess

from omega.ops.fetch_outcomes_all import run_fetch_outcomes


def _fake_run_factory(calls: list[list[str]]):
    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    return _fake_run


def test_fetch_outcomes_all_routes_tennis_tours(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("omega.ops.fetch_outcomes_all.subprocess.run", _fake_run_factory(calls))

    result = run_fetch_outcomes(
        leagues=["ATP", "wta", "grand_slam"],
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["leagues"] == ["atp", "wta", "grand_slam"]
    assert [cmd[2] for cmd in calls] == [
        "omega.ops.fetch_outcomes_tennis",
        "omega.ops.fetch_outcomes_tennis",
        "omega.ops.fetch_outcomes_tennis",
    ]
    assert calls[0][-3:] == ["--leagues", "ATP", "--dry-run"]
    assert calls[1][-3:] == ["--leagues", "WTA", "--dry-run"]
    assert calls[2][-3:] == ["--leagues", "GRAND_SLAM", "--dry-run"]


def test_fetch_outcomes_all_default_includes_tennis(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("omega.ops.fetch_outcomes_all.subprocess.run", _fake_run_factory(calls))

    result = run_fetch_outcomes(dry_run=True)

    assert "tennis" in result["leagues"]
    tennis_calls = [cmd for cmd in calls if cmd[2] == "omega.ops.fetch_outcomes_tennis"]
    assert len(tennis_calls) == 1
    assert "--leagues" not in tennis_calls[0]
