"""Tests for omega.ops.session_run — hardened daily session orchestrator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest

from omega.ops.session_run import (
    _analysis_plan,
    _generate_session_id,
    _league_list,
    _soccer_leagues_in,
    _tennis_leagues_in,
    run_session,
)

_SID = "sess-20260702-000000test"


def _read_sidecar(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(tmp_path: Path, *, rc_by_label: dict[str, int] | None = None, **kwargs):
    """Run run_session with subprocess phases mocked; returns (rc, [(label, cmd)])."""
    calls: list[tuple[str, list[str]]] = []

    def fake(cmd, *, label, dry_run):
        calls.append((label, list(cmd)))
        return (rc_by_label or {}).get(label, 0)

    defaults = dict(
        session_id=_SID,
        date="2026-07-02",
        leagues=["MLB"],
        sidecar_dir=tmp_path,
        trace_inbox=tmp_path / "traces",
        db=str(tmp_path / "t.db"),
        dry_run=False,
    )
    defaults.update(kwargs)
    with patch("omega.ops.session_run._run_subprocess", side_effect=fake):
        rc = run_session(**defaults)
    return rc, calls


# ---------------------------------------------------------------------------
# Helper / utility function tests
# ---------------------------------------------------------------------------


def test_generate_session_id_format():
    # Canonical format: sess-YYYYMMDD-HHMMSSXXXX (compact date, 6-digit UTC
    # time, 4 random hex chars) — matches omega-session-bootstrap.
    sid = _generate_session_id("2026-06-19")
    assert sid.startswith("sess-20260619-")
    parts = sid.split("-")
    assert len(parts) == 3
    assert len(parts[2]) == 10


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
        session_id="sess-20260619-000000test",
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
    assert "sess-20260619-000000test" in plan


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


def test_analysis_plan_mentions_ingest_and_closeout_commands():
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
    assert "omega-validate-trace-export" in plan
    assert "--reopen --ingest --render-report --close" in plan


def test_analysis_plan_prefers_run_batch_over_handrolled_loop():
    plan = _analysis_plan(
        session_id="s",
        date="2026-06-19",
        leagues=["MLB"],
        mlb_games=9,
        mlb_props=10,
        tennis_games=0,
        fifa_games=0,
        mode="research-lean",
        downgraded_leagues=[],
        require_actionable_min=0,
    )
    assert "omega_run_batch" in plan
    assert "ONLY if MCP is unavailable" in plan


# ---------------------------------------------------------------------------
# Phase 1: sidecar open / reopen / collision / closed rejection
# ---------------------------------------------------------------------------


class TestSidecarLifecycle:
    def test_sidecar_created_on_new_session(self, tmp_path):
        rc, _ = _run(tmp_path)
        assert rc == 0
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        assert sc["session_id"] == _SID
        assert sc["closed_at"] is None
        assert sc["window"] == "2026-07-02"
        assert sc["league"] == "MLB"
        # TraceStore resolution recorded (best-effort; never None-and-None here).
        assert sc["runtime_db_status"] is not None
        steps = [e["step"] for e in sc["audit_events"]]
        assert "session_open" in steps
        assert "cowork_preflight" in steps
        assert "analysis_plan" in steps

    def test_same_session_reopen_appends_not_resets(self, tmp_path):
        rc1, _ = _run(tmp_path)
        assert rc1 == 0
        opened_at = _read_sidecar(tmp_path / f"{_SID}.json")["opened_at"]

        rc2, _ = _run(tmp_path, reopen=True)
        assert rc2 == 0
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        assert sc["opened_at"] == opened_at  # not re-bootstrapped
        steps = [e["step"] for e in sc["audit_events"]]
        assert "session_open" in steps and "session_reopen" in steps
        assert steps.count("analysis_plan") == 2  # both invocations audited

    def test_session_id_collision_rejected_without_reopen(self, tmp_path):
        rc1, _ = _run(tmp_path)
        assert rc1 == 0
        rc2, calls2 = _run(tmp_path)  # same id, no --reopen: a different conversation
        assert rc2 == 1
        assert calls2 == []  # failed closed before any phase command ran

    def test_closed_session_cannot_be_reopened(self, tmp_path):
        rc1, _ = _run(tmp_path, ingest=True, render_report=True, close=True)
        assert rc1 == 0
        assert _read_sidecar(tmp_path / f"{_SID}.json")["closed_at"] is not None

        rc2, calls2 = _run(tmp_path, reopen=True)
        assert rc2 == 1
        assert calls2 == []

    def test_reopen_requires_explicit_session_id(self, tmp_path):
        rc, calls = _run(tmp_path, session_id=None, reopen=True)
        assert rc == 1
        assert calls == []

    def test_reopen_of_nonexistent_sidecar_rejected(self, tmp_path):
        rc, calls = _run(tmp_path, reopen=True)
        assert rc == 1
        assert calls == []

    def test_leagues_required_unless_proof(self, tmp_path):
        rc, calls = _run(tmp_path, leagues=[])
        assert rc == 1
        assert calls == []


# ---------------------------------------------------------------------------
# Dry-run / proof modes must not mutate production paths
# ---------------------------------------------------------------------------


class TestDryRunAndProof:
    def test_dry_run_mutates_nothing_and_prints_exact_commands(self, tmp_path, capsys):
        sidecar_dir = tmp_path / "sessions"  # deliberately nonexistent
        with patch("omega.ops.session_run.subprocess.run") as mock_run:
            rc = run_session(
                session_id=_SID,
                date="2026-07-02",
                leagues=["MLB"],
                sidecar_dir=sidecar_dir,
                trace_inbox=tmp_path / "traces",
                db=str(tmp_path / "t.db"),
                ingest=True,
                render_report=True,
                close=True,
                dry_run=True,
            )
        assert rc == 0
        mock_run.assert_not_called()
        assert not sidecar_dir.exists()  # no sidecar dir/file created

        out = capsys.readouterr().out
        assert "sidecar intent" in out
        for module in (
            "omega.ops.cowork_preflight",
            "omega.ops.validate_trace_export",
            "omega.ops.ingest_traces",
            "omega.ops.render_session_report",
            "omega.ops.render_session_audits",
            "omega.ops.validate_session_sidecars",
        ):
            assert module in out, f"dry-run must print the exact {module} command"
        assert "would close sidecar" in out

    def test_proof_mode_delegates_to_prove_lifecycle(self, tmp_path):
        from omega.ops.prove_lifecycle import LifecycleResult

        ok = LifecycleResult(ok=True, work_dir=str(tmp_path), stages=[])
        with patch("omega.ops.prove_lifecycle.run", return_value=ok) as mock_run:
            rc = run_session(proof=True, skip_preflight=True)
        assert rc == 0
        mock_run.assert_called_once_with(keep_artifacts=False, skip_preflight=True)
        # No production sidecar side effects: proof runs in its own temp dir.

    def test_proof_mode_failure_is_nonzero(self, tmp_path):
        from omega.ops.prove_lifecycle import LifecycleResult

        bad = LifecycleResult(ok=False, work_dir=str(tmp_path), stages=[])
        with patch("omega.ops.prove_lifecycle.run", return_value=bad):
            rc = run_session(proof=True, skip_preflight=True)
        assert rc == 1


# ---------------------------------------------------------------------------
# Phase ordering + closeout
# ---------------------------------------------------------------------------


class TestPhaseOrderingAndCloseout:
    def test_commands_invoked_in_lifecycle_order(self, tmp_path):
        rc, calls = _run(tmp_path, ingest=True, render_report=True, close=True)
        assert rc == 0
        assert [label for label, _ in calls] == [
            "preflight",
            "validate-trace-export",
            "ingest-traces",
            "render-report",
            "render-audits",
            "validate-sidecars",
        ]
        cmds = dict(calls)
        assert "--formal-output-gate" in cmds["preflight"]
        assert "--strict" in cmds["validate-trace-export"]
        assert str(tmp_path / "traces") in cmds["validate-trace-export"]
        assert "--strict" in cmds["ingest-traces"]
        assert str(tmp_path) in cmds["ingest-traces"]  # --sidecar-dir threaded through
        assert _SID in cmds["render-report"]
        assert _SID in cmds["render-audits"]
        assert str(tmp_path) in cmds["validate-sidecars"]

    def test_closeout_populates_closed_at_and_exec_stats(self, tmp_path):
        rc, _ = _run(tmp_path, ingest=True, render_report=True, close=True)
        assert rc == 0
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        assert sc["closed_at"] is not None
        assert sc["exec_stats"]["phases"]["ingest-traces"] == 0
        assert sc["exec_stats"]["failed_phases"] == []
        assert sc["pipeline_status"]["session_run"] == "closed"
        assert sc["next_required_action"] == "none"
        steps = [e["step"] for e in sc["audit_events"]]
        assert "session_close" in steps

    def test_close_sidecar_failure_does_not_audit_success(self, tmp_path):
        with patch("omega.ops.session_run.close_sidecar", side_effect=OSError("boom")):
            rc, _ = _run(tmp_path, ingest=True, render_report=True, close=True)

        assert rc == 1
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        assert sc["closed_at"] is None
        assert not any(
            e["step"] == "session_close" and e["status"] == "ok"
            for e in sc["audit_events"]
        )

    def test_validation_and_ingest_outcomes_audited(self, tmp_path):
        rc, _ = _run(tmp_path, ingest=True)
        assert rc == 0
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        by_step = {e["step"]: e for e in sc["audit_events"]}
        assert by_step["validate-trace-export"]["event_type"] == "quality_gate"
        assert by_step["validate-trace-export"]["status"] == "ok"
        assert by_step["ingest-traces"]["status"] == "ok"

    def test_validation_failure_blocks_ingest(self, tmp_path):
        rc, calls = _run(tmp_path, rc_by_label={"validate-trace-export": 1}, ingest=True)
        assert rc == 1
        labels = [label for label, _ in calls]
        assert "validate-trace-export" in labels
        assert "ingest-traces" not in labels  # fail-closed before ingest
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        gate = next(e for e in sc["audit_events"] if e["step"] == "validate-trace-export")
        assert gate["status"] == "fail"

    def test_failure_appends_audit_event_and_leaves_session_open(self, tmp_path):
        rc, _ = _run(
            tmp_path,
            rc_by_label={"ingest-traces": 3},
            ingest=True,
            render_report=True,
            close=True,
        )
        assert rc == 1
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        assert sc["closed_at"] is None  # close refused; retryable with --reopen
        ingest_event = next(e for e in sc["audit_events"] if e["step"] == "ingest-traces")
        assert ingest_event["status"] == "fail"
        close_event = next(e for e in sc["audit_events"] if e["step"] == "session_close")
        assert close_event["status"] == "fail"

    def test_preflight_failure_is_fatal_and_audited(self, tmp_path):
        rc, calls = _run(tmp_path, rc_by_label={"preflight": 1}, ingest=True)
        assert rc == 1
        assert [label for label, _ in calls] == ["preflight"]
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        pf = next(e for e in sc["audit_events"] if e["step"] == "cowork_preflight")
        assert pf["event_type"] == "preflight" and pf["status"] == "fail"

    def test_skip_preflight_records_skipped_event(self, tmp_path):
        rc, calls = _run(tmp_path, skip_preflight=True)
        assert rc == 0
        assert "preflight" not in [label for label, _ in calls]
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        pf = next(e for e in sc["audit_events"] if e["step"] == "cowork_preflight")
        assert pf["status"] == "skipped"

    def test_downgraded_soccer_league_exits_2(self, tmp_path):
        rc, _ = _run(
            tmp_path,
            rc_by_label={"soccer-gate-FIFA_WORLD_CUP_2026": 2},
            leagues=["FIFA_WORLD_CUP_2026"],
            fifa_games=2,
        )
        assert rc == 2
        sc = _read_sidecar(tmp_path / f"{_SID}.json")
        dg = next(e for e in sc["audit_events"] if e["event_type"] == "downgrade")
        assert "research_candidate" in dg["notes"]


# ---------------------------------------------------------------------------
# run_session in dry-run mode (no subprocess calls) — legacy smoke tests
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
    assert "sess-20260619-" in captured
    assert rc == 0


# ---------------------------------------------------------------------------
# main() CLI smoke tests
# ---------------------------------------------------------------------------


def test_main_dry_run_exits_0():
    from omega.ops.session_run import main

    rc = main(
        [
            "--leagues",
            "MLB,TENNIS",
            "--session-id",
            "sess-cli-test",
            "--date",
            "2026-06-19",
            "--mlb-games",
            "3",
            "--tennis-games",
            "5",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_main_missing_leagues_errors():
    from omega.ops.session_run import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--dry-run"])
    assert exc_info.value.code != 0


def test_main_proof_does_not_require_leagues(capsys):
    from omega.ops.session_run import main

    rc = main(["--proof", "--dry-run"])
    assert rc == 0
    assert "prove_lifecycle" in capsys.readouterr().out
