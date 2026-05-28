"""Unit tests for scripts/bug_sentinel.py.

Tests use monkeypatching and tmp files — no real engine imports, no network,
no writes to the repo. Each test exercises one check type in isolation.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root on path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bug_sentinel as bs  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bug(
    bug_id: str = "BUG-TEST-001",
    check_type: str = "manual",
    severity: str = "medium",
    check: dict | None = None,
    suppresses_bet_card: bool = False,
    sport_gates: list[str] | None = None,
    analysis_kind_gates: list[str] | None = None,
    status_at_last_audit: str = "present",
) -> dict:
    return {
        "bug_id": bug_id,
        "title": f"Test bug {bug_id}",
        "severity": severity,
        "suppresses_bet_card": suppresses_bet_card,
        "blocks_ingest": False,
        "sport_gates": sport_gates or [],
        "analysis_kind_gates": analysis_kind_gates or [],
        "check_type": check_type,
        "check": check or {"note": "manual"},
        "status_at_last_audit": status_at_last_audit,
        "last_audited": "2026-05-28",
        "workaround": "test workaround",
    }


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

def test_load_catalog(tmp_path: Path) -> None:
    catalog = {"catalog_version": "1.0", "bugs": [make_bug()]}
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(catalog))
    bugs = bs.load_catalog(p)
    assert len(bugs) == 1
    assert bugs[0]["bug_id"] == "BUG-TEST-001"


def test_load_catalog_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bs.load_catalog(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Grep check
# ---------------------------------------------------------------------------

def test_grep_bad_pattern_found_means_bug(tmp_path: Path) -> None:
    src = tmp_path / "engine.py"
    src.write_text("def _sim_baseball():\n    x = league_avg_rpg / away_def\n")
    bug = make_bug(check_type="grep", check={
        "file": "engine.py",
        "bad_pattern": "league_avg_rpg\\s*/\\s*away_def",
        "good_pattern": "_expected_against_allowed_rate",
        "present_means": "bug_present",
    })
    status, evidence = bs._run_grep(bug, tmp_path)
    assert status == bs.STATUS_PRESENT
    assert "bad_pattern" in evidence


def test_grep_good_pattern_found_means_fixed(tmp_path: Path) -> None:
    src = tmp_path / "engine.py"
    src.write_text("def _sim_baseball():\n    x = _expected_against_allowed_rate(a, b, c)\n")
    bug = make_bug(check_type="grep", check={
        "file": "engine.py",
        "bad_pattern": "league_avg_rpg\\s*/\\s*away_def",
        "good_pattern": "_expected_against_allowed_rate",
        "present_means": "bug_present",
    })
    status, evidence = bs._run_grep(bug, tmp_path)
    assert status == bs.STATUS_FIXED
    assert "good_pattern" in evidence


def test_grep_function_scope_limits_search(tmp_path: Path) -> None:
    # Bad pattern appears only outside the scoped function — should not match
    src = tmp_path / "engine.py"
    src.write_text(textwrap.dedent("""\
        def _other():
            x = league_avg_rpg / away_def

        def _sim_baseball():
            x = _expected_against_allowed_rate(a, b, c)
    """))
    bug = make_bug(check_type="grep", check={
        "file": "engine.py",
        "bad_pattern": "league_avg_rpg\\s*/\\s*away_def",
        "good_pattern": "_expected_against_allowed_rate",
        "function_scope": "_sim_baseball",
        "present_means": "bug_present",
    })
    status, _ = bs._run_grep(bug, tmp_path)
    assert status == bs.STATUS_FIXED


def test_grep_file_missing(tmp_path: Path) -> None:
    bug = make_bug(check_type="grep", check={
        "file": "nonexistent.py",
        "good_pattern": "anything",
    })
    status, evidence = bs._run_grep(bug, tmp_path)
    assert status == bs.STATUS_ERROR
    assert "not found" in evidence


# ---------------------------------------------------------------------------
# Manual check
# ---------------------------------------------------------------------------

def test_manual_always_unknown() -> None:
    bug = make_bug(check_type="manual", check={"note": "Cannot automate."})
    status, evidence = bs._run_manual(bug)
    assert status == bs.STATUS_UNKNOWN
    assert "Cannot automate" in evidence


# ---------------------------------------------------------------------------
# DB query check
# ---------------------------------------------------------------------------

def test_db_query_wal_success(tmp_path: Path) -> None:
    """WAL probe on a local tmp dir should succeed (no FUSE)."""
    # Create a placeholder DB file so the directory exists
    (tmp_path / "omega_traces.db").touch()
    bug = make_bug(check_type="db_query", check={
        "db_path": "omega_traces.db",
        "pragma": "PRAGMA journal_mode=WAL",
        "success_means": "fixed",
        "failure_means": "bug_present",
    })
    status, evidence = bs._run_db_query(bug, tmp_path)
    assert status == bs.STATUS_FIXED
    assert "WAL" in evidence or "wal" in evidence.lower()


def test_db_query_wal_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate FUSE OperationalError — should report bug present."""
    (tmp_path / "omega_traces.db").touch()

    original_connect = sqlite3.connect

    def failing_connect(path: str, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sqlite3, "connect", failing_connect)

    bug = make_bug(check_type="db_query", check={
        "db_path": "omega_traces.db",
        "pragma": "PRAGMA journal_mode=WAL",
        "success_means": "fixed",
        "failure_means": "bug_present",
    })
    status, evidence = bs._run_db_query(bug, tmp_path)
    assert status == bs.STATUS_PRESENT
    assert "OperationalError" in evidence


# ---------------------------------------------------------------------------
# Import test — input snapshot identity
# ---------------------------------------------------------------------------

def _make_good_analyze(snap_fields: list[str]):
    """Return a mock analyze() that includes the given fields in input_snapshot."""
    def _analyze(request, *, session_id, bankroll):
        return {
            "trace_id": "sandbox-test",
            "input_snapshot": {f: "val" for f in snap_fields},
            "result": {},
        }
    return _analyze


def test_input_snapshot_all_fields_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    required = ["player_name", "home_team", "away_team", "game_date", "prop_type", "line"]
    mock_analyze = _make_good_analyze(required)

    with patch.dict("sys.modules", {"omega.core.contracts.service": MagicMock(analyze=mock_analyze)}):
        bug = make_bug(check_type="import_test", check={
            "module": "omega.core.contracts.service",
            "function": "analyze",
            "check_action": "analyze_fixture",
            "fixture": {"player_name": "X", "league": "NBA", "prop_type": "pts",
                        "line": 20.0, "home_team": "H", "away_team": "A",
                        "game_date": "2026-01-01", "odds_over": -110, "odds_under": -110,
                        "player_context": {"pts_mean": 20.0, "pts_std": 5.0},
                        "game_context": {"is_playoff": False, "rest_days": 1},
                        "n_iterations": 100, "seed": 42},
            "required_in_input_snapshot": required,
            "present_means": "fixed",
        })
        status, evidence = bs._check_input_snapshot_identity(bug, bug["check"], tmp_path)

    assert status == bs.STATUS_FIXED
    assert "all required identity fields" in evidence


def test_input_snapshot_missing_fields(tmp_path: Path) -> None:
    mock_analyze = _make_good_analyze(["player_name"])  # missing home_team etc.

    with patch.dict("sys.modules", {"omega.core.contracts.service": MagicMock(analyze=mock_analyze)}):
        bug = make_bug(check_type="import_test", check={
            "module": "omega.core.contracts.service",
            "function": "analyze",
            "check_action": "analyze_fixture",
            "fixture": {},
            "required_in_input_snapshot": ["player_name", "home_team", "game_date"],
            "present_means": "fixed",
        })
        status, evidence = bs._check_input_snapshot_identity(bug, bug["check"], tmp_path)

    assert status == bs.STATUS_PRESENT
    assert "missing" in evidence


# ---------------------------------------------------------------------------
# Import test — MLB draw_prob
# ---------------------------------------------------------------------------

def test_mlb_draw_prob_zero(tmp_path: Path) -> None:
    def _analyze(request, *, session_id, bankroll):
        return {"result": {"simulation": {"draw_prob": 0.0, "home_win_prob": 65.0, "away_win_prob": 35.0}}}

    with patch.dict("sys.modules", {"omega.core.contracts.service": MagicMock(analyze=_analyze)}):
        bug = make_bug(check_type="import_test", check={
            "module": "omega.core.contracts.service",
            "function": "analyze",
            "check_action": "mlb_draw_prob",
            "fixture": {"home_team": "LAD", "away_team": "COL", "league": "MLB",
                        "home_context": {}, "away_context": {},
                        "game_context": {"is_playoff": False, "rest_days": 1},
                        "odds": {}, "n_iterations": 100, "seed": 1},
            "expected_draw_prob": 0.0,
            "present_means": "fixed",
        })
        status, evidence = bs._check_mlb_draw_prob(bug, bug["check"], tmp_path)

    assert status == bs.STATUS_FIXED
    assert "draw_prob=0.0" in evidence


def test_mlb_draw_prob_nonzero_means_bug(tmp_path: Path) -> None:
    def _analyze(request, *, session_id, bankroll):
        return {"result": {"simulation": {"draw_prob": 12.7, "home_win_prob": 60.0, "away_win_prob": 27.3}}}

    with patch.dict("sys.modules", {"omega.core.contracts.service": MagicMock(analyze=_analyze)}):
        bug = make_bug(check_type="import_test", check={
            "module": "omega.core.contracts.service",
            "function": "analyze",
            "check_action": "mlb_draw_prob",
            "fixture": {},
            "expected_draw_prob": 0.0,
            "present_means": "fixed",
        })
        status, evidence = bs._check_mlb_draw_prob(bug, bug["check"], tmp_path)

    assert status == bs.STATUS_PRESENT
    assert "12.7" in evidence


# ---------------------------------------------------------------------------
# Gate summary
# ---------------------------------------------------------------------------

def test_gate_summary_critical_present_suppresses_gates() -> None:
    results = [
        {
            "bug_id": "BUG-TEST-001",
            "status": bs.STATUS_PRESENT,
            "severity": "critical",
            "suppresses_bet_card": True,
            "sport_gates": ["MLB", "NHL"],
            "analysis_kind_gates": ["game"],
            "status_at_last_audit": "present",
        }
    ]
    summary = bs.build_gate_summary(results)
    assert summary["MLB_game"] == "suppressed"
    assert summary["NHL_game"] == "suppressed"
    assert summary["NBA_game"] == "clear"
    assert summary["NBA_prop"] == "clear"


def test_gate_summary_fixed_bug_does_not_suppress() -> None:
    results = [
        {
            "bug_id": "BUG-TEST-001",
            "status": bs.STATUS_FIXED,
            "severity": "critical",
            "suppresses_bet_card": True,
            "sport_gates": ["MLB"],
            "analysis_kind_gates": ["game"],
            "status_at_last_audit": "present",
        }
    ]
    summary = bs.build_gate_summary(results)
    assert summary["MLB_game"] == "clear"


def test_gate_summary_non_suppressing_present_bug_clears() -> None:
    results = [
        {
            "bug_id": "BUG-TEST-001",
            "status": bs.STATUS_PRESENT,
            "severity": "high",
            "suppresses_bet_card": False,  # HIGH but doesn't suppress Bet Cards
            "sport_gates": ["MLB"],
            "analysis_kind_gates": ["game"],
            "status_at_last_audit": "present",
        }
    ]
    summary = bs.build_gate_summary(results)
    assert summary["MLB_game"] == "clear"


# ---------------------------------------------------------------------------
# Report / all_clear logic
# ---------------------------------------------------------------------------

def test_report_all_clear_when_no_critical_present(tmp_path: Path) -> None:
    results = [
        {**make_bug("B1", severity="critical", status_at_last_audit="fixed"),
         "status": bs.STATUS_FIXED, "evidence": "ok"},
        {**make_bug("B2", severity="high", status_at_last_audit="present"),
         "status": bs.STATUS_PRESENT, "evidence": "ok"},
    ]
    report = bs.build_report(results, tmp_path)
    assert report["open_critical"] == 0
    assert report["open_high"] == 1
    assert report["all_clear"] is False  # HIGH is still open


def test_report_regression_only_for_fixed_audit_state(tmp_path: Path) -> None:
    results = [
        {**make_bug("B1", severity="critical", status_at_last_audit="fixed"),
         "status": bs.STATUS_PRESENT, "evidence": "regression!"},
        {**make_bug("B2", severity="low", status_at_last_audit="shadow_mode"),
         "status": bs.STATUS_PRESENT, "evidence": "expected shadow"},
    ]
    report = bs.build_report(results, tmp_path)
    assert report["regression_count"] == 1  # only B1; shadow_mode is not a regression
    assert report["all_clear"] is False


def test_report_shadow_mode_not_a_regression(tmp_path: Path) -> None:
    results = [
        {**make_bug("B1", severity="low", status_at_last_audit="shadow_mode"),
         "status": bs.STATUS_PRESENT, "evidence": "shadow"},
    ]
    report = bs.build_report(results, tmp_path)
    assert report["regression_count"] == 0


def test_all_clear_true_only_when_no_critical_high_or_regression(tmp_path: Path) -> None:
    results = [
        {**make_bug("B1", severity="medium", status_at_last_audit="present"),
         "status": bs.STATUS_PRESENT, "evidence": "medium open"},
    ]
    report = bs.build_report(results, tmp_path)
    # MEDIUM open, no HIGH/CRITICAL, no regression → all_clear
    assert report["all_clear"] is True


# ---------------------------------------------------------------------------
# Run check dispatcher
# ---------------------------------------------------------------------------

def test_run_check_dispatch_manual() -> None:
    bug = make_bug(check_type="manual", check={"note": "Manual only."})
    result = bs.run_check(bug, Path("/tmp"))
    assert result["status"] == bs.STATUS_UNKNOWN
    assert result["bug_id"] == "BUG-TEST-001"


def test_run_check_unknown_type() -> None:
    bug = make_bug(check_type="unknown_type_xyz", check={})
    result = bs.run_check(bug, Path("/tmp"))
    assert result["status"] == bs.STATUS_UNKNOWN


def test_run_check_exception_returns_error(tmp_path: Path) -> None:
    """A check that raises an unexpected exception returns STATUS_ERROR."""
    bug = make_bug(check_type="grep", check={
        "file": "engine.py",
        "bad_pattern": "[invalid regex ((",  # invalid regex → re.error
    })
    (tmp_path / "engine.py").write_text("hello")
    result = bs.run_check(bug, tmp_path)
    assert result["status"] == bs.STATUS_ERROR


# ---------------------------------------------------------------------------
# CI mode exit code
# ---------------------------------------------------------------------------

def test_ci_exit_0_when_all_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = {"catalog_version": "1.0", "bugs": [
        make_bug("B1", check_type="manual", check={"note": "x"}, status_at_last_audit="present")
    ]}
    (tmp_path / "omega" / "qa").mkdir(parents=True)
    (tmp_path / "omega" / "qa" / "bug_catalog.json").write_text(json.dumps(catalog))

    exit_code = bs.main([
        "--ci", "--json",
        "--repo-root", str(tmp_path),
        "--catalog", str(tmp_path / "omega" / "qa" / "bug_catalog.json"),
    ])
    # manual → unknown → no critical/regression → CI PASS → exit 0
    assert exit_code == 0


def test_ci_exit_1_when_critical_present(tmp_path: Path) -> None:
    (tmp_path / "engine.py").write_text("x = league_avg_rpg / away_def\n")
    catalog = {"catalog_version": "1.0", "bugs": [
        make_bug(
            "BUG-CRITICAL",
            check_type="grep",
            severity="critical",
            suppresses_bet_card=True,
            sport_gates=["MLB"],
            analysis_kind_gates=["game"],
            check={
                "file": "engine.py",
                "bad_pattern": "league_avg_rpg\\s*/\\s*away_def",
                "present_means": "bug_present",
            },
            status_at_last_audit="present",
        )
    ]}
    (tmp_path / "omega" / "qa").mkdir(parents=True)
    (tmp_path / "omega" / "qa" / "bug_catalog.json").write_text(json.dumps(catalog))

    exit_code = bs.main([
        "--ci", "--json",
        "--repo-root", str(tmp_path),
        "--catalog", str(tmp_path / "omega" / "qa" / "bug_catalog.json"),
    ])
    assert exit_code == 1
