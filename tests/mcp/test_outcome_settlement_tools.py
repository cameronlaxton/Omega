"""Tests for the batch outcome / settlement / DNP-void MCP tools.

Covers omega_fetch_outcomes, omega_settle_bets, and omega_trace_void_prop —
the tools added to close the hand-scripting gap when gathering outcomes for
pending traces.
"""

from __future__ import annotations

import subprocess

import pytest

from omega.mcp.server import (
    omega_fetch_outcomes,
    omega_settle_bets,
    omega_trace_void_prop,
)
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore


@pytest.fixture
def db_path(monkeypatch, tmp_path):
    monkeypatch.setenv("OMEGA_BET_LEDGER_AUTOLOG", "0")
    return str(tmp_path / "traces.db")


def _persist_prop_trace(store: TraceStore, trace_id: str) -> None:
    store.persist(
        {
            "trace_id": trace_id,
            "run_id": "r",
            "timestamp": "2026-06-02T00:00:00Z",
            "prompt": "p",
            "league": "MLB",
            "matchup": "NYY @ CLE",
            "execution_mode": "native_sim",
            "kind": "prop",
            "result": {"status": "success"},
        }
    )


def _prop_bet(trace_id: str) -> LedgerBet:
    return LedgerBet(
        ledger_id="dnp1",
        trace_id=trace_id,
        bet_date="2026-06-02",
        league="MLB",
        sport="baseball",
        matchup="NYY @ CLE",
        market="player_prop:hits",
        bookmaker="betmgm",
        selection="Aaron Judge Over 0.5 hits",
        selection_descriptor="aaron_judge_over_0.5_hits",
        line=0.5,
        odds=-150,
        stake_amount=25.0,
        status=LedgerStatus.PENDING,
        provenance=BetProvenance.USER_CONFIRMED,
        decision_timestamp="2026-06-02T00:00:00Z",
    )


# --------------------------------------------------------------------------
# omega_trace_void_prop + omega_settle_bets (DNP end-to-end)
# --------------------------------------------------------------------------


def test_void_prop_then_settle_as_void(db_path):
    store = TraceStore(db_path=db_path)
    _persist_prop_trace(store, "dnp-trace")
    store.record_ledger_bet(_prop_bet("dnp-trace"))
    store.close()

    voided = omega_trace_void_prop(
        "dnp-trace",
        player_name="Aaron Judge",
        stat_type="hits",
        reason="dnp",
        db_path=db_path,
    )
    assert voided["status"] == "success"
    assert voided["result"] == "void"

    # Dry run reports the void without writing.
    dry = omega_settle_bets(db_path=db_path)
    assert dry["status"] == "success"
    assert dry["applied"] is False
    assert dry["settled"].get("void") == 1
    assert dry["ungradeable"] == 0

    applied = omega_settle_bets(apply=True, db_path=db_path)
    assert applied["status"] == "success"
    assert applied["applied"] is True
    assert applied["settled"].get("void") == 1

    store = TraceStore(db_path=db_path)
    row = store.get_ledger_bets("dnp-trace")[0]
    store.close()
    assert row["status"] == "void"
    assert row["net_pnl"] == 0.0


def test_void_prop_refuses_to_overwrite_existing_outcome(db_path):
    """If a non-void prop outcome is already attached, void must NOT silently
    no-op and report success -- it returns outcome_exists with the real result."""
    store = TraceStore(db_path=db_path)
    _persist_prop_trace(store, "graded-trace")
    store.attach_prop_outcome(
        "graded-trace",
        player_name="Aaron Judge",
        stat_type="hits",
        stat_value=2.0,
        line=0.5,
        side="over",
    )  # real graded WIN
    store.close()

    result = omega_trace_void_prop(
        "graded-trace",
        player_name="Aaron Judge",
        stat_type="hits",
        db_path=db_path,
    )
    assert result["status"] == "error"
    assert result["error_code"] == "outcome_exists"
    assert result["detail"]["existing_result"] == "win"


def test_void_prop_unknown_trace_errors(db_path):
    result = omega_trace_void_prop(
        "missing",
        player_name="Nobody",
        stat_type="points",
        db_path=db_path,
    )
    assert result["status"] == "error"
    assert result["error_code"] == "trace_not_found"


def test_void_prop_rejects_bad_side(db_path):
    result = omega_trace_void_prop(
        "x", player_name="P", stat_type="points", side="sideways", db_path=db_path
    )
    assert result["status"] == "error"
    assert result["error_code"] == "invalid_request"


def test_settle_bets_rejects_bad_provenance(db_path):
    result = omega_settle_bets(provenance="bogus", db_path=db_path)
    assert result["status"] == "error"
    assert result["error_code"] == "invalid_request"


# --------------------------------------------------------------------------
# omega_fetch_outcomes
# --------------------------------------------------------------------------


def _fake_run_factory(calls: list[list[str]]):
    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    return _fake_run


def test_fetch_outcomes_excludes_soccer_when_omitted(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("omega.ops.fetch_outcomes_all.subprocess.run", _fake_run_factory(calls))

    result = omega_fetch_outcomes(leagues=["nba", "mlb", "props"], dry_run=True)

    assert result["status"] == "success"
    assert result["ok"] is True
    assert result["leagues"] == ["nba", "mlb", "props"]
    assert [r["league"] for r in result["results"]] == ["nba", "mlb", "props"]
    # No soccer module was dispatched.
    assert not any("soccer" in " ".join(cmd) for cmd in calls)
    # dry_run flag is propagated to each sub-command.
    assert all("--dry-run" in cmd for cmd in calls)


def test_fetch_outcomes_accepts_tennis_tours(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("omega.ops.fetch_outcomes_all.subprocess.run", _fake_run_factory(calls))

    result = omega_fetch_outcomes(leagues=["ATP", "WTA"], dry_run=True)

    assert result["status"] == "success"
    assert result["ok"] is True
    assert result["leagues"] == ["atp", "wta"]
    assert all(cmd[2] == "omega.ops.fetch_outcomes_tennis" for cmd in calls)
    assert calls[0][-3:] == ["--leagues", "ATP", "--dry-run"]
    assert calls[1][-3:] == ["--leagues", "WTA", "--dry-run"]


def test_fetch_outcomes_reports_subprocess_failure(monkeypatch):
    def _fail_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr("omega.ops.fetch_outcomes_all.subprocess.run", _fail_run)

    result = omega_fetch_outcomes(leagues=["mlb"])

    assert result["status"] == "success"  # the tool ran; the sub-script failed
    assert result["ok"] is False
    assert result["failures"] == 1
    assert result["results"][0]["returncode"] == 1


def test_fetch_outcomes_reports_timeout_as_payload_failure(monkeypatch):
    def _timeout(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd,
            timeout=kwargs["timeout"],
            output="partial out",
            stderr="hung",
        )

    monkeypatch.setattr("omega.ops.fetch_outcomes_all.subprocess.run", _timeout)

    result = omega_fetch_outcomes(leagues=["mlb"])

    assert result["status"] == "success"
    assert result["ok"] is False
    assert result["failures"] == 1
    assert result["results"][0]["timed_out"] is True
    assert result["results"][0]["returncode"] is None


def test_fetch_outcomes_rejects_unknown_league():
    result = omega_fetch_outcomes(leagues=["nba", "cricket"])
    assert result["status"] == "error"
    assert result["error_code"] == "invalid_request"
