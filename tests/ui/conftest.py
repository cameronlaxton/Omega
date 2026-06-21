"""Shared fixtures for the read-only operator console tests (Phase 8 Milestone A).

Each test gets its own temp trace DB and temp session-sidecar directory so
assertions are deterministic and isolated from the live runtime DB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore


def make_trace(
    trace_id: str,
    *,
    session_id: str | None = None,
    league: str = "NBA",
    kind: str = "game",
    timestamp: str = "2026-03-21T12:00:00Z",
    matchup: str = "Celtics @ Lakers",
    aggregate_quality: float = 0.85,
    recommendations: Any | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A minimal but realistic trace dict accepted by TraceStore.persist()."""
    if recommendations is None:
        recommendations = [
            {
                "side": "home",
                "market": "moneyline",
                "confidence_tier": "A",
                "edge_pct": 4.2,
                "ev_pct": 3.1,
                "recommended_units": 1.5,
            }
        ]
    base: dict[str, Any] = {
        "trace_id": trace_id,
        "run_id": f"r-{trace_id}",
        "timestamp": timestamp,
        "kind": kind,
        "session_id": session_id,
        "prompt": f"{league} {kind}",
        "league": league,
        "matchup": matchup,
        "execution_mode": f"sandbox_{kind}",
        "simulation_seed": 123,
        "aggregate_quality": aggregate_quality,
        "predictions": {"home_win_prob": 0.58, "away_win_prob": 0.42},
        "recommendations": recommendations,
        "odds_snapshot": {"moneyline_home": -150},
        "downgrades": [],
        "result": {"status": "success", "context_source": "provided"},
        "trace_quality": {
            "aggregate_quality": aggregate_quality,
            "calibration_eligible": True,
            "identity_status": "complete",
        },
    }
    base.update(overrides)
    return base


def write_valid_sidecar(
    sessions_dir: Path,
    session_id: str,
    *,
    exec_stats: dict[str, Any] | None = None,
    agent_notes: str = "",
    audit_events: list[dict[str, Any]] | None = None,
    league: str | None = None,
) -> Path:
    """Write a sidecar JSON that validates against the SessionSidecar contract."""
    payload = {
        "session_id": session_id,
        "opened_at": "2026-03-21T11:00:00Z",
        "closed_at": None,
        "model_version": "omega-core-test",
        "purpose": "test session",
        "league": league,
        "window": None,
        "effective_db_path": None,
        "runtime_db_status": None,
        "pipeline_status": {},
        "next_required_action": None,
        "bankroll": 1000.0,
        "bankroll_confirmed": True,
        "exec_stats": exec_stats or {},
        "agent_notes": agent_notes,
        "audit_events": audit_events or [],
    }
    path = sessions_dir / f"{session_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sessions"
    d.mkdir()
    return d


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "omega_traces.db")


@pytest.fixture
def seeded(db_path: str, sessions_dir: Path) -> dict[str, Any]:
    """Seed a temp DB with known traces, an outcome, and one user-confirmed bet.

    Autolog is suppressed during persist so the bet_ledger contains exactly the
    one bet we record explicitly (deterministic for the bet tests).
    """
    store = TraceStore(db_path=db_path)
    with store.autolog_suppressed():
        store.persist(
            make_trace("sandbox-aaa", session_id="sess-test-1", league="NBA", kind="game")
        )
        store.persist(
            make_trace(
                "sandbox-bbb",
                session_id="sess-test-1",
                league="NBA",
                kind="prop",
                matchup="LeBron James pts 25.5",
                recommendations=[
                    {"market": "player_prop:pts", "confidence_tier": "B", "edge_pct": 2.0}
                ],
            )
        )
        store.persist(
            make_trace(
                "sandbox-ccc",
                session_id="sess-test-2",
                league="EPL",
                kind="game",
                timestamp="2026-03-22T12:00:00Z",
                matchup="Arsenal vs Chelsea",
            )
        )
    store.attach_outcome("sandbox-aaa", 110, 100)
    bet = LedgerBet(
        ledger_id="led-aaa-1",
        trace_id="sandbox-aaa",
        bet_date="2026-03-21",
        league="NBA",
        sport="basketball",
        matchup="Celtics @ Lakers",
        market="moneyline",
        bookmaker="draftkings",
        selection="Lakers ML",
        selection_descriptor="home_moneyline",
        odds=-150.0,
        stake_amount=25.0,
        payout_amount=41.67,
        net_pnl=16.67,
        status=LedgerStatus.WON,
        provenance=BetProvenance.USER_CONFIRMED,
        decision_timestamp="2026-03-21T12:00:00Z",
        staking_policy_id="sp-flat-1",
        staking_policy_version=1,
        exposure_limits_version=2,
        sizing_reasons=["max_exposure_cap"],
        correlation_group="nba-2026-03-21",
    )
    store.record_ledger_bet(bet)
    store.close()
    return {"db_path": db_path, "sessions_dir": sessions_dir}


@pytest.fixture
def app(seeded: dict[str, Any]):
    return build_console_app(
        db_path=seeded["db_path"], sessions_dir=str(seeded["sessions_dir"])
    )


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)
