"""Milestone B.3 — Review Queue: operator work buckets from existing reads."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore
from tests.ui.conftest import make_trace


def _client(tmp_path: Path, *, setup=None) -> TestClient:
    db = str(tmp_path / "b3.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        if setup:
            setup(store, sessions)
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def _buckets(client: TestClient) -> dict:
    return {b["code"]: b for b in client.get("/api/review").json()["buckets"]}


def test_review_buckets_count_and_sample(tmp_path):
    def setup(store, sessions):
        store.persist(make_trace("ungraded-1", kind="game", matchup="A @ B"))  # no outcome
        graded = make_trace("graded-1", kind="game")
        store.persist(graded)
        store.attach_outcome("graded-1", 100, 90)
        store.record_ledger_bet(
            LedgerBet(
                ledger_id="led-pending",
                trace_id="ungraded-1",
                bet_date="2026-03-21",
                league="NBA",
                sport="basketball",
                matchup="A @ B",
                market="moneyline",
                bookmaker="dk",
                selection="A ML",
                selection_descriptor="home_moneyline",
                odds=-110.0,
                stake_amount=10.0,
                status=LedgerStatus.PENDING,
                provenance=BetProvenance.USER_CONFIRMED,
                decision_timestamp="2026-03-21T12:00:00Z",
            )
        )
        (sessions / "sess-bad.json").write_text("{ not json", encoding="utf-8")

    buckets = _buckets(_client(tmp_path, setup=setup))
    assert buckets["ungraded_traces"]["count"] == 1
    assert buckets["ungraded_traces"]["items"][0]["id"] == "ungraded-1"
    assert buckets["ungraded_traces"]["items"][0]["href"] == "/traces/ungraded-1"
    assert buckets["pending_bets"]["count"] == 1
    assert buckets["pending_bets"]["severity"] == "warn"
    assert buckets["pending_bets"]["items"][0]["id"] == "led-pending"
    assert buckets["problem_sessions"]["count"] == 1
    assert buckets["problem_sessions"]["items"][0]["id"] == "sess-bad"


def test_review_all_clear(tmp_path):
    buckets = _buckets(_client(tmp_path))
    for code in ("ungraded_traces", "pending_bets", "problem_sessions"):
        assert buckets[code]["count"] == 0
        assert buckets[code]["items"] == []


def test_review_gate_fail_session_flagged(tmp_path):
    import json

    def setup(store, sessions):
        # A valid sidecar whose quality gate failed → flagged as a problem session.
        payload = {
            "session_id": "sess-fail",
            "opened_at": "2026-03-21T11:00:00Z",
            "closed_at": None,
            "model_version": "m",
            "purpose": "t",
            "league": "NBA",
            "window": None,
            "effective_db_path": None,
            "runtime_db_status": None,
            "pipeline_status": {},
            "next_required_action": None,
            "bankroll": 1000.0,
            "bankroll_confirmed": True,
            "exec_stats": {},
            "agent_notes": "",
            "audit_events": [
                {
                    "ts": "2026-03-21T11:30:00Z",
                    "event_type": "quality_gate",
                    "step": "final",
                    "status": "fail",
                }
            ],
        }
        (sessions / "sess-fail.json").write_text(json.dumps(payload), encoding="utf-8")

    buckets = _buckets(_client(tmp_path, setup=setup))
    problems = buckets["problem_sessions"]
    assert problems["count"] == 1
    assert problems["items"][0]["id"] == "sess-fail"
    assert "gate" in problems["items"][0]["detail"].lower()


def test_review_page_renders(tmp_path):
    html = _client(tmp_path).get("/review").text
    assert "<h1>Review Queue</h1>" in html
    assert "Ungraded traces" in html and "Pending bets" in html
