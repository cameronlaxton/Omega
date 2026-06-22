"""Milestone B.3 — Signal Performance page: latest scoring-run read view."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.strategy.signal_performance import SignalPerformanceRow
from omega.trace.store import TraceStore


def _row(signal_type: str, league: str = "NBA", **kw) -> SignalPerformanceRow:
    base = dict(
        signal_type=signal_type,
        source="espn",
        obs_window="last_5",
        league=league,
        sample_size=40,
        direction_correct=26,
        direction_accuracy=0.65,
        mean_confidence=0.7,
        realized_hit_rate=0.62,
        calibration_gap=0.05,
        brier=0.22,
    )
    base.update(kw)
    return SignalPerformanceRow(**base)


def _client(tmp_path: Path, rows=()) -> TestClient:
    db = str(tmp_path / "b3.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    if rows:
        store.upsert_signal_performance(list(rows), dataset_hash="ds-test")
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def test_signals_returns_latest_run_rows(tmp_path):
    client = _client(tmp_path, rows=[_row("injury"), _row("weather", league="NBA")])
    body = client.get("/api/signals").json()
    assert len(body["rows"]) == 2
    assert body["last_scored_at"] is not None
    injury = next(r for r in body["rows"] if r["signal_type"] == "injury")
    assert injury["direction_accuracy"] == 0.65
    assert injury["realized_hit_rate"] == 0.62
    assert injury["brier"] == 0.22
    assert injury["sample_size"] == 40


def test_signals_empty_state_warns(tmp_path):
    body = _client(tmp_path).get("/api/signals").json()
    assert body["rows"] == []
    assert body["last_scored_at"] is None
    assert "no_scoring_run" in {w["code"] for w in body["warnings"]}


def test_signals_filter_by_league(tmp_path):
    client = _client(tmp_path, rows=[_row("injury", league="NBA"), _row("form", league="EPL")])
    nba = client.get("/api/signals?league=NBA").json()["rows"]
    assert {r["signal_type"] for r in nba} == {"injury"}


def test_signals_page_renders(tmp_path):
    html = _client(tmp_path, rows=[_row("injury")]).get("/signals").text
    assert "<h1>Signal Performance</h1>" in html
    assert "injury" in html
    assert 'href="/signals"' in html  # nav enabled
