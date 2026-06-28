"""Data-quality heatmap — per-league coverage of evidence / closing line / outcome.

Pins the doctrine: cells are real counts (green is measured coverage, never an
assumed default), an empty source renders an honest empty state, and the page is
GET-only and read-only.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from omega.ui.service import ConsoleService, _quality_tone
from tests.ui.conftest import make_trace


def test_quality_tone_thresholds():
    assert _quality_tone(None) == "muted"  # nothing to measure -> never green
    assert _quality_tone(0.0) == "bad"
    assert _quality_tone(0.49) == "bad"
    assert _quality_tone(0.5) == "warn"
    assert _quality_tone(0.79) == "warn"
    assert _quality_tone(0.8) == "good"
    assert _quality_tone(1.0) == "good"


def _seed_two_leagues(db: str) -> None:
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        store.persist(make_trace("n1", league="NBA"))
        store.persist(make_trace("n2", league="NBA"))
        store.persist(make_trace("e1", league="EPL"))
    store.attach_outcome("n1", 1, 0)
    store.attach_closing_line(
        "n1", "moneyline", "home_moneyline", -110.0, None, "2026-03-21T19:00:00Z", "dk"
    )
    store.attach_closing_line(
        "n2", "moneyline", "home_moneyline", -110.0, None, "2026-03-21T19:00:00Z", "dk"
    )
    store.close()


def _svc(db: str, sessions: Path) -> ConsoleService:
    return ConsoleService(TraceStore(db_path=db, read_only=True), sessions_dir=str(sessions))


def test_data_quality_aggregates_per_league(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed_two_leagues(db)
    svc = _svc(db, sessions)
    try:
        dq = svc.data_quality()
    finally:
        svc.close()
    by_league = {r.league: r for r in dq.rows}
    assert set(by_league) == {"NBA", "EPL"}

    nba = by_league["NBA"]
    cells = {c.key: c for c in nba.cells}
    assert nba.traces == 2
    assert cells["closing_line"].count == 2 and cells["closing_line"].tone == "good"  # 2/2
    assert cells["outcome"].count == 1  # 1/2 graded
    assert cells["evidence"].count == 0 and cells["evidence"].tone == "bad"  # none attached

    epl = by_league["EPL"]
    assert {c.key: c.count for c in epl.cells}["closing_line"] == 0


def test_data_quality_empty_db_is_honest(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    TraceStore(db_path=db).close()
    svc = _svc(db, sessions)
    try:
        dq = svc.data_quality()
    finally:
        svc.close()
    assert dq.rows == []
    assert any(w.code == "no_traces" for w in dq.warnings)


def test_data_quality_page_and_nav(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed_two_leagues(db)
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    html = client.get("/data-quality").text
    assert "<h1>Data Quality</h1>" in html
    assert "heat-cell" in html and "tone-good" in html
    assert 'href="/data-quality"' in html  # nav item present


def test_data_quality_api_read_only(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed_two_leagues(db)
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    assert client.get("/api/data-quality").status_code == 200
    assert client.post("/api/data-quality", json={}).status_code == 405
