"""V2 Command Center — composed landing summary with per-panel failure isolation.

The Command Center introduces no new DB access; it aggregates existing read
views. These tests pin three doctrine properties: (1) it renders every panel,
(2) one failing panel degrades only itself (the page never 500s), and (3)
missing data renders as an explicit empty state, never a fabricated zero.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from omega.ui.service import ConsoleService


def _service(db_path: str, sessions_dir: Path) -> ConsoleService:
    return ConsoleService(TraceStore(db_path=db_path, read_only=True), sessions_dir=str(sessions_dir))


def test_command_center_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "Sports Analytics Platform" in html
    for title in (
        "Live Market Scanner",
        "Interactive Review Queue",
        "Calibration Tuning",
        "Trace QA Queue",
    ):
        assert title in html, f"missing panel: {title}"


def test_command_center_composes_panels(seeded):
    svc = _service(seeded["db_path"], seeded["sessions_dir"])
    try:
        cc = svc.command_center()
    finally:
        svc.close()
    assert set(cc.panels) >= {"review", "diagnostics", "calibration", "ledger", "failures"}
    # Seeded DB: 3 traces (1 graded) + 1 settled bet → review has work, ledger has a bet.
    assert cc.health is not None and cc.health.trace_count == 3
    assert cc.review_count > 0
    assert cc.panels["review"].state == "data"
    assert cc.panels["diagnostics"].state == "data"
    assert cc.panels["ledger"].state == "data"


def test_command_center_isolates_failing_panel(seeded):
    svc = _service(seeded["db_path"], seeded["sessions_dir"])

    def boom(*_a, **_k):
        raise RuntimeError("simulated source read failure")

    svc.clv_report = boom  # type: ignore[method-assign]
    try:
        cc = svc.command_center()  # must NOT raise
    finally:
        svc.close()
    assert cc.panels["ledger"].state == "degraded"
    assert "unavailable" in (cc.panels["ledger"].message or "").lower()
    # Other panels are unaffected by the one failure.
    assert cc.panels["review"].state in {"data", "empty"}
    assert cc.panels["diagnostics"].state in {"data", "empty"}


def test_command_center_route_survives_panel_failure(client, monkeypatch):
    def boom(self, *_a, **_k):
        raise RuntimeError("simulated source read failure")

    monkeypatch.setattr(ConsoleService, "edge_scanner", boom)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Live Market Scanner" in resp.text
    assert "unavailable" in resp.text.lower()


def test_command_center_empty_states_are_honest(tmp_path):
    db = str(tmp_path / "empty.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    TraceStore(db_path=db).close()  # create empty schema, no rows
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    html = client.get("/").text
    # Honest empty states (muted), not zeros masquerading as data.
    assert "Nothing needs review right now." in html
    assert "No recent DB-backed recommendations found." in html
