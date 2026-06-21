"""Graceful degradation: a missing/unopenable DB must not 500 or write."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore


def _missing_db_client(tmp_path: Path) -> TestClient:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    app = build_console_app(db_path=str(tmp_path / "nope.db"), sessions_dir=str(sessions))
    # raise_server_exceptions=False so a 500 surfaces as a response, not a raise.
    return TestClient(app, raise_server_exceptions=False)


def test_readiness_healthz_degrades_not_500_on_missing_db(tmp_path: Path):
    resp = _missing_db_client(tmp_path).get("/api/healthz")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "degraded"
    assert resp.json()["read_only"] is True


def test_liveness_healthz_ok_without_db(tmp_path: Path):
    resp = _missing_db_client(tmp_path).get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readonly_open_does_not_create_missing_db(tmp_path: Path):
    p = tmp_path / "nope.db"
    store = TraceStore(db_path=str(p), read_only=True)
    try:
        assert p.exists() is False  # read-only open must not create the file
    finally:
        store.close()
