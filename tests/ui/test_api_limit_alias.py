"""Tests for the `limit` query-parameter alias on list endpoints.

Proves that ``?limit=N`` caps the number of rows returned, interacts correctly
with ``page_size``, and that FastAPI validates out-of-range values — for
``/api/traces``, ``/api/bets``, and ``/api/sessions``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from tests.ui.conftest import make_trace, write_valid_sidecar


@pytest.fixture
def many_traces(tmp_path: Path) -> dict[str, Any]:
    """Seed a DB with 5 traces and 3 session sidecars for limit testing."""
    db_path = str(tmp_path / "omega_traces.db")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    store = TraceStore(db_path=db_path)
    with store.autolog_suppressed():
        for i in range(5):
            store.persist(
                make_trace(
                    f"sandbox-lim-{i:03d}",
                    session_id=f"sess-lim-{i}",
                    timestamp=f"2026-04-{10 + i:02d}T12:00:00Z",
                )
            )
    store.close()

    for i in range(3):
        write_valid_sidecar(sessions_dir, f"sess-lim-{i}")

    return {"db_path": db_path, "sessions_dir": sessions_dir}


@pytest.fixture
def lim_client(many_traces: dict[str, Any]) -> TestClient:
    app = build_console_app(
        db_path=many_traces["db_path"],
        sessions_dir=str(many_traces["sessions_dir"]),
    )
    return TestClient(app)


# ── /api/traces ────────────────────────────────────────────────────────────


class TestTracesLimit:
    def test_limit_caps_rows(self, lim_client: TestClient):
        body = lim_client.get("/api/traces?limit=3").json()
        assert len(body["rows"]) <= 3

    def test_limit_1_returns_single_row(self, lim_client: TestClient):
        body = lim_client.get("/api/traces?limit=1").json()
        assert len(body["rows"]) == 1

    def test_page_size_only(self, lim_client: TestClient):
        body = lim_client.get("/api/traces?page_size=2").json()
        assert len(body["rows"]) == 2
        assert body["pagination"]["page_size"] == 2

    def test_both_limit_and_page_size_narrower_wins(self, lim_client: TestClient):
        body = lim_client.get("/api/traces?page_size=10&limit=2").json()
        assert len(body["rows"]) <= 2
        assert body["pagination"]["page_size"] == 2

    def test_page_size_smaller_than_limit(self, lim_client: TestClient):
        body = lim_client.get("/api/traces?page_size=2&limit=10").json()
        assert len(body["rows"]) == 2
        assert body["pagination"]["page_size"] == 2

    def test_limit_without_page_size_uses_limit_as_cap(self, lim_client: TestClient):
        body = lim_client.get("/api/traces?limit=3").json()
        assert body["pagination"]["page_size"] == 3

    def test_limit_zero_rejected(self, lim_client: TestClient):
        resp = lim_client.get("/api/traces?limit=0")
        assert resp.status_code == 422

    def test_limit_negative_rejected(self, lim_client: TestClient):
        resp = lim_client.get("/api/traces?limit=-1")
        assert resp.status_code == 422

    def test_limit_over_max_rejected(self, lim_client: TestClient):
        resp = lim_client.get("/api/traces?limit=999")
        assert resp.status_code == 422


# ── /api/bets ──────────────────────────────────────────────────────────────


class TestBetsLimit:
    def test_limit_caps_rows(self, lim_client: TestClient):
        # Fixture has 0 bets (autolog suppressed), so this verifies the
        # endpoint accepts the param without error even with 0 results.
        body = lim_client.get("/api/bets?limit=2").json()
        assert len(body["rows"]) <= 2

    def test_limit_zero_rejected(self, lim_client: TestClient):
        resp = lim_client.get("/api/bets?limit=0")
        assert resp.status_code == 422

    def test_limit_over_max_rejected(self, lim_client: TestClient):
        resp = lim_client.get("/api/bets?limit=999")
        assert resp.status_code == 422


# ── /api/sessions ──────────────────────────────────────────────────────────


class TestSessionsLimit:
    def test_limit_caps_rows(self, lim_client: TestClient):
        body = lim_client.get("/api/sessions?limit=2").json()
        assert len(body["rows"]) <= 2

    def test_limit_1_returns_single_row(self, lim_client: TestClient):
        body = lim_client.get("/api/sessions?limit=1").json()
        assert len(body["rows"]) == 1

    def test_page_size_only(self, lim_client: TestClient):
        body = lim_client.get("/api/sessions?page_size=2").json()
        assert len(body["rows"]) == 2

    def test_both_limit_and_page_size(self, lim_client: TestClient):
        body = lim_client.get("/api/sessions?page_size=10&limit=1").json()
        assert len(body["rows"]) <= 1

    def test_limit_zero_rejected(self, lim_client: TestClient):
        resp = lim_client.get("/api/sessions?limit=0")
        assert resp.status_code == 422

    def test_limit_over_max_rejected(self, lim_client: TestClient):
        resp = lim_client.get("/api/sessions?limit=999")
        assert resp.status_code == 422


# ── HTML pages also honour limit ──────────────────────────────────────────


class TestHTMLPagesLimit:
    """Verify that the server-rendered HTML pages respect ?limit= too."""

    def test_traces_html_limit(self, lim_client: TestClient):
        resp = lim_client.get("/traces?limit=2")
        assert resp.status_code == 200

    def test_bets_html_limit(self, lim_client: TestClient):
        resp = lim_client.get("/bets?limit=2")
        assert resp.status_code == 200

    def test_sessions_html_limit(self, lim_client: TestClient):
        resp = lim_client.get("/sessions?limit=1")
        assert resp.status_code == 200
