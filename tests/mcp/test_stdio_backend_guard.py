from __future__ import annotations

import os

from omega.mcp.server import _insulate_stdio_backend
from omega.trace.db import resolve_backend

POSTGRES_URL = "postgresql+psycopg://omega:omega@localhost:5432/omega"


def test_stdio_ignores_inherited_database_url(monkeypatch):
    """Antigravity's stdio server must stay on SQLite even if the parent shell
    exported DATABASE_URL, unless OMEGA_MCP_ALLOW_DB_BACKEND=1 is set."""
    monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)
    monkeypatch.delenv("OMEGA_MCP_ALLOW_DB_BACKEND", raising=False)

    _insulate_stdio_backend()

    assert "DATABASE_URL" not in os.environ
    assert resolve_backend().backend == "sqlite"


def test_stdio_honors_database_url_when_opted_in(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)
    monkeypatch.setenv("OMEGA_MCP_ALLOW_DB_BACKEND", "1")

    _insulate_stdio_backend()

    assert os.environ["DATABASE_URL"] == POSTGRES_URL
    assert resolve_backend().backend == "postgres"
