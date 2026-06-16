"""Shared fixtures for the historical replay + backtest test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from omega.trace.store import TraceStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    """Directory holding tiny golden CSV fixtures mirroring real source schemas."""
    return FIXTURES


@pytest.fixture()
def backtest_store(tmp_path: Path) -> TraceStore:
    """An isolated backtest TraceStore (never the production DB)."""
    store = TraceStore(db_path=str(tmp_path / "backtest.db"))
    try:
        yield store
    finally:
        store.close()


@pytest.fixture()
def nfl_alias_table() -> dict:
    """Minimal NFL team alias table used by identity tests (passed explicitly)."""
    return {
        "canonical": [
            "Kansas City Chiefs",
            "San Francisco 49ers",
            "Philadelphia Eagles",
        ],
        "aliases": {
            "KC": "Kansas City Chiefs",
            "Chiefs": "Kansas City Chiefs",
            "SF": "San Francisco 49ers",
            "49ers": "San Francisco 49ers",
            "Eagles": "Philadelphia Eagles",
        },
    }
