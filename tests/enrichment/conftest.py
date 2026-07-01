"""Fixtures for the enrichment subsystem tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from omega.trace.store import TraceStore
from tests.ui.conftest import make_trace


def seed_traces_db(path: str, trace_id: str = "enr-1") -> None:
    """Seed a temp canonical trace DB with one realistic trace."""
    store = TraceStore(db_path=path)
    with store.autolog_suppressed():
        store.persist(
            make_trace(
                trace_id,
                league="NBA",
                kind="game",
                recommendations=[
                    {"side": "home", "market": "moneyline", "confidence_tier": "B",
                     "edge_pct": 4.2, "odds": -150}
                ],
            )
        )
    store.close()


@pytest.fixture
def traces_db(tmp_path: Path) -> str:
    path = str(tmp_path / "traces.db")
    seed_traces_db(path)
    return path


@pytest.fixture
def enrich_db(tmp_path: Path) -> str:
    return str(tmp_path / "enrichments.db")
