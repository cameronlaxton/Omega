"""Milestone 1 — early_market_snapshots isolation (red-team finding 4).

Asserts that early low-liquidity captures land ONLY in early_market_snapshots,
never in closing_lines, and that the CLV-relevant closing-line read is bit-
identical whether the early table holds 0 or many rows.

References:
  omega/trace/schema.py (SCHEMA_V11)
  omega/trace/store.py (record_early_market_snapshot / get_early_market_snapshots)
  docs/phase7/MULTI_SPORT_EXPANSION.md (Milestone 1 acceptance; red-team finding 4)
"""

from __future__ import annotations

import json

import pytest

from omega.trace.market_snapshot import EarlyMarketSnapshot
from omega.trace.schema import CURRENT_VERSION
from omega.trace.store import TraceStore


@pytest.fixture()
def store(tmp_path):
    s = TraceStore(db_path=str(tmp_path / "traces.db"))
    yield s
    s.close()


def _seed_trace(store: TraceStore, trace_id: str) -> None:
    store.conn.execute(
        """INSERT INTO traces (trace_id, run_id, timestamp, prompt, league, full_trace)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (trace_id, "run1", "2026-05-20T00:00:00Z", "p", "WNBA", json.dumps({})),
    )
    store.conn.commit()


def _early(trace_id: str, side: str, captured_at: str) -> EarlyMarketSnapshot:
    return EarlyMarketSnapshot(
        trace_id=trace_id,
        league="WNBA",
        market="total",
        selection_descriptor=f"total:{side}:160.5",
        early_line=160.5,
        early_odds=-110,
        liquidity_profile="low",
        captured_at=captured_at,
        source="the-odds-api:test",
    )


def test_schema_is_current(store):
    assert store.schema_version() == CURRENT_VERSION


def test_early_capture_does_not_write_closing_lines(store):
    _seed_trace(store, "sandbox-w1")
    store.record_early_market_snapshot(_early("sandbox-w1", "over", "2026-05-20T10:00:00Z"))
    store.record_early_market_snapshot(_early("sandbox-w1", "under", "2026-05-20T10:00:00Z"))

    # Early rows exist...
    assert len(store.get_early_market_snapshots("sandbox-w1")) == 2
    # ...but closing_lines is untouched.
    closing = store.get_closing_lines("sandbox-w1")
    assert closing == []
    count = store.conn.execute("SELECT COUNT(*) FROM closing_lines").fetchone()[0]
    assert count == 0


def test_record_is_idempotent(store):
    _seed_trace(store, "sandbox-w2")
    snap = _early("sandbox-w2", "over", "2026-05-20T10:00:00Z")
    id1 = store.record_early_market_snapshot(snap)
    id2 = store.record_early_market_snapshot(snap)
    assert id1 == id2
    assert len(store.get_early_market_snapshots("sandbox-w2")) == 1


def test_clv_read_is_bit_identical_with_or_without_early_rows(store):
    """The closing-line read that feeds CLV must ignore early_market_snapshots."""
    _seed_trace(store, "sandbox-w3")
    # Attach a real closing line.
    store.attach_closing_line(
        trace_id="sandbox-w3",
        market="total",
        selection_descriptor="total:over:160.5",
        closing_odds=-108,
        closing_line=159.5,
        closing_timestamp="2026-05-20T22:00:00Z",
        source="the-odds-api:draftkings",
    )
    before = store.get_closing_lines("sandbox-w3")

    # Flood the early table with 50 rows for the same trace.
    for i in range(50):
        store.record_early_market_snapshot(
            _early("sandbox-w3", f"over_{i}", f"2026-05-20T10:{i:02d}:00Z")
        )

    after = store.get_closing_lines("sandbox-w3")
    assert before == after, "early_market_snapshots leaked into the CLV closing-line read"
    assert len(after) == 1
