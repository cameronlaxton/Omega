"""
Tests for closing_lines persistence on TraceStore (Phase 6e).

Covers:
- attach_closing_line round-trips.
- Idempotent on (trace_id, market, selection_descriptor) — second call returns
  the same closing_id and does not create a duplicate row.
- attach_closing_line on a missing trace_id raises ValueError.
- get_closing_lines returns all snapshots for a trace.
"""
from __future__ import annotations

import tempfile
from typing import Any

from omega.trace.store import TraceStore


def _tmp_store() -> TraceStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return TraceStore(db_path=tmp.name)


def _make_minimal_trace(trace_id: str = "t-clv-001") -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-14T19:00:00Z",
        "prompt": "test",
        "league": "NBA",
        "matchup": "Celtics @ Lakers",
        "execution_mode": "sandbox_game",
        "simulation_seed": 1,
        "aggregate_quality": 0.8,
        "downgrades": [],
    }


class TestAttachClosingLine:
    def test_round_trip(self):
        store = _tmp_store()
        store.persist(_make_minimal_trace("t-clv-001"))

        cid = store.attach_closing_line(
            trace_id="t-clv-001",
            market="spread",
            selection_descriptor="home_spread_-3.5",
            closing_odds=-110.0,
            closing_line=-5.5,
            closing_timestamp="2026-05-14T22:55:00Z",
            source="the-odds-api:draftkings",
        )
        assert cid

        rows = store.get_closing_lines("t-clv-001")
        assert len(rows) == 1
        row = rows[0]
        assert row["closing_id"] == cid
        assert row["market"] == "spread"
        assert row["selection_descriptor"] == "home_spread_-3.5"
        assert row["closing_odds"] == -110.0
        assert row["closing_line"] == -5.5
        assert row["source"] == "the-odds-api:draftkings"
        store.close()

    def test_missing_trace_raises(self):
        store = _tmp_store()
        try:
            store.attach_closing_line(
                trace_id="does-not-exist",
                market="moneyline",
                selection_descriptor="home_moneyline",
                closing_odds=-150.0,
                closing_line=None,
                closing_timestamp="2026-05-14T22:55:00Z",
                source="the-odds-api:fanduel",
            )
        except ValueError as exc:
            assert "does-not-exist" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        store.close()

    def test_idempotent_on_unique_key(self):
        store = _tmp_store()
        store.persist(_make_minimal_trace("t-clv-002"))

        cid1 = store.attach_closing_line(
            trace_id="t-clv-002",
            market="total",
            selection_descriptor="total_over_226.5",
            closing_odds=-105.0,
            closing_line=226.5,
            closing_timestamp="2026-05-14T22:55:00Z",
            source="the-odds-api:draftkings",
        )
        # Re-call with different odds — should keep the first close
        cid2 = store.attach_closing_line(
            trace_id="t-clv-002",
            market="total",
            selection_descriptor="total_over_226.5",
            closing_odds=-120.0,  # different value
            closing_line=227.5,
            closing_timestamp="2026-05-14T23:55:00Z",
            source="the-odds-api:fanduel",
        )
        assert cid1 == cid2

        rows = store.get_closing_lines("t-clv-002")
        assert len(rows) == 1
        assert rows[0]["closing_odds"] == -105.0  # first wins
        store.close()

    def test_multiple_markets_per_trace(self):
        store = _tmp_store()
        store.persist(_make_minimal_trace("t-clv-003"))

        store.attach_closing_line(
            trace_id="t-clv-003",
            market="moneyline",
            selection_descriptor="home_moneyline",
            closing_odds=-150.0,
            closing_line=None,
            closing_timestamp="2026-05-14T22:55:00Z",
            source="the-odds-api:draftkings",
        )
        store.attach_closing_line(
            trace_id="t-clv-003",
            market="total",
            selection_descriptor="total_over_226.5",
            closing_odds=-110.0,
            closing_line=226.5,
            closing_timestamp="2026-05-14T22:55:00Z",
            source="the-odds-api:draftkings",
        )
        rows = store.get_closing_lines("t-clv-003")
        assert len(rows) == 2
        markets = sorted(r["market"] for r in rows)
        assert markets == ["moneyline", "total"]
        store.close()
