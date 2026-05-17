from __future__ import annotations

from omega.trace.market_snapshot import MarketSnapshot
from omega.trace.store import TraceStore


def test_market_snapshot_idempotency_and_movement(tmp_path):
    store = TraceStore(db_path=str(tmp_path / "traces.db"))
    first = MarketSnapshot(
        league="NBA",
        provider_event_id="evt-1",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        bookmaker="betmgm",
        market="spreads",
        selection="Los Angeles Lakers",
        point=-3.5,
        price=-110,
        snapshot_timestamp="2026-05-17T20:00:00Z",
        provider_last_update="2026-05-17T20:00:00Z",
        source="the-odds-api:betmgm",
    )
    second = first.model_copy(
        update={
            "point": -4.5,
            "price": -115,
            "snapshot_timestamp": "2026-05-17T21:00:00Z",
            "provider_last_update": "2026-05-17T21:00:00Z",
        }
    )

    first_id = store.record_market_snapshot(first)
    assert store.record_market_snapshot(first) == first_id
    store.record_market_snapshot(second)

    rows = store.get_market_snapshots(
        provider_event_id="evt-1",
        market="spreads",
        bookmaker="betmgm",
        selection="Los Angeles Lakers",
    )
    movement = store.compute_market_movement("evt-1", "spreads", "Los Angeles Lakers", "betmgm")

    assert len(rows) == 2
    assert movement is not None
    assert movement["point_delta"] == -1.0
    assert movement["price_delta"] == -5.0
    store.close()
