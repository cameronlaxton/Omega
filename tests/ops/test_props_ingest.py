"""Prop ingest plumbing: adapter read_prop_markets + dataset persistence round-trip."""

from __future__ import annotations

from omega.historical.adapters.csv_player_stats import CsvPlayerStatsAdapter
from omega.historical.contracts import HistoricalEvent, HistoricalOutcome, HistoricalPropMarket
from omega.historical.manifests import load_normalized_dataset, save_normalized_dataset

_CSV = (
    "date,home_team,away_team,player_name,stat_type,line,over_price,under_price,"
    "book,timestamp,tier_hint\n"
    "2024-01-22,Lakers,Heat,LeBron James,pts,24.5,-110,-110,"
    "betmgm,2024-01-21T18:00:00Z,opening\n"
    "2024-01-22,Lakers,Heat,No Line Guy,pts,,-110,-110,"
    "betmgm,2024-01-21T18:00:00Z,opening\n"  # missing line -> skipped
)

_VOID_CSV = (
    "date,home_team,away_team,player_name,stat_type,stat_value,void\n"
    "2024-01-22,Lakers,Heat,Bench Guy,pts,,true\n"
)


def test_read_prop_markets_skips_lineless_rows(tmp_path):
    csv_path = tmp_path / "markets.csv"
    csv_path.write_text(_CSV, encoding="utf-8")
    markets = CsvPlayerStatsAdapter("NBA").read_prop_markets(str(csv_path))
    all_markets = [m for ms in markets.values() for m in ms]
    assert len(all_markets) == 1
    m = all_markets[0]
    assert m.player_name == "LeBron James"
    assert m.line == 24.5
    assert m.over_price == -110.0
    assert m.book == "betmgm"
    assert m.timestamp == "2024-01-21T18:00:00+00:00"
    assert m.tier_hint == "opening"


def test_read_prop_outcomes_preserves_void_rows(tmp_path):
    csv_path = tmp_path / "outcomes.csv"
    csv_path.write_text(_VOID_CSV, encoding="utf-8")
    outcomes = CsvPlayerStatsAdapter("NBA").read_prop_outcomes(str(csv_path))
    all_outcomes = [o for rows in outcomes.values() for o in rows]
    assert len(all_outcomes) == 1
    assert all_outcomes[0].player_name == "Bench Guy"
    assert all_outcomes[0].stat_value is None
    assert all_outcomes[0].void is True


def test_normalized_dataset_round_trips_props(tmp_path):
    ev = HistoricalEvent(
        event_id="evt1", league="NBA", sport_family="basketball",
        start_time="2024-01-22T00:00:00+00:00", home_team="Lakers", away_team="Heat",
        source_name="test",
    )
    oc = HistoricalOutcome(event_id="evt1", home_score=112, away_score=104)
    market = HistoricalPropMarket(
        event_key="evt1", player_name="LeBron James", stat_type="pts", line=24.5
    )
    ctx = {"evt1|LeBron James|pts": {"pts_mean": 27.0}}
    save_normalized_dataset(
        "m1", events=[ev], outcomes=[oc],
        prop_markets={"evt1": [market]}, prop_context=ctx, root=tmp_path,
    )
    loaded = load_normalized_dataset("m1", root=tmp_path)
    assert loaded["prop_markets"]["evt1"][0].line == 24.5
    assert loaded["prop_context"] == ctx
