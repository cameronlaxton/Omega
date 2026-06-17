"""As-of player-prop context builder: no leakage, no fabricated context."""

from __future__ import annotations

import pytest

from omega.historical.adapters.csv_player_stats import PlayerStatObservation
from omega.historical.contracts import HistoricalEvent, HistoricalPropMarket
from omega.historical.identity import event_key
from omega.historical.normalize import parse_datetime_utc
from omega.historical.prop_context import (
    PropContextBuildConfig,
    build_prop_context,
    prop_context_key,
    targets_from_prop_markets,
)

LG, FAM = "NBA", "basketball"


def _event(date: str, home: str, away: str, season: str = "2024") -> HistoricalEvent:
    start = parse_datetime_utc(date)
    return HistoricalEvent(
        event_id=event_key(LG, start, home, away),
        league=LG,
        sport_family=FAM,
        season=season,
        start_time=start,
        home_team=home,
        away_team=away,
        source_name="test",
    )


def _obs(event: HistoricalEvent, value: float, *, player: str = "Player A"):
    return PlayerStatObservation(
        event_key=event.event_id,
        date=event.start_time,
        player_name=player,
        stat_type="pts",
        stat_value=value,
        player_id="p1",
        season=event.season,
    )


def test_prop_context_filters_strictly_before_decision_time():
    e1 = _event("2024-01-01", "A", "B")
    e2 = _event("2024-01-08", "C", "A")
    e3 = _event("2024-01-15", "A", "D")
    target = _event("2024-01-22", "A", "E")
    future = _event("2024-01-29", "A", "F")
    markets = {
        target.event_id: [
            HistoricalPropMarket(
                event_key=target.event_id,
                player_name="Player A",
                stat_type="pts",
                line=24.5,
            )
        ]
    }
    observations = [
        _obs(e1, 20.0),
        _obs(e2, 22.0),
        _obs(e3, 24.0),
        _obs(target, 99.0),  # same event row must not enter pre-decision context
        _obs(future, 120.0),
    ]

    result = build_prop_context(
        manifest_id="m",
        league=LG,
        targets=targets_from_prop_markets([e1, e2, e3, target], markets),
        observations=observations,
        config=PropContextBuildConfig(lookback_games=10, min_history_games=2),
    )

    ctx = result.context[prop_context_key(target.event_id, "Player A", "pts")]
    assert ctx["pts_mean"] == pytest.approx(22.0)
    assert ctx["pts_median"] == pytest.approx(22.0)
    assert ctx["as_of"] == e3.start_time
    assert ctx["sample_size"] == 3
    assert result.audit.missing_context_rate == 0.0


def test_low_sample_context_is_marked_without_fabricated_std():
    e1 = _event("2024-01-01", "A", "B")
    target = _event("2024-01-22", "A", "E")
    markets = {
        target.event_id: [
            HistoricalPropMarket(
                event_key=target.event_id,
                player_name="Player A",
                stat_type="pts",
                line=24.5,
            )
        ]
    }

    result = build_prop_context(
        manifest_id="m",
        league=LG,
        targets=targets_from_prop_markets([e1, target], markets),
        observations=[_obs(e1, 20.0)],
        config=PropContextBuildConfig(lookback_games=10, min_history_games=5),
    )

    ctx = result.context[prop_context_key(target.event_id, "Player A", "pts")]
    assert ctx["pts_mean"] == pytest.approx(20.0)
    assert "pts_std" not in ctx
    assert ctx["missing_keys"] == ["pts_std"]
    assert ctx["is_low_sample"] is True
    assert ctx["is_imputed"] is False
    assert ctx["imputed_keys"] == []
    assert result.audit.sample_size_buckets["1-4"] == 1


def test_missing_context_entry_omits_required_mean():
    target = _event("2024-01-22", "A", "E")
    markets = {
        target.event_id: [
            HistoricalPropMarket(
                event_key=target.event_id,
                player_name="Missing Player",
                stat_type="pts",
                line=24.5,
            )
        ]
    }

    result = build_prop_context(
        manifest_id="m",
        league=LG,
        targets=targets_from_prop_markets([target], markets),
        observations=[],
    )

    ctx = result.context[prop_context_key(target.event_id, "Missing Player", "pts")]
    assert "pts_mean" not in ctx
    assert ctx["missing_context"] is True
    assert ctx["missing_keys"] == ["pts_mean", "pts_std"]
    assert result.audit.missing_context_rate == 1.0
