from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

from omega.integrations.odds_api import (
    OddsApiClient,
    parse_event_markets,
    parse_events,
    parse_events_metadata,
    parse_historical_events,
    parse_historical_snapshot,
    parse_sports,
)


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_opener(captured: dict, payload: dict | list):
    def opener(url: str, timeout: int):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    return opener


def test_parse_events_uses_market_last_update_when_present():
    events = parse_events(
        [
            {
                "id": "evt-1",
                "sport_key": "basketball_nba",
                "commence_time": "2026-05-16T23:00:00Z",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "spreads",
                                "last_update": "2026-05-16T22:45:00Z",
                                "outcomes": [
                                    {
                                        "name": "Los Angeles Lakers",
                                        "price": -110,
                                        "point": -3.5,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )

    assert len(events) == 1
    assert events[0].event_id == "evt-1"
    assert events[0].books[0].last_update == "2026-05-16T22:45:00Z"
    assert events[0].books[0].point == -3.5


def test_parse_historical_snapshot_accepts_wrapped_list():
    snapshot = parse_historical_snapshot(
        {
            "timestamp": "2026-05-16T22:55:00Z",
            "previous_timestamp": "2026-05-16T22:50:00Z",
            "next_timestamp": "2026-05-16T23:00:00Z",
            "data": [
                {
                    "id": "evt-1",
                    "sport_key": "basketball_nba",
                    "home_team": "Los Angeles Lakers",
                    "away_team": "Boston Celtics",
                    "bookmakers": [],
                }
            ],
        }
    )

    assert snapshot.timestamp == "2026-05-16T22:55:00Z"
    assert snapshot.previous_timestamp == "2026-05-16T22:50:00Z"
    assert len(snapshot.events) == 1


def test_parse_historical_events_accepts_wrapped_data():
    events = parse_historical_events(
        {
            "timestamp": "2026-05-16T22:55:00Z",
            "data": [
                {
                    "id": "evt-1",
                    "sport_key": "basketball_nba",
                    "commence_time": "2026-05-16T23:00:00Z",
                    "home_team": "Los Angeles Lakers",
                    "away_team": "Boston Celtics",
                }
            ],
        }
    )

    assert events[0].event_id == "evt-1"
    assert events[0].home_team == "Los Angeles Lakers"


def test_parse_sports_and_current_events_metadata():
    sports = parse_sports(
        [
            {
                "key": "basketball_nba",
                "group": "Basketball",
                "title": "NBA",
                "description": "US Basketball",
                "active": True,
                "has_outrights": False,
            }
        ]
    )
    events = parse_events_metadata(
        [
            {
                "id": "evt-1",
                "sport_key": "basketball_nba",
                "commence_time": "2026-05-16T23:00:00Z",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
            }
        ]
    )

    assert sports[0].key == "basketball_nba"
    assert sports[0].active is True
    assert events[0].event_id == "evt-1"


def test_parse_event_markets_accepts_bookmaker_market_objects():
    markets = parse_event_markets(
        {
            "bookmakers": [
                {
                    "key": "betmgm",
                    "markets": [{"key": "player_points"}, {"key": "player_rebounds"}],
                }
            ]
        }
    )

    assert markets[0].bookmaker == "betmgm"
    assert markets[0].markets == ["player_points", "player_rebounds"]


def test_fetch_historical_odds_builds_paid_endpoint_and_cost(tmp_path: Path):
    captured: dict = {}
    client = OddsApiClient(
        api_key="test-key",
        monthly_budget=100,
        budget_file=str(tmp_path / "budget.json"),
        url_opener=_fake_opener(
            captured,
            {"timestamp": "2026-05-16T22:55:00Z", "data": []},
        ),
    )

    snapshot = client.fetch_historical_odds(
        league="NBA",
        date="2026-05-16T22:55:00Z",
        markets="h2h,spreads",
        bookmakers="draftkings",
    )

    assert snapshot.timestamp == "2026-05-16T22:55:00Z"
    assert "/v4/historical/sports/basketball_nba/odds" in captured["url"]
    assert "date=2026-05-16T22%3A55%3A00Z" in captured["url"]
    assert "bookmakers=draftkings" in captured["url"]
    assert "apiKey=test-key" in captured["url"]
    assert client.remaining_budget() == 80


def test_fetch_current_event_odds_builds_event_endpoint_with_betmgm(tmp_path: Path):
    captured: dict = {}
    client = OddsApiClient(
        api_key="test-key",
        monthly_budget=100,
        budget_file=str(tmp_path / "budget.json"),
        url_opener=_fake_opener(
            captured,
            {
                "id": "evt-1",
                "sport_key": "basketball_nba",
                "commence_time": "2026-05-16T23:00:00Z",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmakers": [],
            },
        ),
    )

    event = client.fetch_current_event_odds(
        "NBA",
        "evt-1",
        markets="player_points",
        bookmakers="betmgm",
    )

    assert event.event_id == "evt-1"
    assert "/v4/sports/basketball_nba/events/evt-1/odds" in captured["url"]
    assert "markets=player_points" in captured["url"]
    assert "bookmakers=betmgm" in captured["url"]


def test_get_json_rejects_double_versioned_path_before_budget_mutation(tmp_path: Path):
    captured: dict = {}
    budget_file = tmp_path / "budget.json"
    client = OddsApiClient(
        api_key="test-key",
        monthly_budget=100,
        budget_file=str(budget_file),
        url_opener=_fake_opener(captured, []),
    )

    with pytest.raises(ValueError, match='Use "/sports/baseball_mlb/odds"'):
        client._get_json("/v4/sports/baseball_mlb/odds", {"regions": "us"})

    assert not budget_file.exists()
    assert captured == {}


def test_get_json_preserves_correct_path_url_and_api_key(tmp_path: Path):
    captured: dict = {}
    client = OddsApiClient(
        api_key="test-key",
        monthly_budget=100,
        budget_file=str(tmp_path / "budget.json"),
        url_opener=_fake_opener(captured, []),
    )

    result = client._get_json("/sports/baseball_mlb/odds", {"regions": "us"})

    assert result == []
    assert "/v4/sports/baseball_mlb/odds" in captured["url"]
    assert "regions=us" in captured["url"]
    assert "apiKey=test-key" in captured["url"]
