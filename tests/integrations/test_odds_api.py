from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from omega.integrations.odds_api import (
    OddsApiClient,
    parse_events,
    parse_historical_events,
    parse_historical_snapshot,
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
