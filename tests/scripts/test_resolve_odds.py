from __future__ import annotations

from omega.integrations.odds_api import (
    BookOdds,
    EventMarketAvailability,
    EventOdds,
    HistoricalEvent,
)
from omega.integrations.odds_resolver import resolve_odds


class FakeOddsClient:
    last_quota_headers = {"x-requests-remaining": "99"}

    def __init__(self, market_available: bool = True):
        self.market_available = market_available
        self.event_odds_bookmakers: list[str | None] = []

    def fetch_events(self, league, commence_time_from=None, commence_time_to=None):
        return [
            HistoricalEvent(
                event_id="evt-1",
                sport_key="basketball_nba",
                commence_time="2026-05-17T23:00:00Z",
                home_team="Los Angeles Lakers",
                away_team="Boston Celtics",
            )
        ]

    def fetch_event_markets(self, league, event_id, regions="us", bookmakers=None):
        markets = ["player_points"] if self.market_available else ["player_rebounds"]
        return [EventMarketAvailability(bookmaker=bookmakers or "betmgm", markets=markets)]

    def fetch_current_event_odds(self, league, event_id, regions="us", markets="", bookmakers=None):
        self.event_odds_bookmakers.append(bookmakers)
        books = [
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="h2h",
                selection="Los Angeles Lakers",
                price=-150,
                point=None,
                last_update="2026-05-17T20:00:00Z",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="h2h",
                selection="Boston Celtics",
                price=130,
                point=None,
                last_update="2026-05-17T20:00:00Z",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="spreads",
                selection="Los Angeles Lakers",
                price=-110,
                point=-3.5,
                last_update="2026-05-17T20:00:00Z",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="player_points",
                selection="Over",
                price=-115,
                point=27.5,
                last_update="2026-05-17T20:00:00Z",
                description="Jayson Tatum",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="player_points",
                selection="Under",
                price=-105,
                point=27.5,
                last_update="2026-05-17T20:00:00Z",
                description="Jayson Tatum",
                event_id=event_id,
            ),
        ]
        return EventOdds(
            event_id=event_id,
            sport_key="basketball_nba",
            commence_time="2026-05-17T23:00:00Z",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            books=books,
        )


def test_resolve_game_defaults_to_betmgm():
    client = FakeOddsClient()
    result = resolve_odds(
        kind="game",
        league="NBA",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        client=client,
    )

    assert result["status"] == "success"
    assert client.event_odds_bookmakers == ["betmgm"]
    assert result["request_patch"]["odds"]["moneyline_home"] == -150
    assert result["request_patch"]["odds"]["spread_home"] == -3.5


def test_resolve_game_line_shopping_does_not_force_betmgm():
    client = FakeOddsClient()
    result = resolve_odds(
        kind="game",
        league="NBA",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        line_shopping=True,
        client=client,
    )

    assert result["status"] == "success"
    assert client.event_odds_bookmakers == [None]
    assert result["line_shopping"] is True


def test_resolve_prop_returns_exact_betmgm_patch():
    result = resolve_odds(
        kind="prop",
        league="NBA",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        player_name="Jayson Tatum",
        prop_type="pts",
        line=27.5,
        client=FakeOddsClient(),
    )

    assert result["status"] == "success"
    assert result["request_patch"] == {"line": 27.5, "odds_over": -115, "odds_under": -105}


def test_resolve_wnba_points_prop_uses_standard_ou_market():
    result = resolve_odds(
        kind="prop",
        league="WNBA",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        player_name="Jayson Tatum",
        prop_type="pts",
        line=27.5,
        client=FakeOddsClient(),
    )

    assert result["status"] == "success"
    assert result["request_patch"] == {"line": 27.5, "odds_over": -115, "odds_under": -105}
    assert {q["provider_market_key"] for q in result["quotes"]} == {"player_points"}


def test_resolve_prop_does_not_fallback_when_betmgm_market_missing():
    result = resolve_odds(
        kind="prop",
        league="NBA",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        player_name="Jayson Tatum",
        prop_type="pts",
        client=FakeOddsClient(market_available=False),
    )

    assert result["status"] == "unavailable"
    assert "betmgm does not list market" in result["skipped_reasons"][0]
