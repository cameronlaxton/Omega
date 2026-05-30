from __future__ import annotations

import pytest

from omega.integrations.odds_api import (
    BookOdds,
    EventMarketAvailability,
    EventOdds,
    HistoricalEvent,
)
from omega.integrations.odds_cache import OddsCache
from omega.integrations.odds_resolver import resolve_odds


@pytest.fixture(autouse=True)
def _isolate_odds_cache(tmp_path, monkeypatch):
    """Point OddsCache at a unique per-test SQLite file. Without this, resolve_odds
    shares the real ~/.omega/runtime cache across tests, so one test's success
    payload leaks into later tests' cache lookups and corrupts their assertions."""
    monkeypatch.setattr(
        OddsCache, "_resolve_db_path", lambda self: tmp_path / "odds_cache.db"
    )


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


class FakeSoccerOddsClient:
    """3-way (Home/Draw/Away) h2h client for soccer ingestion tests."""

    last_quota_headers = {"x-requests-remaining": "99"}

    def __init__(self):
        self.event_odds_bookmakers: list[str | None] = []

    def fetch_events(self, league, commence_time_from=None, commence_time_to=None):
        return [
            HistoricalEvent(
                event_id="evt-soc-1",
                sport_key="soccer_epl",
                commence_time="2026-05-17T15:00:00Z",
                home_team="Arsenal",
                away_team="Chelsea",
            )
        ]

    def fetch_event_markets(self, league, event_id, regions="us", bookmakers=None):
        return [EventMarketAvailability(bookmaker=bookmakers or "betmgm", markets=["h2h"])]

    def fetch_current_event_odds(self, league, event_id, regions="us", markets="", bookmakers=None):
        self.event_odds_bookmakers.append(bookmakers)
        books = [
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="h2h",
                selection="Arsenal",
                price=-120,
                point=None,
                last_update="2026-05-17T12:00:00Z",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="h2h",
                selection="Draw",
                price=260,
                point=None,
                last_update="2026-05-17T12:00:00Z",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="h2h",
                selection="Chelsea",
                price=320,
                point=None,
                last_update="2026-05-17T12:00:00Z",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="btts",
                selection="Yes",
                price=-110,
                point=None,
                last_update="2026-05-17T12:00:00Z",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="draw_no_bet",
                selection="Arsenal",
                price=-200,
                point=None,
                last_update="2026-05-17T12:00:00Z",
                event_id=event_id,
            ),
            BookOdds(
                bookmaker=bookmakers or "draftkings",
                market="correct_score",
                selection="2 - 1",
                price=900,
                point=None,
                last_update="2026-05-17T12:00:00Z",
                event_id=event_id,
            ),
        ]
        return EventOdds(
            event_id=event_id,
            sport_key="soccer_epl",
            commence_time="2026-05-17T15:00:00Z",
            home_team="Arsenal",
            away_team="Chelsea",
            books=books,
        )


def test_resolve_soccer_game_captures_draw_price():
    client = FakeSoccerOddsClient()
    result = resolve_odds(
        kind="game",
        league="EPL",
        home_team="Arsenal",
        away_team="Chelsea",
        all_books=False,
        client=client,
    )

    assert result["status"] == "success"
    odds = result["request_patch"]["odds"]
    assert odds["moneyline_home"] == -120
    assert odds["moneyline_away"] == 320
    assert odds["moneyline_draw"] == 260
    # Exotic markets (Gap 5)
    assert odds["btts_yes"] == -110
    assert odds["dnb_home"] == -200
    assert odds["correct_score"] == {"2-1": 900}


def test_resolve_game_defaults_to_betmgm():
    client = FakeOddsClient()
    result = resolve_odds(
        kind="game",
        league="NBA",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        all_books=False,
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
        all_books=False,
        client=FakeOddsClient(market_available=False),
    )

    assert result["status"] == "unavailable"
    assert "betmgm does not list market" in result["skipped_reasons"][0]
