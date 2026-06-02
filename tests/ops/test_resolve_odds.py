from __future__ import annotations

import pytest

from omega.integrations import odds_resolver as resolver
from omega.integrations.odds_api import (
    BookOdds,
    EventMarketAvailability,
    EventOdds,
    HistoricalEvent,
    OddsApiBudgetExceeded,
)
from omega.integrations.odds_cache import OddsCache
from omega.integrations.odds_resolver import list_events, resolve_odds


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


class FakeEventListClient:
    last_quota_headers = {"x-requests-remaining": "99"}

    def __init__(self, events: list[HistoricalEvent] | None = None):
        self.events = events
        self.calls = 0

    def fetch_events(
        self,
        league,
        commence_time_from=None,
        commence_time_to=None,
        *,
        request_cost=0,
    ):
        self.calls += 1
        if self.events is not None:
            return self.events
        return [
            HistoricalEvent(
                event_id="evt-list-1",
                sport_key="basketball_nba",
                commence_time="2026-06-02T23:00:00Z",
                home_team="Indiana Pacers",
                away_team="Boston Celtics",
            )
        ]


class BudgetExceededClient:
    last_quota_headers = {}

    def budget_status(self):
        return {"current_usage": 30000, "monthly_cap": 30000}

    def fetch_events(self, *args, **kwargs):
        raise OddsApiBudgetExceeded("Monthly budget exceeded for 2026-06: used=30000 cap=30000")


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
    # Default single-book mode carries no advisory best-price block.
    assert "best_prices" not in result


def test_list_events_uses_cache_for_repeated_slate_lookup(tmp_path):
    cache = OddsCache(db_path=tmp_path / "events.db")
    client = FakeEventListClient()

    first = list_events(
        league="NBA",
        commence_time_from="2026-06-02T00:00:00Z",
        commence_time_to="2026-06-03T00:00:00Z",
        client=client,
        cache=cache,
    )
    second = list_events(
        league="NBA",
        commence_time_from="2026-06-02T00:00:00Z",
        commence_time_to="2026-06-03T00:00:00Z",
        client=client,
        cache=cache,
    )

    assert client.calls == 1
    assert first["status"] == "success"
    assert second["events"][0]["event_id"] == "evt-list-1"
    assert "source: local_cache" in second["metadata"]
    assert "cache_kind: event_list" in second["metadata"]


def test_list_events_caches_empty_slate(tmp_path):
    cache = OddsCache(db_path=tmp_path / "events.db")
    client = FakeEventListClient(events=[])

    first = list_events(league="NBA", client=client, cache=cache)
    second = list_events(league="NBA", client=client, cache=cache)

    assert client.calls == 1
    assert first["status"] == "empty"
    assert second["status"] == "empty"
    assert second["skipped_reasons"] == ["no events returned"]
    assert "source: local_cache" in second["metadata"]


def test_list_events_budget_exhausted_cli_stops(monkeypatch, capsys):
    monkeypatch.setattr(resolver, "OddsApiClient", BudgetExceededClient)
    monkeypatch.setattr(
        "sys.argv",
        ["omega-resolve-odds", "--list-events", "--league", "NBA"],
    )

    code = resolver.main()

    assert code == 1
    captured = capsys.readouterr()
    assert (
        "[error] Odds API Budget Exhausted. Current usage: 30000, Monthly Cap: 30000. "
        "To expand, adjust OMEGA_ODDS_API_MONTHLY_BUDGET in .env."
    ) in captured.err


def test_resolve_game_line_shopping_does_not_force_betmgm():
    client = FakeOddsClient()