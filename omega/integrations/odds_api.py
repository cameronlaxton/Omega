"""
omega.integrations.odds_api - the-odds-api.com thin client.

POST-DECISION USE ONLY.
======================
This module is restricted to closing-line capture for CLV computation and
historical odds snapshots for replay/backtest artifacts. Pre-decision sourcing
(the lines / odds the LLM injects into analyze()) stays in the agent's cited
evidence flow; the API key should never be exposed in prompts, frontend code, or
trace blobs.

Endpoint reference:
- Live odds:              GET /v4/sports/{sport}/odds
- Historical odds:        GET /v4/historical/sports/{sport}/odds
- Historical events:      GET /v4/historical/sports/{sport}/events
- Historical event odds:  GET /v4/historical/sports/{sport}/events/{eventId}/odds

API key is read from OMEGA_ODDS_API_KEY. Fetch methods raise
OddsApiKeyMissing if the key is absent.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("omega.integrations.odds_api")

_BASE_URL = "https://api.the-odds-api.com/v4"
_REQUEST_TIMEOUT_SECONDS = 15
_DEFAULT_MONTHLY_BUDGET = int(os.environ.get("OMEGA_ODDS_API_MONTHLY_BUDGET", "450"))
_DEFAULT_BUDGET_FILE = "omega_odds_api_budget.json"


# Omega league code -> the-odds-api sport key. Add a row when extending coverage.
SPORT_KEY_MAP: Dict[str, str] = {
    "NBA": "basketball_nba",
    "WNBA": "basketball_wnba",
    "NCAAB": "basketball_ncaab",
    "NCAAM": "basketball_ncaab",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
    "EPL": "soccer_epl",
    "MLS": "soccer_usa_mls",
    "LA_LIGA": "soccer_spain_la_liga",
    "BUNDESLIGA": "soccer_germany_bundesliga",
    "SERIE_A": "soccer_italy_serie_a",
    "LIGUE_1": "soccer_france_ligue_one",
    "CHAMPIONS_LEAGUE": "soccer_uefa_champs_league",
    "UFC": "mma_mixed_martial_arts",
    "MMA": "mma_mixed_martial_arts",
}


def sport_key_for(league: str) -> Optional[str]:
    """Resolve an Omega league code to the-odds-api sport key, or None."""
    if not league:
        return None
    return SPORT_KEY_MAP.get(league.upper())


class OddsApiKeyMissing(RuntimeError):
    """Raised when a fetch is attempted without OMEGA_ODDS_API_KEY set."""


class OddsApiBudgetExceeded(RuntimeError):
    """Raised when the configured monthly request budget would be exceeded."""


@dataclass(frozen=True)
class BookOdds:
    """One sportsbook's price for one selection."""

    bookmaker: str
    market: str
    selection: str
    price: float
    point: Optional[float]
    last_update: str
    description: Optional[str] = None


@dataclass(frozen=True)
class EventOdds:
    """All books' odds for a single event."""

    event_id: str
    sport_key: str
    commence_time: str
    home_team: str
    away_team: str
    books: List[BookOdds]


@dataclass(frozen=True)
class HistoricalEvent:
    """One event returned by the historical events endpoint."""

    event_id: str
    sport_key: str
    commence_time: str
    home_team: str
    away_team: str


@dataclass(frozen=True)
class HistoricalSnapshot:
    """Metadata wrapper around a historical odds snapshot."""

    timestamp: str
    previous_timestamp: Optional[str]
    next_timestamp: Optional[str]
    events: List[EventOdds]


class OddsApiClient:
    """Thin wrapper around the-odds-api with local monthly budget tracking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        monthly_budget: int = _DEFAULT_MONTHLY_BUDGET,
        budget_file: Optional[str] = None,
        url_opener: Callable = urllib.request.urlopen,
    ) -> None:
        self._api_key = api_key or os.environ.get("OMEGA_ODDS_API_KEY")
        self._monthly_budget = monthly_budget
        self._budget_file = Path(budget_file) if budget_file else Path.cwd() / _DEFAULT_BUDGET_FILE
        self._url_opener = url_opener

    # ------------------------------------------------------------------
    # Budget bookkeeping
    # ------------------------------------------------------------------

    def _current_month_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _read_budget(self) -> Dict[str, int]:
        if not self._budget_file.exists():
            return {}
        try:
            return json.loads(self._budget_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Budget file unreadable, resetting: %s", self._budget_file)
            return {}

    def _write_budget(self, data: Dict[str, int]) -> None:
        self._budget_file.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    def _consume_budget(self, n: int = 1) -> None:
        data = self._read_budget()
        key = self._current_month_key()
        used = data.get(key, 0)
        if used + n > self._monthly_budget:
            raise OddsApiBudgetExceeded(
                f"Monthly budget exceeded for {key}: used={used}, requested={n}, "
                f"cap={self._monthly_budget}"
            )
        data[key] = used + n
        self._write_budget(data)

    def remaining_budget(self) -> int:
        data = self._read_budget()
        return max(0, self._monthly_budget - data.get(self._current_month_key(), 0))

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: Dict[str, Any], request_cost: int = 1) -> Any:
        if not self._api_key:
            raise OddsApiKeyMissing("OMEGA_ODDS_API_KEY not set")
        self._consume_budget(request_cost)
        query = dict(params)
        query["apiKey"] = self._api_key
        url = f"{_BASE_URL}{path}?{urllib.parse.urlencode(query)}"
        logger.debug("GET %s (params redacted)", path)
        with self._url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ------------------------------------------------------------------
    # Current odds
    # ------------------------------------------------------------------

    def fetch_event_odds(
        self,
        league: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
    ) -> List[EventOdds]:
        """Fetch current event odds for a league across books."""
        sport_key = _require_sport_key(league)
        params = _odds_params(regions=regions, markets=markets, bookmakers=bookmakers)
        payload = self._get_json(f"/sports/{sport_key}/odds", params)
        return parse_events(payload)

    def fetch_nba_odds(
        self,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
    ) -> List[EventOdds]:
        """Back-compat shim. Delegates to fetch_event_odds(league='NBA')."""
        return self.fetch_event_odds(
            league="NBA", regions=regions, markets=markets, bookmakers=bookmakers
        )

    # ------------------------------------------------------------------
    # Historical odds (paid API plans)
    # ------------------------------------------------------------------

    def fetch_historical_odds(
        self,
        league: str,
        date: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
    ) -> HistoricalSnapshot:
        """Fetch a sport-level historical featured-market snapshot.

        The Odds API returns the closest snapshot equal to or earlier than
        ``date``. This endpoint is the preferred CLV/backtest source for
        featured markets because it covers every event in one call.
        """
        sport_key = _require_sport_key(league)
        params = _odds_params(regions=regions, markets=markets, bookmakers=bookmakers)
        params["date"] = date
        payload = self._get_json(
            f"/historical/sports/{sport_key}/odds",
            params,
            request_cost=_historical_cost(regions=regions, markets=markets),
        )
        return parse_historical_snapshot(payload)

    def fetch_historical_events(
        self,
        league: str,
        date: str,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
        event_ids: Optional[str] = None,
    ) -> List[HistoricalEvent]:
        """Fetch historical event metadata for resolving event IDs."""
        sport_key = _require_sport_key(league)
        params: Dict[str, Any] = {"date": date, "dateFormat": "iso"}
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        if event_ids:
            params["eventIds"] = event_ids
        payload = self._get_json(f"/historical/sports/{sport_key}/events", params)
        return parse_historical_events(payload)

    def fetch_historical_event_odds(
        self,
        league: str,
        event_id: str,
        date: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
    ) -> HistoricalSnapshot:
        """Fetch historical odds for one event, including prop/additional markets."""
        sport_key = _require_sport_key(league)
        params = _odds_params(regions=regions, markets=markets, bookmakers=bookmakers)
        params["date"] = date
        payload = self._get_json(
            f"/historical/sports/{sport_key}/events/{event_id}/odds",
            params,
            request_cost=_historical_cost(regions=regions, markets=markets),
        )
        return parse_historical_snapshot(payload)


def _require_sport_key(league: str) -> str:
    sport_key = sport_key_for(league)
    if sport_key is None:
        raise ValueError(
            f"No the-odds-api sport key mapped for league={league!r}. "
            "Add it to SPORT_KEY_MAP in omega.integrations.odds_api."
        )
    return sport_key


def _odds_params(
    regions: str,
    markets: str,
    bookmakers: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    else:
        params["regions"] = regions
    return params


def _historical_cost(regions: str, markets: str) -> int:
    region_count = max(1, len([r for r in regions.split(",") if r.strip()]))
    market_count = max(1, len([m for m in markets.split(",") if m.strip()]))
    return 10 * region_count * market_count


# ---------------------------------------------------------------------------
# Parsing (public for testability)
# ---------------------------------------------------------------------------


def parse_events(payload: Any) -> List[EventOdds]:
    """Parse the-odds-api event list into EventOdds dataclasses."""
    events: List[EventOdds] = []
    if not isinstance(payload, list):
        return events
    for evt in payload:
        books: List[BookOdds] = []
        for bm in evt.get("bookmakers") or []:
            bm_key = bm.get("key", "")
            bm_last_update = bm.get("last_update", "")
            for market in bm.get("markets") or []:
                m_key = market.get("key", "")
                last_update = market.get("last_update") or bm_last_update
                for outcome in market.get("outcomes") or []:
                    books.append(
                        BookOdds(
                            bookmaker=bm_key,
                            market=m_key,
                            selection=outcome.get("name", ""),
                            price=float(outcome.get("price", 0)),
                            point=(
                                float(outcome["point"])
                                if outcome.get("point") is not None
                                else None
                            ),
                            last_update=last_update,
                            description=outcome.get("description"),
                        )
                    )
        events.append(
            EventOdds(
                event_id=str(evt.get("id", "")),
                sport_key=evt.get("sport_key", ""),
                commence_time=evt.get("commence_time", ""),
                home_team=evt.get("home_team", ""),
                away_team=evt.get("away_team", ""),
                books=books,
            )
        )
    return events


def parse_historical_snapshot(payload: Any) -> HistoricalSnapshot:
    """Parse a wrapped historical odds or historical event-odds response."""
    if not isinstance(payload, dict):
        return HistoricalSnapshot("", None, None, [])
    data = payload.get("data")
    event_payload = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    return HistoricalSnapshot(
        timestamp=str(payload.get("timestamp", "")),
        previous_timestamp=payload.get("previous_timestamp"),
        next_timestamp=payload.get("next_timestamp"),
        events=parse_events(event_payload),
    )


def parse_historical_events(payload: Any) -> List[HistoricalEvent]:
    """Parse a wrapped historical events response."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    events: List[HistoricalEvent] = []
    for evt in data:
        events.append(
            HistoricalEvent(
                event_id=str(evt.get("id", "")),
                sport_key=evt.get("sport_key", ""),
                commence_time=evt.get("commence_time", ""),
                home_team=evt.get("home_team", ""),
                away_team=evt.get("away_team", ""),
            )
        )
    return events
