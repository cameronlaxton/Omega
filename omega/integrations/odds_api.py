"""
omega.integrations.odds_api — the-odds-api.com thin client.

Free tier = 500 requests/month. The client tracks usage in a local JSON file
and refuses requests that would exceed a configured monthly budget.

Endpoint reference:
- Live odds:     GET /v4/sports/{sport}/odds
- Scores:        GET /v4/sports/{sport}/scores
- Historical:    GET /v4/historical/sports/{sport}/odds?date=ISO8601 (paid plan)

NOTE: The free tier does NOT include historical endpoints. "Closing line" for
free-tier users is the live snapshot taken right before tip-off, persisted
through the closing-line scheduled task. Plan accordingly when scheduling fetches.

API key is read from the OMEGA_ODDS_API_KEY environment variable. The client
is safe to instantiate without a key — fetch methods will raise
`OddsApiKeyMissing` if you try to make a request.
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
_DEFAULT_MONTHLY_BUDGET = 450  # leave headroom under the 500 free-tier ceiling
_DEFAULT_BUDGET_FILE = "omega_odds_api_budget.json"


class OddsApiKeyMissing(RuntimeError):
    """Raised when a fetch is attempted without OMEGA_ODDS_API_KEY set."""


class OddsApiBudgetExceeded(RuntimeError):
    """Raised when the configured monthly request budget would be exceeded."""


@dataclass(frozen=True)
class BookOdds:
    """One sportsbook's price for one selection."""
    bookmaker: str           # e.g. "draftkings", "fanduel"
    market: str              # e.g. "h2h", "spreads", "totals"
    selection: str           # team name (h2h/spreads) or "Over"/"Under" (totals)
    price: float             # American odds
    point: Optional[float]   # line value (None for h2h)
    last_update: str         # ISO 8601


@dataclass(frozen=True)
class EventOdds:
    """All books' odds for a single event."""
    event_id: str
    sport_key: str
    commence_time: str       # ISO 8601
    home_team: str
    away_team: str
    books: List[BookOdds]


class OddsApiClient:
    """Thin wrapper around the-odds-api with monthly budget tracking."""

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
                f"Monthly budget exceeded for {key}: used={used}, requested={n}, cap={self._monthly_budget}"
            )
        data[key] = used + n
        self._write_budget(data)

    def remaining_budget(self) -> int:
        data = self._read_budget()
        return max(0, self._monthly_budget - data.get(self._current_month_key(), 0))

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: Dict[str, Any]) -> Any:
        if not self._api_key:
            raise OddsApiKeyMissing("OMEGA_ODDS_API_KEY not set")
        self._consume_budget(1)
        query = dict(params)
        query["apiKey"] = self._api_key
        url = f"{_BASE_URL}{path}?{urllib.parse.urlencode(query)}"
        logger.debug("GET %s (params redacted)", path)
        with self._url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ------------------------------------------------------------------
    # NBA convenience
    # ------------------------------------------------------------------

    def fetch_nba_odds(
        self,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
    ) -> List[EventOdds]:
        """Fetch current NBA odds for the given markets across US books.

        One API request per call regardless of how many events return.
        """
        params: Dict[str, Any] = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        payload = self._get_json("/sports/basketball_nba/odds", params)
        return parse_events(payload)


# ---------------------------------------------------------------------------
# Parsing (public for testability)
# ---------------------------------------------------------------------------

def parse_events(payload: Any) -> List[EventOdds]:
    """Parse the-odds-api event list into our EventOdds dataclass."""
    events: List[EventOdds] = []
    if not isinstance(payload, list):
        return events
    for evt in payload:
        books: List[BookOdds] = []
        for bm in evt.get("bookmakers") or []:
            bm_key = bm.get("key", "")
            last_update = bm.get("last_update", "")
            for market in bm.get("markets") or []:
                m_key = market.get("key", "")
                for outcome in market.get("outcomes") or []:
                    books.append(BookOdds(
                        bookmaker=bm_key,
                        market=m_key,
                        selection=outcome.get("name", ""),
                        price=float(outcome.get("price", 0)),
                        point=(
                            float(outcome["point"])
                            if outcome.get("point") is not None else None
                        ),
                        last_update=last_update,
                    ))
        events.append(EventOdds(
            event_id=str(evt.get("id", "")),
            sport_key=evt.get("sport_key", ""),
            commence_time=evt.get("commence_time", ""),
            home_team=evt.get("home_team", ""),
            away_team=evt.get("away_team", ""),
            books=books,
        ))
    return events
