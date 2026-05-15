"""
omega.integrations.odds_api — the-odds-api.com thin client.

DEPRECATED (Phase 6f) — pending removal after JIT ingest reaches steady state.
=============================================================================
Closing-line capture has moved to agent-driven JIT WebFetch + the file-based
ingest path at `scripts/ingest_closing_lines.py`. This client and its sole
consumer (`scripts/fetch_closing_lines.py`) are kept as a fallback for one
release while the JIT protocol bakes. Do not add new call sites.

POST-DECISION USE ONLY.
======================
This module is restricted to **closing-line capture for CLV computation**.
Pre-decision sourcing (the lines / odds the LLM injects into analyze()) is
done by the agent itself via WebFetch on direct sportsbook pages per
prompts/system_prompt.txt §6.1.5. Do not add pre-decision call sites here;
the player-props add-on the-odds-api charges for is **not in budget** and
not required for sandbox operation.

The single supported entry point is :meth:`OddsApiClient.fetch_nba_odds`,
which returns one snapshot per call (h2h / spreads / totals across US books).
Schedule the snapshot to run shortly before tip-off — that snapshot IS the
"closing line" for CLV purposes on the free tier.

Free tier = 500 requests/month. The client tracks usage in a local JSON file
and refuses requests that would exceed a configured monthly budget.

Endpoint reference (kept for documentation only — only /odds is used today):
- Live odds:     GET /v4/sports/{sport}/odds        (USED — closing snapshot)
- Scores:        GET /v4/sports/{sport}/scores      (not used)
- Historical:    GET /v4/historical/sports/...      (paid plan, not used)

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


# Omega league code -> the-odds-api sport key. Add a row when extending coverage.
# Sport keys are documented at https://the-odds-api.com/sports-odds-data/sports-apis.html.
SPORT_KEY_MAP: Dict[str, str] = {
    "NBA":   "basketball_nba",
    "WNBA":  "basketball_wnba",
    "NCAAB": "basketball_ncaab",
    "NCAAM": "basketball_ncaab",
    "NFL":   "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "MLB":   "baseball_mlb",
    "NHL":   "icehockey_nhl",
    "EPL":   "soccer_epl",
    "MLS":   "soccer_usa_mls",
    "LA_LIGA":          "soccer_spain_la_liga",
    "BUNDESLIGA":       "soccer_germany_bundesliga",
    "SERIE_A":          "soccer_italy_serie_a",
    "LIGUE_1":          "soccer_france_ligue_one",
    "CHAMPIONS_LEAGUE": "soccer_uefa_champs_league",
    "UFC":   "mma_mixed_martial_arts",
    "MMA":   "mma_mixed_martial_arts",
}


def sport_key_for(league: str) -> Optional[str]:
    """Resolve an Omega league code to a the-odds-api sport key, or None."""
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
    # Event-odds fetch (sport-agnostic)
    # ------------------------------------------------------------------

    def fetch_event_odds(
        self,
        league: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
    ) -> List[EventOdds]:
        """Fetch current event odds for a league across US books.

        One API request per call regardless of how many events return.

        POST-DECISION USE ONLY (closing-line capture). Do not call this from
        any pre-decision path; the agent sources lines via WebFetch on direct
        sportsbook pages per system_prompt.txt §6.1.5.

        ``markets`` is restricted to game markets (h2h / spreads / totals).
        Player-prop markets are not enabled on the free tier and are not in
        budget; passing a player-prop market key here will return an empty
        list from the API rather than charging extra.

        Raises ``ValueError`` if the league isn't in :data:`SPORT_KEY_MAP`.
        """
        sport_key = sport_key_for(league)
        if sport_key is None:
            raise ValueError(
                f"No the-odds-api sport key mapped for league={league!r}. "
                "Add it to SPORT_KEY_MAP in omega.integrations.odds_api."
            )
        params: Dict[str, Any] = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        payload = self._get_json(f"/sports/{sport_key}/odds", params)
        return parse_events(payload)

    def fetch_nba_odds(
        self,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
    ) -> List[EventOdds]:
        """Back-compat shim. Delegates to :meth:`fetch_event_odds` with league='NBA'."""
        return self.fetch_event_odds(
            league="NBA", regions=regions, markets=markets, bookmakers=bookmakers,
        )


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
