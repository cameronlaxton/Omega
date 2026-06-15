"""
omega.integrations.odds_api - the-odds-api.com thin client.

This module is the only maintained The Odds API client in Omega. It is allowed
to resolve local Cowork pre-decision market inputs, closing-line captures, and
historical market snapshots. It never computes recommendations and never owns
simulation, calibration, edge, EV, Kelly, staking, grading, or trace IDs.

Default current-odds resolution is BetMGM-first. Multi-book requests are
explicit line-shopping/consensus operations, not the routine path.

Endpoint reference:
- Live odds:              GET /v4/sports/{sport}/odds
- Events:                 GET /v4/sports/{sport}/events
- Event odds:             GET /v4/sports/{sport}/events/{eventId}/odds
- Event markets:          GET /v4/sports/{sport}/events/{eventId}/markets
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
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega.integrations._guards import assert_not_replay_mode

UTC = timezone.utc

logger = logging.getLogger("omega.integrations.odds_api")

_BASE_URL = "https://api.the-odds-api.com/v4"
_REQUEST_TIMEOUT_SECONDS = 15
_DEFAULT_MONTHLY_BUDGET = int(os.environ.get("OMEGA_ODDS_API_MONTHLY_BUDGET", "20000"))
_DEFAULT_BUDGET_FILE = "omega_odds_api_budget.json"
DEFAULT_BOOKMAKER = "betmgm"
DEFAULT_MARKETS = "h2h,spreads,totals"


# Omega league code -> the-odds-api sport key. Add a row when extending coverage.
SPORT_KEY_MAP: dict[str, str] = {
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
    "WORLD_CUP": "soccer_fifa_world_cup",
    "FIFA_WORLD_CUP_2026": "soccer_fifa_world_cup",
    "UFC": "mma_mixed_martial_arts",
    "MMA": "mma_mixed_martial_arts",
}

# Tennis is intentionally absent from SPORT_KEY_MAP: the provider's tennis
# keys are per-tournament (tennis_atp_wimbledon, tennis_wta_french_open, ...)
# and churn through the season, so a static "ATP" -> "tennis_atp" entry would
# 404. Resolve active keys dynamically via resolve_tennis_sport_keys().
TENNIS_TOUR_KEY_PREFIXES: dict[str, str] = {
    "ATP": "tennis_atp",
    "WTA": "tennis_wta",
}

_TENNIS_KEYS_CACHE_DIR = Path("data/cache/odds_api")
_TENNIS_KEYS_TTL_SECONDS = 24 * 3600


def resolve_tennis_sport_keys(
    client: "OddsApiClient",
    tour: str,
    *,
    cache_dir: str | Path | None = None,
    ttl_seconds: float = _TENNIS_KEYS_TTL_SECONDS,
) -> list[str]:
    """Return the active per-tournament sport keys for an ATP/WTA tour.

    Filters the provider's sports index for active ``tennis_<tour>*`` keys,
    cached on disk for a day. On a fetch failure the last-good cached list is
    served (Part 8 sport-key-churn mitigation) — only a cold failure raises.
    """
    prefix = TENNIS_TOUR_KEY_PREFIXES.get(tour.upper())
    if prefix is None:
        raise ValueError(
            f"tour must be one of {sorted(TENNIS_TOUR_KEY_PREFIXES)}, got {tour!r}"
        )
    cache_path = Path(cache_dir or _TENNIS_KEYS_CACHE_DIR) / f"tennis_keys_{tour.lower()}.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < ttl_seconds:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        sports = client.fetch_sports(all_sports=True)
    except Exception as exc:  # noqa: BLE001 - stale fallback is the documented path
        if cache_path.exists():
            logger.warning(
                "tennis sport-key refresh failed (%s); serving stale last-good list", exc
            )
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise
    keys = sorted(
        s.key
        for s in sports
        if s.active and (s.key == prefix or s.key.startswith(prefix + "_"))
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(keys), encoding="utf-8")
    return keys


def sport_key_for(league: str) -> str | None:
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
    point: float | None
    last_update: str
    description: str | None = None
    event_id: str | None = None
    snapshot_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventOdds:
    """All books' odds for a single event."""

    event_id: str
    sport_key: str
    commence_time: str
    home_team: str
    away_team: str
    books: list[BookOdds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sport_key": self.sport_key,
            "commence_time": self.commence_time,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "books": [book.to_dict() for book in self.books],
        }


@dataclass(frozen=True)
class HistoricalEvent:
    """One event returned by the historical events endpoint."""

    event_id: str
    sport_key: str
    commence_time: str
    home_team: str
    away_team: str


@dataclass(frozen=True)
class ScoreEvent:
    """One game returned by GET /sports/{sport}/scores.

    Used to derive schedule facts (rest days) — only the completion flag, the
    commence time, and the two team names are needed; live scores are ignored.
    """

    event_id: str
    sport_key: str
    commence_time: str
    completed: bool
    home_team: str
    away_team: str
    # Final per-participant scores from the provider ``scores`` array, as
    # ((name, score), ...) pairs. Empty for upcoming games and for callers
    # that predate this field. For tennis the score value is sets won — the
    # outcome-grading path (omega-fetch-outcomes-tennis) reads it.
    scores: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SportInfo:
    """One sport returned by The Odds API sports endpoint."""

    key: str
    group: str
    title: str
    description: str
    active: bool
    has_outrights: bool


@dataclass(frozen=True)
class EventMarketAvailability:
    """Recently seen market keys for one bookmaker on one event."""

    bookmaker: str
    markets: list[str]


@dataclass(frozen=True)
class HistoricalSnapshot:
    """Metadata wrapper around a historical odds snapshot."""

    timestamp: str
    previous_timestamp: str | None
    next_timestamp: str | None
    events: list[EventOdds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "previous_timestamp": self.previous_timestamp,
            "next_timestamp": self.next_timestamp,
            "events": [event.to_dict() for event in self.events],
        }


class OddsApiClient:
    """Thin wrapper around the-odds-api with local monthly budget tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        monthly_budget: int = _DEFAULT_MONTHLY_BUDGET,
        budget_file: str | None = None,
        url_opener: Callable = urllib.request.urlopen,
    ) -> None:
        self._api_key = api_key or os.environ.get("OMEGA_ODDS_API_KEY")
        self._monthly_budget = monthly_budget
        self._budget_file = Path(budget_file) if budget_file else Path.cwd() / _DEFAULT_BUDGET_FILE
        self._url_opener = url_opener
        self.last_quota_headers: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Budget bookkeeping
    # ------------------------------------------------------------------

    def _current_month_key(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m")

    def _read_budget(self) -> dict[str, int]:
        if not self._budget_file.exists():
            return {}
        try:
            return json.loads(self._budget_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Budget file unreadable, resetting: %s", self._budget_file)
            return {}

    def _write_budget(self, data: dict[str, int]) -> None:
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

    def current_usage(self) -> int:
        data = self._read_budget()
        return data.get(self._current_month_key(), 0)

    def monthly_budget(self) -> int:
        return self._monthly_budget

    def budget_status(self) -> dict[str, int]:
        return {
            "current_usage": self.current_usage(),
            "monthly_cap": self.monthly_budget(),
        }

    def remaining_budget(self) -> int:
        return max(0, self._monthly_budget - self.current_usage())

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: dict[str, Any], request_cost: int = 1) -> Any:
        assert_not_replay_mode("Odds API fetch")
        if not self._api_key:
            raise OddsApiKeyMissing("OMEGA_ODDS_API_KEY not set")
        if path == "/v4" or path.startswith("/v4/"):
            corrected_path = path.removeprefix("/v4") or "/"
            raise ValueError(
                "Odds API path must not include /v4 prefix. "
                f'Use "{corrected_path}", not "{path}".'
            )
        self._consume_budget(request_cost)
        query = dict(params)
        query["apiKey"] = self._api_key
        url = f"{_BASE_URL}{path}?{urllib.parse.urlencode(query)}"
        logger.debug("GET %s (params redacted)", path)
        with self._url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            self.last_quota_headers = _quota_headers(resp)
            return json.loads(resp.read().decode("utf-8"))

    def fetch_sports(self, all_sports: bool = True) -> list[SportInfo]:
        """Fetch supported sports metadata.

        This is useful for coverage audits and does not perform any Omega
        betting math.
        """
        params: dict[str, Any] = {"all": str(bool(all_sports)).lower()}
        payload = self._get_json("/sports", params)
        return parse_sports(payload)

    def fetch_events(
        self,
        league: str,
        commence_time_from: str | None = None,
        commence_time_to: str | None = None,
        *,
        request_cost: int = 0,
        sport_key: str | None = None,
    ) -> list[HistoricalEvent]:
        """Fetch current/live event metadata for resolving event IDs."""
        if sport_key is None:
            sport_key = _require_sport_key(league)
        params: dict[str, Any] = {"dateFormat": "iso"}
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        payload = self._get_json(f"/sports/{sport_key}/events", params, request_cost=request_cost)
        return parse_events_metadata(payload)

    def fetch_scores(
        self, league: str, days_from: int = 3, *, sport_key: str | None = None
    ) -> list[ScoreEvent]:
        """Fetch recent (completed) and upcoming games for a league.

        ``days_from`` is the number of days in the past to include completed
        games for; the-odds-api caps it at 3. Used to derive schedule facts
        (e.g. rest days) for leagues without a free ESPN scoreboard.

        ``sport_key`` overrides the static SPORT_KEY_MAP lookup — required for
        tennis, whose provider keys are per-tournament (see
        :func:`resolve_tennis_sport_keys`).
        """
        if sport_key is None:
            sport_key = _require_sport_key(league)
        params: dict[str, Any] = {"dateFormat": "iso", "daysFrom": int(days_from)}
        payload = self._get_json(f"/sports/{sport_key}/scores", params)
        return parse_scores(payload)

    def fetch_event_markets(
        self,
        league: str,
        event_id: str,
        regions: str = "us",
        bookmakers: str | None = None,
    ) -> list[EventMarketAvailability]:
        """Fetch recently seen market keys per bookmaker for one event."""
        sport_key = _require_sport_key(league)
        params = _event_params(regions=regions, bookmakers=bookmakers)
        payload = self._get_json(f"/sports/{sport_key}/events/{event_id}/markets", params)
        return parse_event_markets(payload)

    # ------------------------------------------------------------------
    # Current odds
    # ------------------------------------------------------------------

    def fetch_event_odds(
        self,
        league: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = None,
    ) -> list[EventOdds]:
        """Fetch current event odds for a league across books."""
        sport_key = _require_sport_key(league)
        params = _odds_params(regions=regions, markets=markets, bookmakers=bookmakers)
        payload = self._get_json(f"/sports/{sport_key}/odds", params)
        return parse_events(payload)

    def fetch_current_event_odds(
        self,
        league: str,
        event_id: str,
        regions: str = "us",
        markets: str = DEFAULT_MARKETS,
        bookmakers: str | None = None,
    ) -> EventOdds:
        """Fetch current odds for one event, including props/additional markets."""
        sport_key = _require_sport_key(league)
        params = _odds_params(regions=regions, markets=markets, bookmakers=bookmakers)
        payload = self._get_json(f"/sports/{sport_key}/events/{event_id}/odds", params)
        events = parse_events([payload] if isinstance(payload, dict) else payload)
        if not events:
            return EventOdds("", sport_key, "", "", "", [])
        return events[0]

    def fetch_nba_odds(
        self,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = None,
    ) -> list[EventOdds]:
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
        bookmakers: str | None = None,
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
        commence_time_from: str | None = None,
        commence_time_to: str | None = None,
        event_ids: str | None = None,
    ) -> list[HistoricalEvent]:
        """Fetch historical event metadata for resolving event IDs."""
        sport_key = _require_sport_key(league)
        params: dict[str, Any] = {"date": date, "dateFormat": "iso"}
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
        bookmakers: str | None = None,
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
    bookmakers: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    else:
        params["regions"] = regions
    return params


def _event_params(regions: str, bookmakers: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"dateFormat": "iso"}
    if bookmakers:
        params["bookmakers"] = bookmakers
    else:
        params["regions"] = regions
    return params


def _historical_cost(regions: str, markets: str) -> int:
    region_count = max(1, len([r for r in regions.split(",") if r.strip()]))
    market_count = max(1, len([m for m in markets.split(",") if m.strip()]))
    return 10 * region_count * market_count


def _quota_headers(resp: Any) -> dict[str, str]:
    headers = getattr(resp, "headers", None)
    if not headers:
        return {}
    out: dict[str, str] = {}
    for key in ("x-requests-remaining", "x-requests-used", "x-requests-last"):
        try:
            value = headers.get(key)
        except AttributeError:
            value = None
        if value is not None:
            out[key] = str(value)
    return out


# ---------------------------------------------------------------------------
# Parsing (public for testability)
# ---------------------------------------------------------------------------


def parse_events(payload: Any) -> list[EventOdds]:
    """Parse the-odds-api event list into EventOdds dataclasses."""
    events: list[EventOdds] = []
    if not isinstance(payload, list):
        return events
    for evt in payload:
        books: list[BookOdds] = []
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
                            event_id=str(evt.get("id", "")),
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


def parse_sports(payload: Any) -> list[SportInfo]:
    """Parse sports metadata from GET /sports."""
    sports: list[SportInfo] = []
    if not isinstance(payload, list):
        return sports
    for item in payload:
        sports.append(
            SportInfo(
                key=str(item.get("key", "")),
                group=str(item.get("group", "")),
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                active=bool(item.get("active", False)),
                has_outrights=bool(item.get("has_outrights", False)),
            )
        )
    return sports


def parse_events_metadata(payload: Any) -> list[HistoricalEvent]:
    """Parse current event metadata from GET /sports/{sport}/events."""
    events: list[HistoricalEvent] = []
    if not isinstance(payload, list):
        return events
    for evt in payload:
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


def parse_scores(payload: Any) -> list[ScoreEvent]:
    """Parse recent/upcoming games from GET /sports/{sport}/scores."""
    scores: list[ScoreEvent] = []
    if not isinstance(payload, list):
        return scores
    for evt in payload:
        if not isinstance(evt, dict):
            continue
        raw_scores = evt.get("scores") or []
        score_pairs = tuple(
            (str(s.get("name", "")), str(s.get("score", "")))
            for s in raw_scores
            if isinstance(s, dict)
        )
        scores.append(
            ScoreEvent(
                event_id=str(evt.get("id", "")),
                sport_key=str(evt.get("sport_key", "")),
                commence_time=str(evt.get("commence_time", "")),
                completed=bool(evt.get("completed", False)),
                home_team=str(evt.get("home_team", "")),
                away_team=str(evt.get("away_team", "")),
                scores=score_pairs,
            )
        )
    return scores


def parse_event_markets(payload: Any) -> list[EventMarketAvailability]:
    """Parse event-market availability from GET /events/{eventId}/markets.

    The provider shape is intentionally parsed tolerantly because market
    availability may be returned either as bookmaker objects containing a
    markets list or as a direct mapping in fixtures.
    """
    data = payload
    if isinstance(payload, dict):
        data = payload.get("bookmakers") or payload.get("data") or payload.get("markets") or []
    if isinstance(data, dict):
        data = [{"key": key, "markets": value} for key, value in data.items()]
    out: list[EventMarketAvailability] = []
    if not isinstance(data, list):
        return out
    for row in data:
        if isinstance(row, str):
            out.append(EventMarketAvailability(bookmaker="", markets=[row]))
            continue
        markets_raw = row.get("markets") or []
        markets: list[str] = []
        for item in markets_raw:
            if isinstance(item, str):
                markets.append(item)
            elif isinstance(item, dict):
                key = item.get("key")
                if key:
                    markets.append(str(key))
        out.append(
            EventMarketAvailability(
                bookmaker=str(row.get("key") or row.get("bookmaker") or ""),
                markets=markets,
            )
        )
    return out


def parse_historical_snapshot(payload: Any) -> HistoricalSnapshot:
    """Parse a wrapped historical odds or historical event-odds response."""
    if not isinstance(payload, dict):
        return HistoricalSnapshot("", None, None, [])
    data = payload.get("data")
    event_payload = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    snapshot = HistoricalSnapshot(
        timestamp=str(payload.get("timestamp", "")),
        previous_timestamp=payload.get("previous_timestamp"),
        next_timestamp=payload.get("next_timestamp"),
        events=parse_events(event_payload),
    )
    events: list[EventOdds] = []
    for event in snapshot.events:
        books = [
            BookOdds(
                bookmaker=book.bookmaker,
                market=book.market,
                selection=book.selection,
                price=book.price,
                point=book.point,
                last_update=book.last_update,
                description=book.description,
                event_id=book.event_id,
                snapshot_timestamp=snapshot.timestamp,
            )
            for book in event.books
        ]
        events.append(
            EventOdds(
                event_id=event.event_id,
                sport_key=event.sport_key,
                commence_time=event.commence_time,
                home_team=event.home_team,
                away_team=event.away_team,
                books=books,
            )
        )
    return HistoricalSnapshot(
        timestamp=snapshot.timestamp,
        previous_timestamp=snapshot.previous_timestamp,
        next_timestamp=snapshot.next_timestamp,
        events=events,
    )


def parse_historical_events(payload: Any) -> list[HistoricalEvent]:
    """Parse a wrapped historical events response."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    events: list[HistoricalEvent] = []
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
