"""
The Odds API client -- fetch live odds for upcoming games.

Uses The Odds API v4 (free tier: 500 requests/month).
Returns structured game + bookmaker data in American odds format.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("omega.data.acquisition.odds_api")

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

LEAGUE_SPORT_MAPPING: Dict[str, str] = {
    "NBA": "basketball_nba",
    "NFL": "americanfootball_nfl",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
    "NCAAB": "basketball_ncaab",
    "NCAAF": "americanfootball_ncaaf",
    "WNBA": "basketball_wnba",
    "MLS": "soccer_usa_mls",
    "EPL": "soccer_epl",
    "UFC": "mma_mixed_martial_arts",
}

REQUEST_TIMEOUT = 10.0
RATE_LIMIT_DELAY = 1.0
_last_request_time: float = 0.0


def _get_api_key() -> Optional[str]:
    return os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")


def _rate_limit() -> None:
    """Enforce rate limiting between requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()


def _get_sport_key(league: str) -> Optional[str]:
    """Convert league name to Odds API sport key."""
    return LEAGUE_SPORT_MAPPING.get(league.upper())


def _make_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Make a request to The Odds API."""
    api_key = _get_api_key()
    if not api_key:
        logger.warning("ODDS_API_KEY not set, skipping odds API")
        return None

    _rate_limit()

    url = f"{ODDS_API_BASE_URL}/{endpoint}"
    request_params: Dict[str, Any] = {"apiKey": api_key}
    if params:
        request_params.update(params)

    try:
        response = httpx.get(url, params=request_params, timeout=REQUEST_TIMEOUT)

        remaining = response.headers.get("x-requests-remaining", "?")
        logger.debug("Odds API requests remaining: %s", remaining)

        if response.status_code == 401:
            logger.error("Invalid ODDS_API_KEY")
            return None
        if response.status_code == 429:
            logger.warning("Odds API rate limit exceeded")
            return None
        if response.status_code != 200:
            logger.error("Odds API error: %d", response.status_code)
            return None

        return response.json()

    except httpx.TimeoutException:
        logger.error("Odds API request timed out")
        return None
    except httpx.HTTPError as exc:
        logger.error("Odds API request failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_upcoming_odds(league: str) -> List[Dict[str, Any]]:
    """Get upcoming games with odds for a league.

    Args:
        league: League code (NBA, NFL, MLB, NHL, NCAAB, NCAAF, etc.)

    Returns:
        List of games with bookmaker odds (moneyline, spread, totals).
    """
    sport_key = _get_sport_key(league)
    if not sport_key:
        logger.warning("Unsupported league for odds API: %s", league)
        return []

    data = _make_request(
        f"sports/{sport_key}/odds",
        params={
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
    )

    if data is None:
        return []

    games: List[Dict[str, Any]] = []
    for event in data:
        game: Dict[str, Any] = {
            "game_id": event.get("id", ""),
            "sport": event.get("sport_key", ""),
            "league": league.upper(),
            "commence_time": event.get("commence_time", ""),
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "bookmakers": [],
        }

        for bookmaker in event.get("bookmakers", []):
            book_data: Dict[str, Any] = {
                "name": bookmaker.get("title", ""),
                "key": bookmaker.get("key", ""),
                "last_update": bookmaker.get("last_update", ""),
                "markets": {},
            }

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                outcomes = []
                for outcome in market.get("outcomes", []):
                    outcomes.append({
                        "name": outcome.get("name", ""),
                        "price": outcome.get("price", 0),
                        "point": outcome.get("point"),
                    })
                book_data["markets"][market_key] = outcomes

            game["bookmakers"].append(book_data)

        games.append(game)

    logger.info("Odds API returned %d games for %s", len(games), league)
    return games


def get_game_odds(game_id: str, league: str) -> Optional[Dict[str, Any]]:
    """Get current odds for a specific game by ID.

    Args:
        game_id: The Odds API event ID.
        league: League code.

    Returns:
        Game odds data or None if not found.
    """
    games = get_upcoming_odds(league)
    for game in games:
        if game.get("game_id") == game_id:
            return game
    return None


def get_player_props(game_id: str, league: str) -> List[Dict[str, Any]]:
    """Get player props for a specific game.

    Note: Player props require higher API tiers. Returns empty list on free tier.
    """
    sport_key = _get_sport_key(league)
    if not sport_key:
        return []

    prop_markets = [
        "player_points", "player_rebounds", "player_assists",
        "player_pass_yds", "player_rush_yds", "player_reception_yds",
    ]

    props: List[Dict[str, Any]] = []

    for market in prop_markets:
        data = _make_request(
            f"sports/{sport_key}/events/{game_id}/odds",
            params={
                "regions": "us",
                "markets": market,
                "oddsFormat": "american",
            },
        )

        if data and isinstance(data, dict):
            for bookmaker in data.get("bookmakers", []):
                for mkt in bookmaker.get("markets", []):
                    props.append({
                        "bookmaker": bookmaker.get("title", ""),
                        "market": mkt.get("key", ""),
                        "outcomes": mkt.get("outcomes", []),
                    })

    return props


def extract_consensus_odds(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract consensus (median) odds from multi-bookmaker game data.

    For each game, computes the median moneyline, spread, and total
    across all bookmakers.

    Args:
        games: Output from get_upcoming_odds().

    Returns:
        List of dicts with consensus odds per game.
    """
    import statistics

    consensus: List[Dict[str, Any]] = []

    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        ml_home: List[float] = []
        ml_away: List[float] = []
        spreads_home: List[float] = []
        totals: List[float] = []

        for book in game.get("bookmakers", []):
            markets = book.get("markets", {})

            # Moneylines
            for outcome in markets.get("h2h", []):
                price = outcome.get("price")
                if price is not None:
                    if outcome.get("name") == home:
                        ml_home.append(float(price))
                    elif outcome.get("name") == away:
                        ml_away.append(float(price))

            # Spreads
            for outcome in markets.get("spreads", []):
                if outcome.get("name") == home:
                    point = outcome.get("point")
                    if point is not None:
                        spreads_home.append(float(point))

            # Totals
            for outcome in markets.get("totals", []):
                if outcome.get("name") == "Over":
                    point = outcome.get("point")
                    if point is not None:
                        totals.append(float(point))

        entry: Dict[str, Any] = {
            "game_id": game.get("game_id"),
            "league": game.get("league"),
            "home_team": home,
            "away_team": away,
            "commence_time": game.get("commence_time"),
            "moneyline_home": statistics.median(ml_home) if ml_home else None,
            "moneyline_away": statistics.median(ml_away) if ml_away else None,
            "spread_home": statistics.median(spreads_home) if spreads_home else None,
            "total": statistics.median(totals) if totals else None,
            "num_books": len(game.get("bookmakers", [])),
        }
        consensus.append(entry)

    return consensus


def check_api_status() -> Dict[str, Any]:
    """Check Odds API status and remaining requests."""
    api_key = _get_api_key()
    if not api_key:
        return {"status": "no_key", "message": "ODDS_API_KEY not set"}

    try:
        response = httpx.get(
            f"{ODDS_API_BASE_URL}/sports",
            params={"apiKey": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        return {
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "requests_remaining": response.headers.get("x-requests-remaining"),
            "requests_used": response.headers.get("x-requests-used"),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Collector-protocol wrapper
# ---------------------------------------------------------------------------

class OddsApiCollector:
    """Evidence-class collector backed by The Odds API.

    Serves ``odds`` evidence type.
    Implements the :class:`~omega.evidence.collectors.base.Collector` protocol.
    """

    @property
    def name(self) -> str:
        return "odds_api"

    @property
    def evidence_types(self) -> set[str]:
        return {"odds"}

    @property
    def supported_leagues(self) -> set[str]:
        return set(LEAGUE_SPORT_MAPPING.keys())

    @property
    def trust_tier(self) -> int:
        return 1

    def collect(
        self,
        entity: str,
        league: str,
        data_type: str,
    ):
        """Collect odds evidence from The Odds API.

        Returns a :class:`CollectorResult` or ``None``.
        """
        from omega.evidence.collectors.base import CollectorResult

        league_upper = league.upper()
        if league_upper not in self.supported_leagues:
            return None
        if data_type != "odds":
            return None

        try:
            games = get_upcoming_odds(league_upper)
            if not games:
                return None

            entity_lower = entity.lower()
            matched = [
                g for g in games
                if entity_lower in g.get("home_team", "").lower()
                or entity_lower in g.get("away_team", "").lower()
            ]

            if matched:
                consensus = extract_consensus_odds(matched)
                return CollectorResult(
                    data={"odds": consensus, "raw_games": matched},
                    source="odds_api",
                    method="structured_api",
                    trust_tier=1,
                    confidence=0.95,
                    entity_matched=entity,
                )

            # No entity match — return league-wide odds
            consensus = extract_consensus_odds(games)
            return CollectorResult(
                data={"odds": consensus},
                source="odds_api",
                method="structured_api",
                trust_tier=1,
                confidence=0.90,
                entity_matched=entity,
            )

        except Exception as exc:
            logger.debug("OddsApiCollector.collect failed: %s", exc)
            return None
