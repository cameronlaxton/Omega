"""
ESPN API client -- free, no-auth schedule and scoreboard data.

Uses the public ESPN site API (site.api.espn.com) which requires no API key.
Provides schedules, scores, standings, and basic game details.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("omega.data.acquisition.espn")

ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports"

LEAGUE_PATHS: Dict[str, str] = {
    "NBA": "basketball/nba",
    "NFL": "football/nfl",
    "MLB": "baseball/mlb",
    "NHL": "hockey/nhl",
    "NCAAB": "basketball/mens-college-basketball",
    "NCAAF": "football/college-football",
    "WNBA": "basketball/wnba",
    "MLS": "soccer/usa.1",
}

REQUEST_TIMEOUT = 8.0
RATE_LIMIT_DELAY = 0.3
_last_request_time: float = 0.0


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()


def _get_league_path(league: str) -> Optional[str]:
    return LEAGUE_PATHS.get(league.upper())


def _make_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Make a request to the ESPN API."""
    _rate_limit()

    url = f"{ESPN_API_BASE}/{endpoint}"

    try:
        response = httpx.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            logger.warning("ESPN API returned %d for %s", response.status_code, endpoint)
            return None
        return response.json()
    except httpx.TimeoutException:
        logger.error("ESPN API request timed out: %s", endpoint)
        return None
    except httpx.HTTPError as exc:
        logger.error("ESPN API request failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_competitors(competition: Dict[str, Any]) -> tuple:
    """Parse home/away team info from a competition dict."""
    home_team = None
    away_team = None

    for comp in competition.get("competitors", []):
        team_info = {
            "id": comp.get("id"),
            "name": comp.get("team", {}).get("displayName", ""),
            "abbreviation": comp.get("team", {}).get("abbreviation", ""),
            "score": comp.get("score", "0"),
            "record": (
                comp.get("records", [{}])[0].get("summary", "")
                if comp.get("records")
                else ""
            ),
        }

        if comp.get("homeAway") == "home":
            home_team = team_info
        else:
            away_team = team_info

    return home_team, away_team


def _parse_odds(competition: Dict[str, Any]) -> Dict[str, Any]:
    """Parse odds data from a competition dict."""
    odds = competition.get("odds", [])
    if not odds:
        return {}

    item = odds[0]
    return {
        "spread": item.get("details", ""),
        "over_under": item.get("overUnder"),
        "spread_home": item.get("spread"),
        "provider": item.get("provider", {}).get("name", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_todays_games(league: str) -> List[Dict[str, Any]]:
    """Get today's games for a league.

    Args:
        league: League code (NBA, NFL, MLB, NHL, NCAAB, NCAAF, WNBA)

    Returns:
        List of games scheduled for today with team info and odds.
    """
    league_path = _get_league_path(league)
    if not league_path:
        logger.warning("Unsupported league for ESPN: %s", league)
        return []

    today = datetime.now().strftime("%Y%m%d")
    data = _make_request(f"{league_path}/scoreboard", params={"dates": today})

    if data is None:
        return []

    games: List[Dict[str, Any]] = []

    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        home_team, away_team = _parse_competitors(competition)
        odds_data = _parse_odds(competition)

        games.append({
            "game_id": event.get("id", ""),
            "league": league.upper(),
            "name": event.get("name", ""),
            "short_name": event.get("shortName", ""),
            "date": event.get("date", ""),
            "status": event.get("status", {}).get("type", {}).get("description", ""),
            "status_detail": event.get("status", {}).get("type", {}).get("detail", ""),
            "venue": competition.get("venue", {}).get("fullName", ""),
            "home_team": home_team,
            "away_team": away_team,
            "odds": odds_data,
            "broadcast": (
                competition.get("broadcasts", [{}])[0].get("names", [])
                if competition.get("broadcasts")
                else []
            ),
        })

    logger.info("ESPN returned %d games for %s today", len(games), league)
    return games


def get_upcoming_games(league: str, days: int = 7) -> List[Dict[str, Any]]:
    """Get upcoming games for a league over the next N days.

    Args:
        league: League code
        days: Number of days to look ahead (default 7)

    Returns:
        List of upcoming games.
    """
    league_path = _get_league_path(league)
    if not league_path:
        return []

    all_games: List[Dict[str, Any]] = []
    current_date = datetime.now()

    for i in range(days):
        date = current_date + timedelta(days=i)
        date_str = date.strftime("%Y%m%d")

        data = _make_request(f"{league_path}/scoreboard", params={"dates": date_str})
        if not data:
            continue

        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            home_team, away_team = _parse_competitors(competition)

            all_games.append({
                "game_id": event.get("id", ""),
                "league": league.upper(),
                "name": event.get("name", ""),
                "date": event.get("date", ""),
                "status": event.get("status", {}).get("type", {}).get("description", ""),
                "home_team": home_team,
                "away_team": away_team,
            })

    return all_games


def get_game_details(game_id: str, league: str) -> Optional[Dict[str, Any]]:
    """Get detailed information for a specific game.

    Args:
        game_id: ESPN event ID
        league: League code

    Returns:
        Detailed game information or None.
    """
    league_path = _get_league_path(league)
    if not league_path:
        return None

    data = _make_request(f"{league_path}/summary", params={"event": game_id})
    if not data:
        return None

    return _parse_game_details(data, game_id, league)


def _parse_game_details(data: Dict[str, Any], game_id: str, league: str) -> Dict[str, Any]:
    """Parse detailed game data from ESPN summary response."""
    game_info = data.get("gameInfo", {})
    boxscore = data.get("boxscore", {})
    leaders = data.get("leaders", [])
    predictor = data.get("predictor", {})

    details: Dict[str, Any] = {
        "game_id": game_id,
        "league": league,
        "venue": game_info.get("venue", {}).get("fullName", ""),
        "attendance": game_info.get("attendance"),
        "weather": game_info.get("weather", {}),
        "officials": [
            official.get("displayName", "")
            for official in game_info.get("officials", [])
        ],
        "teams": [],
        "leaders": [],
        "win_probability": {},
    }

    for team_data in boxscore.get("teams", []):
        team = team_data.get("team", {})
        details["teams"].append({
            "id": team.get("id"),
            "name": team.get("displayName", ""),
            "abbreviation": team.get("abbreviation", ""),
            "stats": [
                {"name": stat.get("label", ""), "value": stat.get("displayValue", "")}
                for stat in team_data.get("statistics", [])
            ],
        })

    for leader in leaders:
        details["leaders"].append({
            "team": leader.get("team", {}).get("displayName", ""),
            "categories": [
                {
                    "name": cat.get("displayName", ""),
                    "leaders": [
                        {
                            "name": ld.get("athlete", {}).get("displayName", ""),
                            "value": ld.get("displayValue", ""),
                        }
                        for ld in cat.get("leaders", [])
                    ],
                }
                for cat in leader.get("leaders", [])
            ],
        })

    if predictor:
        details["win_probability"] = {
            "home": predictor.get("homeTeam", {}).get("gameProjection"),
            "away": predictor.get("awayTeam", {}).get("gameProjection"),
        }

    return details


def get_scoreboard(league: str, date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get scoreboard with scores for a league on a specific date.

    Args:
        league: League code
        date: Date in YYYYMMDD format (default: today)

    Returns:
        List of games with scores and status.
    """
    league_path = _get_league_path(league)
    if not league_path:
        return []

    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    data = _make_request(f"{league_path}/scoreboard", params={"dates": date})
    if data is None:
        return []

    games: List[Dict[str, Any]] = []

    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        status_obj = event.get("status", {})

        home_team = ""
        away_team = ""
        home_score = 0
        away_score = 0

        for comp in competition.get("competitors", []):
            team_name = comp.get("team", {}).get("displayName", "")
            score_str = comp.get("score", "0")
            try:
                score = int(score_str) if score_str else 0
            except ValueError:
                score = 0

            if comp.get("homeAway") == "home":
                home_team = team_name
                home_score = score
            else:
                away_team = team_name
                away_score = score

        status_type = status_obj.get("type", {})
        is_final = status_type.get("completed", False) or status_type.get("name") == "STATUS_FINAL"

        games.append({
            "game_id": event.get("id", ""),
            "league": league.upper(),
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "status": "Final" if is_final else status_type.get("description", "Scheduled"),
            "date": event.get("date", ""),
        })

    return games


def get_standings(league: str) -> List[Dict[str, Any]]:
    """Get current standings for a league."""
    league_path = _get_league_path(league)
    if not league_path:
        return []

    data = _make_request(f"{league_path}/standings")
    if data is None:
        return []

    standings: List[Dict[str, Any]] = []
    for group in data.get("children", []):
        division_name = group.get("name", "")
        for entry in group.get("standings", {}).get("entries", []):
            team = entry.get("team", {})
            stats = {
                stat.get("name"): stat.get("displayValue", "")
                for stat in entry.get("stats", [])
            }
            standings.append({
                "team_id": team.get("id"),
                "team_name": team.get("displayName", ""),
                "abbreviation": team.get("abbreviation", ""),
                "division": division_name,
                "stats": stats,
            })

    return standings


def check_api_status() -> Dict[str, Any]:
    """Check ESPN API availability."""
    try:
        response = httpx.get(
            f"{ESPN_API_BASE}/basketball/nba/scoreboard",
            timeout=REQUEST_TIMEOUT,
        )
        return {
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "available_leagues": list(LEAGUE_PATHS.keys()),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Collector-protocol wrapper
# ---------------------------------------------------------------------------

class EspnCollector:
    """Evidence-class collector backed by the ESPN public API.

    Serves ``schedule`` and ``team_stat`` (via standings) evidence types.
    Implements the :class:`~omega.evidence.collectors.base.Collector` protocol.
    """

    @property
    def name(self) -> str:
        return "espn"

    @property
    def evidence_types(self) -> set[str]:
        return {"schedule", "team_stat"}

    @property
    def supported_leagues(self) -> set[str]:
        return set(LEAGUE_PATHS.keys())

    @property
    def trust_tier(self) -> int:
        return 1

    def collect(
        self,
        entity: str,
        league: str,
        data_type: str,
    ):
        """Collect evidence from ESPN.

        Returns a :class:`CollectorResult` or ``None``.
        """
        from omega.evidence.collectors.base import CollectorResult

        league_upper = league.upper()
        if league_upper not in self.supported_leagues:
            return None
        if data_type not in self.evidence_types:
            return None

        try:
            if data_type == "schedule":
                return self._collect_schedule(entity, league_upper)
            if data_type == "team_stat":
                return self._collect_team_stats(entity, league_upper)
        except Exception as exc:
            logger.debug("EspnCollector.collect failed: %s", exc)
        return None

    # -- internal -----------------------------------------------------------

    def _collect_schedule(self, entity: str, league: str):
        from omega.evidence.collectors.base import CollectorResult

        games = get_todays_games(league)
        if not games:
            return None

        entity_lower = entity.lower()
        matched = [
            g for g in games
            if entity_lower in (g.get("home_team") or {}).get("name", "").lower()
            or entity_lower in (g.get("away_team") or {}).get("name", "").lower()
            or entity_lower in g.get("name", "").lower()
        ]
        data = matched if matched else games
        return CollectorResult(
            data={"games": data},
            source="espn",
            method="structured_api",
            trust_tier=1,
            confidence=0.95,
            entity_matched=entity,
        )

    def _collect_team_stats(self, entity: str, league: str):
        from omega.evidence.collectors.base import CollectorResult

        standings = get_standings(league)
        if not standings:
            return None

        entity_lower = entity.lower()
        matched = [
            s for s in standings
            if entity_lower in s.get("team_name", "").lower()
            or entity_lower == s.get("abbreviation", "").lower()
        ]
        if not matched:
            return None

        return CollectorResult(
            data={"standings": matched},
            source="espn",
            method="structured_api",
            trust_tier=1,
            confidence=0.85,
            entity_matched=entity,
        )
