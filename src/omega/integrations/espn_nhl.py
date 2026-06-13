"""
omega.integrations.espn_nhl -- ESPN public scoreboard for NHL final scores.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from omega.integrations._guards import assert_not_replay_mode

logger = logging.getLogger("omega.integrations.espn_nhl")
_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
_REQUEST_TIMEOUT_SECONDS = 15

NHL_TEAMS: dict[str, list[str]] = {
    "Anaheim Ducks": ["ana", "ducks", "anaheim"],
    "Boston Bruins": ["bos", "bruins", "boston"],
    "Buffalo Sabres": ["buf", "sabres", "buffalo"],
    "Calgary Flames": ["cgy", "flames", "calgary"],
    "Carolina Hurricanes": ["car", "hurricanes", "carolina"],
    "Chicago Blackhawks": ["chi", "blackhawks", "chicago"],
    "Colorado Avalanche": ["col", "avalanche", "colorado"],
    "Columbus Blue Jackets": ["cbj", "blue jackets", "columbus"],
    "Dallas Stars": ["dal", "stars", "dallas"],
    "Detroit Red Wings": ["det", "red wings", "detroit"],
    "Edmonton Oilers": ["edm", "oilers", "edmonton"],
    "Florida Panthers": ["fla", "panthers", "florida"],
    "Los Angeles Kings": ["la", "kings", "los angeles"],
    "Minnesota Wild": ["min", "wild", "minnesota"],
    "Montreal Canadiens": ["mtl", "canadiens", "montreal"],
    "Nashville Predators": ["nsh", "predators", "nashville"],
    "New Jersey Devils": ["nj", "devils", "new jersey"],
    "New York Islanders": ["nyi", "islanders", "new york"],
    "New York Rangers": ["nyr", "rangers", "new york"],
    "Ottawa Senators": ["ott", "senators", "ottawa"],
    "Philadelphia Flyers": ["phi", "flyers", "philadelphia"],
    "Pittsburgh Penguins": ["pit", "penguins", "pittsburgh"],
    "San Jose Sharks": ["sj", "sharks", "san jose"],
    "Seattle Kraken": ["sea", "kraken", "seattle"],
    "St. Louis Blues": ["stl", "blues", "st. louis"],
    "Tampa Bay Lightning": ["tb", "lightning", "tampa bay"],
    "Toronto Maple Leafs": ["tor", "maple leafs", "toronto"],
    "Utah Hockey Club": ["utah", "hockey club", "utah"],
    "Vancouver Canucks": ["van", "canucks", "vancouver"],
    "Vegas Golden Knights": ["vgk", "golden knights", "vegas"],
    "Washington Capitals": ["wsh", "capitals", "washington"],
    "Winnipeg Jets": ["wpg", "jets", "winnipeg"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in NHL_TEAMS.items():
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical


@dataclass(frozen=True)
class FinalGame:
    event_id: str
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: str


def canonical_team(name_or_alias: str) -> str | None:
    if not name_or_alias:
        return None
    key = name_or_alias.strip().lower()
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    for alias_key, canonical in _ALIAS_TO_CANONICAL.items():
        if len(alias_key) >= 4 and alias_key in key:
            return canonical
    return None


def fetch_scoreboard(date: str, url_opener=urllib.request.urlopen) -> list[FinalGame]:
    assert_not_replay_mode("ESPN NHL scoreboard fetch")
    date_str = date.replace("-", "")
    query = urllib.parse.urlencode({"dates": date_str})
    url = f"{_SCOREBOARD_URL}?{query}"
    logger.debug("fetching ESPN scoreboard: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_scoreboard(payload)


def parse_scoreboard(payload: dict) -> list[FinalGame]:
    results: list[FinalGame] = []
    for event in payload.get("events") or []:
        event_id = str(event.get("id") or "")
        iso_date = (event.get("date") or "")[:10]
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        status_obj = (comp.get("status") or {}).get("type") or {}
        status = (status_obj.get("name") or "").lower()
        status_short = status.replace("status_", "")
        if not status_short.startswith("final"):
            logger.debug("skipping event %s - non-final status %r", event_id, status_short)
            continue
        home = away = None
        home_score = away_score = 0
        for competitor in comp.get("competitors") or []:
            team_blob = competitor.get("team") or {}
            display_name = team_blob.get("displayName") or team_blob.get("name") or ""
            canonical = canonical_team(display_name) or canonical_team(team_blob.get("abbreviation", ""))
            if not canonical:
                logger.warning("Unmapped ESPN team: %r (abbr=%r)", display_name, team_blob.get("abbreviation"))
                canonical = display_name
            score = int(competitor.get("score") or 0)
            if competitor.get("homeAway") == "home":
                home, home_score = canonical, score
            elif competitor.get("homeAway") == "away":
                away, away_score = canonical, score
        if not home or not away:
            logger.debug("skipping event %s - missing home/away", event_id)
            continue
        results.append(FinalGame(
            event_id=event_id,
            date=iso_date,
            home_team=home,
            away_team=away,
            home_score=home_score,
            away_score=away_score,
            status=status_short,
        ))
    return results


def fetch_team_context(team_name: str, url_opener=urllib.request.urlopen) -> dict[str, float]:
    """Fetch deep historical baseline stats for an NHL team."""
    c_team = canonical_team(team_name)
    if not c_team:
        return {}
    abbr = NHL_TEAMS[c_team][0]
    url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{abbr}/statistics"
    try:
        with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        stats = {}
        if "results" in data and "stats" in data["results"] and "categories" in data["results"]["stats"]:
            for cat in data["results"]["stats"]["categories"]:
                for s in cat.get("stats", []):
                    stats[s["name"]] = float(s.get("value", 0.0))
        return stats
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch team stats for %s: %s", team_name, exc)
        return {}


def fetch_player_context(player_id: str, url_opener=urllib.request.urlopen) -> dict[str, Any]:
    """Fetch seasonal statistics for an NHL player. player_id must be the ESPN athlete ID."""
    url = f"https://site.api.espn.com/apis/common/v3/sports/hockey/nhl/athletes/{player_id}"
    try:
        with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        stats = {}
        athlete = data.get("athlete", {})
        summary = athlete.get("statsSummary", {})
        for s in summary.get("statistics", []):
            stats[s["name"]] = float(s.get("value", 0.0))
        return stats
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch player stats for %s: %s", player_id, exc)
        return {}
