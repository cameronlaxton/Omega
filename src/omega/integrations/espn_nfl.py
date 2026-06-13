"""
omega.integrations.espn_nfl -- ESPN public scoreboard for NFL final scores.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from omega.integrations._guards import assert_not_replay_mode

logger = logging.getLogger("omega.integrations.espn_nfl")
_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
_REQUEST_TIMEOUT_SECONDS = 15

NFL_TEAMS: dict[str, list[str]] = {
    "Arizona Cardinals": ["ari", "cardinals", "arizona"],
    "Atlanta Falcons": ["atl", "falcons", "atlanta"],
    "Baltimore Ravens": ["bal", "ravens", "baltimore"],
    "Buffalo Bills": ["buf", "bills", "buffalo"],
    "Carolina Panthers": ["car", "panthers", "carolina"],
    "Chicago Bears": ["chi", "bears", "chicago"],
    "Cincinnati Bengals": ["cin", "bengals", "cincinnati"],
    "Cleveland Browns": ["cle", "browns", "cleveland"],
    "Dallas Cowboys": ["dal", "cowboys", "dallas"],
    "Denver Broncos": ["den", "broncos", "denver"],
    "Detroit Lions": ["det", "lions", "detroit"],
    "Green Bay Packers": ["gb", "packers", "green bay"],
    "Houston Texans": ["hou", "texans", "houston"],
    "Indianapolis Colts": ["ind", "colts", "indianapolis"],
    "Jacksonville Jaguars": ["jax", "jaguars", "jacksonville"],
    "Kansas City Chiefs": ["kc", "chiefs", "kansas city"],
    "Las Vegas Raiders": ["lv", "raiders", "las vegas"],
    "Los Angeles Chargers": ["lac", "chargers", "los angeles"],
    "Los Angeles Rams": ["lar", "rams", "los angeles"],
    "Miami Dolphins": ["mia", "dolphins", "miami"],
    "Minnesota Vikings": ["min", "vikings", "minnesota"],
    "New England Patriots": ["ne", "patriots", "new england"],
    "New Orleans Saints": ["no", "saints", "new orleans"],
    "New York Giants": ["nyg", "giants", "new york"],
    "New York Jets": ["nyj", "jets", "new york"],
    "Philadelphia Eagles": ["phi", "eagles", "philadelphia"],
    "Pittsburgh Steelers": ["pit", "steelers", "pittsburgh"],
    "San Francisco 49ers": ["sf", "49ers", "san francisco"],
    "Seattle Seahawks": ["sea", "seahawks", "seattle"],
    "Tampa Bay Buccaneers": ["tb", "buccaneers", "tampa bay"],
    "Tennessee Titans": ["ten", "titans", "tennessee"],
    "Washington Commanders": ["wsh", "commanders", "washington"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in NFL_TEAMS.items():
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
    assert_not_replay_mode("ESPN NFL scoreboard fetch")
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
    """Fetch deep historical baseline stats for an NFL team."""
    c_team = canonical_team(team_name)
    if not c_team:
        return {}
    abbr = NFL_TEAMS[c_team][0]
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}/statistics"
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
    """Fetch seasonal statistics for an NFL player. player_id must be the ESPN athlete ID."""
    url = f"https://site.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}"
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
