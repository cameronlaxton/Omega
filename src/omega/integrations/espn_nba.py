"""
omega.integrations.espn_nba -- ESPN public scoreboard for NBA final scores.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass

from omega.integrations._guards import assert_not_replay_mode

logger = logging.getLogger("omega.integrations.espn_nba")
_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
_REQUEST_TIMEOUT_SECONDS = 15

NBA_TEAMS: dict[str, list[str]] = {
    "Atlanta Hawks": ["atl", "hawks", "atlanta"],
    "Boston Celtics": ["bos", "celtics", "boston"],
    "Brooklyn Nets": ["bkn", "bro", "nets", "brooklyn"],
    "Charlotte Hornets": ["cha", "hornets", "charlotte"],
    "Chicago Bulls": ["chi", "bulls", "chicago"],
    "Cleveland Cavaliers": ["cle", "cavs", "cavaliers", "cleveland"],
    "Dallas Mavericks": ["dal", "mavs", "mavericks", "dallas"],
    "Denver Nuggets": ["den", "nuggets", "denver"],
    "Detroit Pistons": ["det", "pistons", "detroit"],
    "Golden State Warriors": ["gs", "gsw", "warriors", "golden state"],
    "Houston Rockets": ["hou", "rockets", "houston"],
    "Indiana Pacers": ["ind", "pacers", "indiana"],
    "LA Clippers": ["lac", "clippers", "los angeles clippers"],
    "Los Angeles Lakers": ["lal", "lakers", "la lakers"],
    "Memphis Grizzlies": ["mem", "grizzlies", "memphis"],
    "Miami Heat": ["mia", "heat", "miami"],
    "Milwaukee Bucks": ["mil", "bucks", "milwaukee"],
    "Minnesota Timberwolves": ["min", "wolves", "timberwolves", "minnesota"],
    "New Orleans Pelicans": ["nop", "no", "pelicans", "new orleans"],
    "New York Knicks": ["ny", "nyk", "knicks", "new york"],
    "Oklahoma City Thunder": ["okc", "thunder", "oklahoma city"],
    "Orlando Magic": ["orl", "magic", "orlando"],
    "Philadelphia 76ers": ["phi", "sixers", "76ers", "philadelphia"],
    "Phoenix Suns": ["phx", "pho", "suns", "phoenix"],
    "Portland Trail Blazers": ["por", "blazers", "trail blazers", "portland"],
    "Sacramento Kings": ["sac", "kings", "sacramento"],
    "San Antonio Spurs": ["sa", "sas", "spurs", "san antonio"],
    "Toronto Raptors": ["tor", "raptors", "toronto"],
    "Utah Jazz": ["uta", "jazz", "utah"],
    "Washington Wizards": ["was", "wsh", "wizards", "washington"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in NBA_TEAMS.items():
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
    assert_not_replay_mode("ESPN NBA scoreboard fetch")
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
        home = away = None
        home_score = away_score = 0
        for competitor in comp.get("competitors") or []:
            team_blob = competitor.get("team") or {}
            display_name = team_blob.get("displayName") or team_blob.get("name") or ""
            canonical = canonical_team(display_name) or canonical_team(
                team_blob.get("abbreviation", "")
            )
            if not canonical:
                logger.warning(
                    "Unmapped ESPN team: %r (abbr=%r)", display_name, team_blob.get("abbreviation")
                )
                canonical = display_name
            score = int(competitor.get("score") or 0)
            if competitor.get("homeAway") == "home":
                home, home_score = canonical, score
            elif competitor.get("homeAway") == "away":
                away, away_score = canonical, score
        if not home or not away:
            logger.debug("skipping event %s - missing home/away", event_id)
            continue
        results.append(
            FinalGame(
                event_id=event_id,
                date=iso_date,
                home_team=home,
                away_team=away,
                home_score=home_score,
                away_score=away_score,
                status=status_short,
            )
        )
    return results
