"""
omega.integrations.espn_wnba -- ESPN public scoreboard for WNBA final scores.

In-season live fetch path for WNBA. Mirrors espn_nba.py (same ESPN basketball
scoreboard shape) with a WNBA team-alias map and the WNBA scoreboard URL. The
richer historical source for WNBA replay artifacts is wehoop (see
scripts/refresh_wehoop.py); this module is the live path only.
"""
from __future__ import annotations
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from omega.integrations._guards import assert_not_replay_mode

logger = logging.getLogger("omega.integrations.espn_wnba")
_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
_REQUEST_TIMEOUT_SECONDS = 15

WNBA_TEAMS: dict[str, list[str]] = {
    "Atlanta Dream": ["atl", "dream", "atlanta"],
    "Chicago Sky": ["chi", "sky", "chicago"],
    "Connecticut Sun": ["conn", "con", "sun", "connecticut"],
    "Dallas Wings": ["dal", "wings", "dallas"],
    "Golden State Valkyries": ["gs", "gsv", "valkyries", "golden state"],
    "Indiana Fever": ["ind", "fever", "indiana"],
    "Las Vegas Aces": ["lv", "lva", "las", "aces", "las vegas"],
    "Los Angeles Sparks": ["la", "las", "sparks", "los angeles"],
    "Minnesota Lynx": ["min", "lynx", "minnesota"],
    "New York Liberty": ["ny", "nyl", "liberty", "new york"],
    "Phoenix Mercury": ["phx", "pho", "mercury", "phoenix"],
    "Seattle Storm": ["sea", "storm", "seattle"],
    "Washington Mystics": ["was", "wsh", "mystics", "washington"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in WNBA_TEAMS.items():
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
    assert_not_replay_mode("ESPN WNBA scoreboard fetch")
    date_str = date.replace("-", "")
    query = urllib.parse.urlencode({"dates": date_str})
    url = f"{_SCOREBOARD_URL}?{query}"
    logger.debug("fetching ESPN WNBA scoreboard: %s", url)
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
            canonical = canonical_team(display_name) or canonical_team(team_blob.get("abbreviation", ""))
            if not canonical:
                logger.warning("Unmapped ESPN WNBA team: %r (abbr=%r)", display_name, team_blob.get("abbreviation"))
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
