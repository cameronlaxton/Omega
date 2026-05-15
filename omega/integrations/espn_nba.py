"""
omega.integrations.espn_nba — ESPN public scoreboard for NBA final scores.

The ESPN site API serves public scoreboards at:
  https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD

No authentication required. Returns events with team abbreviations, names,
locations, scores, and status. We expose two things:
  - `fetch_scoreboard(date)` — raw HTTP fetch + JSON parse, returns FinalGame list
  - `canonical_team(name_or_alias)` — fuzzy lookup against a known alias table

Stale aliases are the most common source of unmatched outcomes. When the
weekly health report flags an unmapped team string, add it to TEAM_ALIASES.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("omega.integrations.espn_nba")

_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
_REQUEST_TIMEOUT_SECONDS = 15


# Canonical team names (left side = canonical, right side = aliases)
# Canonical name matches what omega_lite_standalone treats as the official label.
NBA_TEAMS: Dict[str, List[str]] = {
    "Atlanta Hawks":        ["atl", "hawks", "atlanta"],
    "Boston Celtics":       ["bos", "celtics", "boston"],
    "Brooklyn Nets":        ["bkn", "bro", "nets", "brooklyn"],
    "Charlotte Hornets":    ["cha", "hornets", "charlotte"],
    "Chicago Bulls":        ["chi", "bulls", "chicago"],
    "Cleveland Cavaliers":  ["cle", "cavs", "cavaliers", "cleveland"],
    "Dallas Mavericks":     ["dal", "mavs", "mavericks", "dallas"],
    "Denver Nuggets":       ["den", "nuggets", "denver"],
    "Detroit Pistons":      ["det", "pistons", "detroit"],
    "Golden State Warriors": ["gs", "gsw", "warriors", "golden state"],
    "Houston Rockets":      ["hou", "rockets", "houston"],
    "Indiana Pacers":       ["ind", "pacers", "indiana"],
    "LA Clippers":          ["lac", "clippers", "los angeles clippers"],
    "Los Angeles Lakers":   ["lal", "lakers", "la lakers"],
    "Memphis Grizzlies":    ["mem", "grizzlies", "memphis"],
    "Miami Heat":           ["mia", "heat", "miami"],
    "Milwaukee Bucks":      ["mil", "bucks", "milwaukee"],
    "Minnesota Timberwolves": ["min", "wolves", "timberwolves", "minnesota"],
    "New Orleans Pelicans": ["nop", "no", "pelicans", "new orleans"],
    "New York Knicks":      ["ny", "nyk", "knicks", "new york"],
    "Oklahoma City Thunder": ["okc", "thunder", "oklahoma city"],
    "Orlando Magic":        ["orl", "magic", "orlando"],
    "Philadelphia 76ers":   ["phi", "sixers", "76ers", "philadelphia"],
    "Phoenix Suns":         ["phx", "pho", "suns", "phoenix"],
    "Portland Trail Blazers": ["por", "blazers", "trail blazers", "portland"],
    "Sacramento Kings":     ["sac", "kings", "sacramento"],
    "San Antonio Spurs":    ["sa", "sas", "spurs", "san antonio"],
    "Toronto Raptors":      ["tor", "raptors", "toronto"],
    "Utah Jazz":            ["uta", "jazz", "utah"],
    "Washington Wizards":   ["was", "wsh", "wizards", "washington"],
}

# Reverse map: alias (lowercased) → canonical name
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _aliases in NBA_TEAMS.items():
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical


@dataclass(frozen=True)
class FinalGame:
    """A completed NBA game with attributed final scores."""
    event_id: str               # ESPN event id
    date: str                   # YYYY-MM-DD (Eastern game date)
    home_team: str              # canonical name
    away_team: str              # canonical name
    home_score: int
    away_score: int
    status: str                 # "final", "in_progress", "scheduled", "postponed", ...


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

def canonical_team(name_or_alias: str) -> Optional[str]:
    """Resolve a team string to its canonical NBA team name.

    Returns None if the string does not match any known alias. Callers should
    log unmapped strings so the table can be extended.
    """
    if not name_or_alias:
        return None
    key = name_or_alias.strip().lower()
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    # Substring match only for aliases >= 4 chars to avoid false matches
    # (e.g. "ZZZ Not A Team" contains "no" — the 2-letter Pelicans abbreviation).
    for alias_key, canonical in _ALIAS_TO_CANONICAL.items():
        if len(alias_key) >= 4 and alias_key in key:
            return canonical
    return None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def fetch_scoreboard(
    date: str,
    url_opener=urllib.request.urlopen,
) -> List[FinalGame]:
    """Fetch the ESPN NBA scoreboard for a given Eastern game date.

    Args:
        date: YYYY-MM-DD or YYYYMMDD.
        url_opener: injectable for tests (defaults to urllib.request.urlopen).

    Returns:
        List of FinalGame (may include in-progress / scheduled if you call mid-day;
        consumers should filter by status == "final" for grading).
    """
    date_str = date.replace("-", "")
    query = urllib.parse.urlencode({"dates": date_str})
    url = f"{_SCOREBOARD_URL}?{query}"
    logger.debug("fetching ESPN scoreboard: %s", url)

    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    return parse_scoreboard(payload)


def parse_scoreboard(payload: dict) -> List[FinalGame]:
    """Parse the ESPN JSON envelope into a list of FinalGame.

    Public for testability — feed it a fixture dict to verify field extraction
    without making a network call.
    """
    results: List[FinalGame] = []
    for event in payload.get("events") or []:
        event_id = str(event.get("id") or "")
        iso_date = (event.get("date") or "")[:10]  # YYYY-MM-DD
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]

        status_obj = (comp.get("status") or {}).get("type") or {}
        status = (status_obj.get("name") or "").lower()
        # ESPN uses: STATUS_FINAL, STATUS_IN_PROGRESS, STATUS_SCHEDULED, STATUS_POSTPONED
        status_short = status.replace("status_", "")

        home = away = None
        home_score = away_score = 0
        for competitor in comp.get("competitors") or []:
            team_blob = competitor.get("team") or {}
            display_name = team_blob.get("displayName") or team_blob.get("name") or ""
            canonical = canonical_team(display_name) or canonical_team(team_blob.get("abbreviation", ""))
            if not canonical:
                logger.warning("Unmapped ESPN team: %r (abbr=%r)", display_name, team_blob.get("abbreviation"))
                canonical = display_name  # preserve raw so caller can flag
            score = int(competitor.get("score") or 0)
            if competitor.get("homeAway") == "home":
                home, home_score = canonical, score
            elif competitor.get("homeAway") == "away":
                away, away_score = canonical, score

        if not home or not away:
            logger.debug("skipping event %s — missing home/away", event_id)
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
