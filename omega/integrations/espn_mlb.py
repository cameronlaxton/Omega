"""
omega.integrations.espn_mlb — ESPN public scoreboard for MLB final scores.

The ESPN site API serves public scoreboards at:
  https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=YYYYMMDD

No authentication required. Returns events with team abbreviations, names,
locations, scores, and status. We expose two things:
  - `fetch_scoreboard(date)` — raw HTTP fetch + JSON parse, returns FinalGame list
  - `canonical_team(name_or_alias)` — fuzzy lookup against a known alias table

Stale aliases are the most common source of unmatched outcomes. When the
weekly health report flags an unmapped team string, add it to MLB_TEAMS.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger("omega.integrations.espn_mlb")

_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
_REQUEST_TIMEOUT_SECONDS = 15


# Canonical team names (left side = canonical, right side = aliases)
# Canonical name matches the official label used by the core contracts.
MLB_TEAMS: dict[str, list[str]] = {
    # American League East
    "Baltimore Orioles": ["bal", "orioles", "baltimore", "o's"],
    "Boston Red Sox": ["bos", "red sox", "boston"],
    "New York Yankees": ["nyy", "ny yankees", "yankees", "new york yankees", "bronx bombers"],
    "Tampa Bay Rays": ["tb", "tba", "rays", "tampa bay"],
    "Toronto Blue Jays": ["tor", "blue jays", "toronto"],
    # American League Central
    "Chicago White Sox": ["cws", "chw", "white sox", "chicago white sox"],
    "Cleveland Guardians": ["cle", "guardians", "cleveland"],
    "Detroit Tigers": ["det", "tigers", "detroit"],
    "Kansas City Royals": ["kc", "kcr", "royals", "kansas city"],
    "Minnesota Twins": ["min", "twins", "minnesota"],
    # American League West
    "Houston Astros": ["hou", "astros", "houston"],
    "Los Angeles Angels": ["laa", "angels", "los angeles angels", "anaheim", "anaheim angels"],
    "Athletics": [
        "oak",
        "a's",
        "oakland",
        "oakland athletics",
        "sacramento athletics",
        "sacramento",
    ],
    "Seattle Mariners": ["sea", "mariners", "seattle"],
    "Texas Rangers": ["tex", "rangers", "texas"],
    # National League East
    "Atlanta Braves": ["atl", "braves", "atlanta"],
    "Miami Marlins": ["mia", "marlins", "miami"],
    "New York Mets": ["nym", "ny mets", "mets", "new york mets"],
    "Philadelphia Phillies": ["phi", "phillies", "philadelphia"],
    "Washington Nationals": ["was", "wsh", "nats", "nationals", "washington"],
    # National League Central
    "Chicago Cubs": ["chc", "cubs", "chicago cubs"],
    "Cincinnati Reds": ["cin", "reds", "cincinnati"],
    "Milwaukee Brewers": ["mil", "brewers", "milwaukee"],
    "Pittsburgh Pirates": ["pit", "pirates", "pittsburgh"],
    "St. Louis Cardinals": ["stl", "cardinals", "st. louis", "saint louis", "st louis"],
    # National League West
    "Arizona Diamondbacks": ["ari", "ariz", "diamondbacks", "d-backs", "arizona"],
    "Colorado Rockies": ["col", "rockies", "colorado"],
    "Los Angeles Dodgers": ["lad", "dodgers", "los angeles dodgers"],
    "San Diego Padres": ["sd", "sdp", "padres", "san diego"],
    "San Francisco Giants": ["sf", "sfg", "giants", "san francisco"],
}

# Reverse map: alias (lowercased) → canonical name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in MLB_TEAMS.items():
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical


@dataclass(frozen=True)
class FinalGame:
    """A completed MLB game with attributed final scores."""

    event_id: str  # ESPN event id
    date: str  # YYYY-MM-DD (Eastern game date)
    home_team: str  # canonical name
    away_team: str  # canonical name
    home_score: int
    away_score: int
    status: str  # "final", "in_progress", "scheduled", "postponed", ...


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def canonical_team(name_or_alias: str) -> str | None:
    """Resolve a team string to its canonical MLB team name.

    Returns None if the string does not match any known alias. Callers should
    log unmapped strings so the table can be extended.
    """
    if not name_or_alias:
        return None
    key = name_or_alias.strip().lower()
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    # Substring match only for aliases >= 4 chars to avoid false matches.
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
) -> list[FinalGame]:
    """Fetch the ESPN MLB scoreboard for a given Eastern game date.

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


def parse_scoreboard(payload: dict) -> list[FinalGame]:
    """Parse the ESPN JSON envelope into a list of FinalGame.

    Public for testability — feed it a fixture dict to verify field extraction
    without making a network call.
    """
    results: list[FinalGame] = []
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
            canonical = canonical_team(display_name) or canonical_team(
                team_blob.get("abbreviation", "")
            )
            if not canonical:
                logger.warning(
                    "Unmapped ESPN team: %r (abbr=%r)", display_name, team_blob.get("abbreviation")
                )
                canonical = display_name  # preserve raw so caller can flag
            score = int(competitor.get("score") or 0)
            if competitor.get("homeAway") == "home":
                home, home_score = canonical, score
            elif competitor.get("homeAway") == "away":
                away, away_score = canonical, score

        if not home or not away:
            logger.debug("skipping event %s — missing home/away", event_id)
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
