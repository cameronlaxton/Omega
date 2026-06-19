"""
omega.integrations.espn_soccer -- ESPN public scoreboard for soccer final scores.

Live in-season fetch path for the soccer leagues Omega simulates (see
``omega/core/config/leagues.py`` — MLS, EPL, LA_LIGA, BUNDESLIGA, SERIE_A,
LIGUE_1, CHAMPIONS_LEAGUE, LIGA_MX). Mirrors espn_wnba.py but with two
soccer-specific differences:

1. ESPN's soccer scoreboard is **per-competition**, so :func:`fetch_scoreboard`
   takes an Omega league code and resolves it to the right ESPN league slug via
   :data:`SOCCER_LEAGUE_SLUGS`.
2. Soccer is a **3-way** result (home win / draw / away win). ESPN marks a
   completed match with ``STATUS_FULL_TIME``; :func:`parse_scoreboard`
   normalizes any completed match to ``status == "final"`` so the outcome-fetch
   script can use the same ``status == "final"`` filter as the other sports.
   Draw detection itself is handled downstream by ``TraceStore.attach_outcome``
   (equal scores → ``result == "draw"``).

Team-name canonicalization across hundreds of clubs cannot be fully enumerated,
so :func:`canonical_team` maps a curated set of common aliases and otherwise
falls back to a normalized form of the input. Both the trace's stored team name
and the ESPN display name pass through the same function, so identical names
match even when not in the alias table; mismatched short names are reported as
unmatched and can be added to :data:`SOCCER_TEAM_ALIASES` over time.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass

from omega.integrations._guards import assert_not_replay_mode

logger = logging.getLogger("omega.integrations.espn_soccer")

_SCOREBOARD_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
_REQUEST_TIMEOUT_SECONDS = 15

# Omega league code -> ESPN soccer competition slug.
SOCCER_LEAGUE_SLUGS: dict[str, str] = {
    "MLS": "usa.1",
    "EPL": "eng.1",
    "PREMIER_LEAGUE": "eng.1",
    "LA_LIGA": "esp.1",
    "LALIGA": "esp.1",
    "BUNDESLIGA": "ger.1",
    "SERIE_A": "ita.1",
    "LIGUE_1": "fra.1",
    "CHAMPIONS_LEAGUE": "uefa.champions",
    "LIGA_MX": "mex.1",
    "WORLD_CUP": "fifa.world",
    "FIFA_WORLD_CUP_2026": "fifa.world",
}

# Curated alias -> canonical club name. Not exhaustive: unknown clubs fall back
# to a normalized form of the ESPN display name (see canonical_team). Extend
# this when fetch_outcomes_soccer.py reports an unmatched trace.
SOCCER_TEAM_ALIASES: dict[str, str] = {
    # England
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester united": "Manchester United",
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "wolves": "Wolverhampton Wanderers",
    "newcastle": "Newcastle United",
    "west ham": "West Ham United",
    "brighton": "Brighton & Hove Albion",
    "nottingham forest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    # Spain
    "barca": "Barcelona",
    "barça": "Barcelona",
    "fc barcelona": "Barcelona",
    "real": "Real Madrid",
    "real madrid": "Real Madrid",
    "atletico": "Atletico Madrid",
    "atlético madrid": "Atletico Madrid",
    "athletic": "Athletic Club",
    # Germany
    "bayern": "Bayern Munich",
    "fc bayern": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "gladbach": "Borussia Monchengladbach",
    "leverkusen": "Bayer Leverkusen",
    # Italy
    "inter": "Inter Milan",
    "internazionale": "Inter Milan",
    "milan": "AC Milan",
    "juve": "Juventus",
    "napoli": "Napoli",
    "roma": "AS Roma",
    # France
    "psg": "Paris Saint-Germain",
    "paris": "Paris Saint-Germain",
    "paris saint germain": "Paris Saint-Germain",
    "paris saint-germain": "Paris Saint-Germain",
    "marseille": "Olympique Marseille",
    "om": "Olympique Marseille",
    "lyon": "Olympique Lyonnais",
    # MLS / Liga MX (common short forms)
    "lafc": "Los Angeles FC",
    "la galaxy": "LA Galaxy",
    "nycfc": "New York City FC",
    "atlanta": "Atlanta United",
    "america": "Club America",
    "club américa": "Club America",
    "chivas": "Guadalajara",
    # International / World Cup (national teams)
    "mexico": "Mexico",
    "el tri": "Mexico",
    "australia": "Australia",
    "socceroos": "Australia",
    "ecuador": "Ecuador",
    "saudi arabia": "Saudi Arabia",
    "saudi": "Saudi Arabia",
    "argentina": "Argentina",
    "brazil": "Brazil",
    "brasil": "Brazil",
    "france": "France",
    "germany": "Germany",
    "england": "England",
    "spain": "Spain",
    "usa": "United States",
    "united states": "United States",
    "usmnt": "United States",
    "bosnia & herzegovina": "Bosnia-Herzegovina",
    "bosnia-herzegovina": "Bosnia-Herzegovina",
    "turkey": "Türkiye",
    "türkiye": "Türkiye",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
}


@dataclass(frozen=True)
class FinalGame:
    event_id: str
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: str
    league: str = ""
    # True when ESPN reports the match was decided after regulation (extra time
    # or a penalty shootout). For a SINGLE-LEG knockout this means the sides were
    # level at 90' — i.e. the 3-way (1X2) "draw" market settles as a draw even
    # though the ESPN score may show the ET/shootout winner. ``status_detail``
    # keeps the raw ESPN status name for provenance/auditing.
    decided_after_regulation: bool = False
    status_detail: str = ""


def _normalize(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def canonical_team(name_or_alias: str) -> str | None:
    """Resolve a club name/alias to a canonical name.

    Returns an alias-table hit when available, otherwise a normalized
    (whitespace-collapsed, title-cased) form of the input so identical ESPN /
    trace names still match. Returns None only for empty input.
    """
    if not name_or_alias:
        return None
    key = _normalize(name_or_alias)
    if not key:
        return None
    if key in SOCCER_TEAM_ALIASES:
        return SOCCER_TEAM_ALIASES[key]
    return name_or_alias.strip()


def espn_slug(league: str) -> str | None:
    """ESPN competition slug for an Omega soccer league code, or None."""
    return SOCCER_LEAGUE_SLUGS.get((league or "").upper())


def fetch_scoreboard(
    date: str,
    league: str,
    url_opener=urllib.request.urlopen,
) -> list[FinalGame]:
    """Fetch the ESPN scoreboard for one soccer ``league`` on ``date`` (ISO).

    Raises ValueError for an unknown league code.
    """
    assert_not_replay_mode("ESPN soccer scoreboard fetch")
    slug = espn_slug(league)
    if slug is None:
        raise ValueError(f"Unknown soccer league code: {league!r}")
    date_str = date.replace("-", "")
    query = urllib.parse.urlencode({"dates": date_str})

    slugs = [slug]
    if league.upper() == "WORLD_CUP":
        slugs.append("fifa.friendly")

    results: list[FinalGame] = []
    for s in slugs:
        url = f"{_SCOREBOARD_BASE}/{s}/scoreboard?{query}"
        logger.debug("fetching ESPN soccer scoreboard: %s", url)
        try:
            with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            results.extend(parse_scoreboard(payload, league=league))
        except Exception as exc:
            logger.warning("Failed to fetch scoreboard for %s on %s: %s", s, date, exc)
            if len(slugs) == 1:
                raise
    return results


def parse_scoreboard(payload: dict, league: str = "") -> list[FinalGame]:
    """Parse an ESPN soccer scoreboard payload into FinalGame rows.

    Completed matches (``status.type.completed`` true, or ``state == "post"``)
    are normalized to ``status == "final"`` so callers can filter uniformly
    across sports. In-progress/scheduled matches keep their ESPN short name.
    """
    results: list[FinalGame] = []
    for event in payload.get("events") or []:
        event_id = str(event.get("id") or "")
        iso_date = (event.get("date") or "")[:10]
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        status_obj = (comp.get("status") or {}).get("type") or {}
        completed = bool(status_obj.get("completed")) or (status_obj.get("state") == "post")
        raw_status_name = status_obj.get("name") or ""
        if completed:
            status = "final"
        else:
            status = raw_status_name.lower().replace("status_", "")
        # Detect extra-time / penalty-shootout finishes (info ESPN otherwise drops
        # once we normalize to "final"). Match on the status name + detail text.
        _et_pen_text = f"{raw_status_name} {status_obj.get('detail') or ''} {status_obj.get('description') or ''}".upper()
        decided_after_regulation = any(
            tok in _et_pen_text
            for tok in ("_ET", "EXTRA_TIME", "EXTRATIME", "AET", "PENALT", "SHOOTOUT", "_PEN")
        )
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
                    "Unmapped ESPN soccer team: %r (abbr=%r)",
                    display_name,
                    team_blob.get("abbreviation"),
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
                status=status,
                league=league,
                decided_after_regulation=decided_after_regulation,
                status_detail=raw_status_name,
            )
        )
    return results
