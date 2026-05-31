"""
omega.integrations.espn_boxscore — ESPN public summary/box-score for player stats.

Used by scripts/fetch_outcomes_props.py to resolve a prop trace's player stat
against the actual game line. We separate the HTTP fetch from the JSON parse
so tests can hit `parse_box_score()` with a fixture instead of the network.

ESPN summary endpoints:
  NBA:  https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<event_id>
  MLB:  https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event=<event_id>
  Soccer: https://site.api.espn.com/apis/site/v2/sports/soccer/<slug>/summary?event=<event_id>
  (where <slug> comes from SOCCER_LEAGUE_SLUGS — e.g. "uefa.champions", "fifa.world")

The boxscore JSON shape (paraphrased):
    {
      "boxscore": {
        "players": [
          {
            "team": {"displayName": "Boston Celtics"},
            "statistics": [
              {
                "name": "starters" | "batting" | "pitching" | ...,
                "keys": ["MIN", "FG", "PTS", ...]      # NBA
                          | ["AB", "R", "H", "RBI", "HR", ...]  # MLB batting
                          | ["IP", "H", "R", "ER", "BB", "K", ...],  # MLB pitching
                "athletes": [
                  {"athlete": {"displayName": "Jayson Tatum"}, "stats": ["32", "10", "27", ...]},
                  ...
                ]
              },
              ...
            ]
          },
          ...
        ]
      }
    }

Unmapped prop_types are logged at WARNING and surfaced to the caller as
unresolved — we never silently zero them out.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from omega.integrations._guards import assert_not_replay_mode
from omega.integrations.espn_soccer import SOCCER_LEAGUE_SLUGS

logger = logging.getLogger("omega.integrations.espn_boxscore")

_REQUEST_TIMEOUT_SECONDS = 15
_SUMMARY_URLS = {
    "NBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
    "WNBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary",
    "MLB": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary",
}
_SOCCER_SUMMARY_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"


# ---------------------------------------------------------------------------
# Stat type mappings
# ---------------------------------------------------------------------------

# Map omega prop_type (lowercased) → ESPN stat-row key (string match against
# the `keys` array in the boxscore category). Multiple aliases per stat are
# allowed; the first key found in the category wins.
NBA_STAT_KEYS: dict[str, tuple[str, ...]] = {
    "pts": ("PTS", "points"),
    "points": ("PTS", "points"),
    "reb": ("REB", "rebounds"),
    "rebounds": ("REB", "rebounds"),
    "ast": ("AST", "assists"),
    "assists": ("AST", "assists"),
    "stl": ("STL", "steals"),
    "steals": ("STL", "steals"),
    "blk": ("BLK", "blocks"),
    "blocks": ("BLK", "blocks"),
    "3pm": ("3PTM", "3PM", "threePointFieldGoalsMade-threePointFieldGoalsAttempted"),
    "threes": ("3PTM", "3PM", "threePointFieldGoalsMade-threePointFieldGoalsAttempted"),
    "pra": ("PRA",),
}

# WNBA box-score stat keys are identical to the NBA layout (same ESPN basketball
# summary shape). Aliased rather than copied so the two never drift apart.
WNBA_STAT_KEYS: dict[str, tuple[str, ...]] = NBA_STAT_KEYS

# Batting and pitching live under separate categories — we surface them
# both and let the caller pick by prop_type semantics.
MLB_BATTING_KEYS: dict[str, tuple[str, ...]] = {
    "hits": ("H", "hits"),
    "runs": ("R", "runs"),
    "rbi": ("RBI", "RBIs"),
    "rbis": ("RBI", "RBIs"),
    "hr": ("HR", "homeRuns"),
    "home_runs": ("HR", "homeRuns"),
    "sb": ("SB", "stolenBases"),
    "stolen_bases": ("SB", "stolenBases"),
    "bb": ("BB", "walks"),
    "walks": ("BB", "walks"),
}

MLB_PITCHING_KEYS: dict[str, tuple[str, ...]] = {
    "strikeouts": ("K", "strikeouts"),
    "strikeouts_pitched": ("K", "strikeouts"),
    "k": ("K", "strikeouts"),
    "pitching_outs": ("IP", "fullInnings.partInnings"),
    "outs_recorded": ("IP", "fullInnings.partInnings"),
    "earned_runs": ("ER", "earnedRuns"),
    "er": ("ER", "earnedRuns"),
    "hits_allowed": ("H", "hits"),
    "walks_allowed": ("BB", "walks"),
}

# ESPN soccer summary keys (omega prop_type vocabulary, used by supported_prop_type).
SOCCER_STAT_KEYS: dict[str, tuple[str, ...]] = {
    "goals": ("totalGoals",),
    "assists": ("goalAssists",),
    "shots": ("totalShots",),
    "shots_on_target": ("shotsOnTarget",),
    "yellow_cards": ("yellowCards",),
    "red_cards": ("redCards",),
}

# ESPN soccer uses rosters[].roster[].stats (a list of {name, value} objects)
# rather than boxscore.players[].statistics[].keys/athletes. This maps omega
# prop_type → the ESPN roster stat `name` field.
SOCCER_ROSTER_STAT_MAP: dict[str, str] = {
    "goals": "totalGoals",
    "assists": "goalAssists",
    "shots": "totalShots",
    "shots_on_target": "shotsOnTarget",
    "yellow_cards": "yellowCards",
    "red_cards": "redCards",
}


# ---------------------------------------------------------------------------
# Player name normalization
# ---------------------------------------------------------------------------

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.\'`]")


def normalize_player_name(name: str | None) -> str:
    """Normalize a player name for cross-source matching.

    Strips accents, lowercases, removes punctuation (periods, apostrophes),
    collapses whitespace, drops trailing Jr./Sr./II/III/IV/V suffix.
    """
    if not name:
        return ""
    # Unicode → ASCII (strip accents)
    nkfd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nkfd if not unicodedata.combining(c))
    s = ascii_only.lower()
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _SUFFIX_RE.sub("", s).strip()
    return s


# ---------------------------------------------------------------------------
# Box score parsing
# ---------------------------------------------------------------------------


def _stat_key_map_for(
    league: str,
    category_name: str,
    keys: list[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Pick the appropriate stat-type map by league and category."""
    league_upper = league.upper()
    if league_upper == "NBA":
        return NBA_STAT_KEYS
    if league_upper == "WNBA":
        return WNBA_STAT_KEYS
    if league_upper == "MLB":
        cat = (category_name or "").lower()
        key_set = {key for key in (keys or [])}
        pitching_keys = {
            "IP",
            "K",
            "ER",
            "fullInnings.partInnings",
            "earnedRuns",
            "pitches-strikes",
            "ERA",
        }
        if "pitch" in cat or key_set.intersection(pitching_keys):
            return MLB_PITCHING_KEYS
        return MLB_BATTING_KEYS
    if league_upper in SOCCER_LEAGUE_SLUGS:
        return SOCCER_STAT_KEYS
    return {}


def _summary_url(league: str, event_id: str) -> str:
    """Build the ESPN summary URL for the given league and event."""
    league_upper = league.upper()
    if league_upper in _SUMMARY_URLS:
        return f"{_SUMMARY_URLS[league_upper]}?{urllib.parse.urlencode({'event': event_id})}"
    slug = SOCCER_LEAGUE_SLUGS.get(league_upper)
    if slug:
        return (
            f"{_SOCCER_SUMMARY_BASE}/{slug}/summary"
            f"?{urllib.parse.urlencode({'event': event_id})}"
        )
    raise ValueError(f"No ESPN summary endpoint configured for league={league!r}")


def _parse_ip_to_outs(value: str) -> float | None:
    """Convert MLB innings-pitched string like '6.2' (6 innings, 2 outs) to outs."""
    try:
        whole_str, _, frac_str = value.partition(".")
        whole = int(whole_str) if whole_str else 0
        frac = int(frac_str) if frac_str else 0
        return float(whole * 3 + frac)
    except (TypeError, ValueError):
        return None


def _to_number(value: Any) -> float | None:
    """Coerce an ESPN stat string ('27', '4-of-9', '32:14') to a float when sensible."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("--", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _made_from_made_attempted(value: Any) -> float | None:
    """Return the made count from an ESPN value like '4-9'."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    made, sep, _attempted = s.partition("-")
    if not sep:
        return _to_number(s)
    try:
        return float(made)
    except ValueError:
        return None


def _parse_soccer_roster(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Parse an ESPN soccer summary payload's ``rosters`` section.

    Soccer summary uses ``rosters[].roster[].stats`` (a list of ``{name, value}``
    objects) instead of the ``boxscore.players`` parallel-array format used by
    basketball and baseball. Returns ``{player_norm → {prop_type → value}}``.
    """
    out: dict[str, dict[str, float]] = {}
    for team_blob in payload.get("rosters") or []:
        for entry in team_blob.get("roster") or []:
            athlete = entry.get("athlete") or {}
            display = athlete.get("displayName") or athlete.get("shortName") or ""
            player_norm = normalize_player_name(display)
            if not player_norm:
                continue
            raw_stats: dict[str, float] = {}
            for stat in entry.get("stats") or []:
                name = stat.get("name")
                value = stat.get("value")
                if name and value is not None:
                    try:
                        raw_stats[name] = float(value)
                    except (TypeError, ValueError):
                        pass
            if not raw_stats:
                continue
            player_stats = out.setdefault(player_norm, {})
            for prop_type, espn_stat_name in SOCCER_ROSTER_STAT_MAP.items():
                if espn_stat_name in raw_stats:
                    player_stats.setdefault(prop_type, raw_stats[espn_stat_name])
    return out


def parse_box_score(
    payload: dict[str, Any],
    league: str,
) -> dict[str, dict[str, float]]:
    """Parse an ESPN summary payload into ``{player_norm → {stat_type → value}}``.

    Player names are normalized via :func:`normalize_player_name`. Stat types
    use the omega prop_type vocabulary (NBA_STAT_KEYS / MLB_BATTING_KEYS /
    MLB_PITCHING_KEYS / SOCCER_ROSTER_STAT_MAP). For soccer leagues the
    ``rosters`` section is used; all other sports use ``boxscore.players``.
    """
    if league.upper() in SOCCER_LEAGUE_SLUGS:
        return _parse_soccer_roster(payload)

    out: dict[str, dict[str, float]] = {}
    boxscore = payload.get("boxscore") or {}
    for team_blob in boxscore.get("players") or []:
        for category in team_blob.get("statistics") or []:
            cat_name = category.get("name") or ""
            keys: list[str] = list(category.get("keys") or [])
            if not keys:
                continue
            key_index = {k: i for i, k in enumerate(keys)}
            stat_map = _stat_key_map_for(league, cat_name, keys)
            if not stat_map:
                continue

            for athlete_blob in category.get("athletes") or []:
                athlete = athlete_blob.get("athlete") or {}
                display = athlete.get("displayName") or athlete.get("shortName") or ""
                player_norm = normalize_player_name(display)
                if not player_norm:
                    continue
                stats: list[Any] = list(athlete_blob.get("stats") or [])

                player_stats = out.setdefault(player_norm, {})
                for prop_type, espn_keys in stat_map.items():
                    if prop_type == "pra":
                        continue
                    for ek in espn_keys:
                        if ek not in key_index:
                            continue
                        raw = stats[key_index[ek]] if key_index[ek] < len(stats) else None
                        if ek in ("IP", "fullInnings.partInnings"):
                            value = _parse_ip_to_outs(str(raw)) if raw is not None else None
                        elif prop_type in ("3pm", "threes"):
                            value = _made_from_made_attempted(raw)
                        else:
                            value = _to_number(raw)
                        if value is None:
                            continue
                        # Don't overwrite (e.g. starters category before bench)
                        player_stats.setdefault(prop_type, value)
                        break
                if league.upper() in ("NBA", "WNBA"):
                    pts = player_stats.get("pts")
                    pts = player_stats.get("points") if pts is None else pts
                    reb = player_stats.get("reb")
                    reb = player_stats.get("rebounds") if reb is None else reb
                    ast = player_stats.get("ast")
                    ast = player_stats.get("assists") if ast is None else ast
                    if pts is not None and reb is not None and ast is not None:
                        player_stats.setdefault("pra", pts + reb + ast)
    return out


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------


def fetch_box_score(
    league: str,
    event_id: str,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Fetch the raw ESPN summary JSON for an event. Use :func:`parse_box_score`
    on the result to extract player stats."""
    assert_not_replay_mode("ESPN box score fetch")
    url = _summary_url(league, event_id)
    logger.debug("fetching ESPN box score: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def supported_prop_type(league: str, prop_type: str, category_hint: str = "") -> bool:
    """Return True if this league/prop_type pair is graded by the box-score parser."""
    league_upper = league.upper()
    if league_upper == "NBA":
        return prop_type.lower() in NBA_STAT_KEYS
    if league_upper == "WNBA":
        return prop_type.lower() in WNBA_STAT_KEYS
    if league_upper == "MLB":
        pt = prop_type.lower()
        return pt in MLB_BATTING_KEYS or pt in MLB_PITCHING_KEYS
    if league_upper in SOCCER_LEAGUE_SLUGS:
        return prop_type.lower() in SOCCER_STAT_KEYS
    return False
