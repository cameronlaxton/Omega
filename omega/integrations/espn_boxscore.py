"""
omega.integrations.espn_boxscore — ESPN public summary/box-score for player stats.

Used by scripts/fetch_outcomes_props.py to resolve a prop trace's player stat
against the actual game line. We separate the HTTP fetch from the JSON parse
so tests can hit `parse_box_score()` with a fixture instead of the network.

ESPN summary endpoints:
  NBA: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<event_id>
  MLB: https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event=<event_id>

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

logger = logging.getLogger("omega.integrations.espn_boxscore")

_REQUEST_TIMEOUT_SECONDS = 15
_SUMMARY_URLS = {
    "NBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
    "WNBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary",
    "MLB": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary",
}


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
    if league.upper() == "NBA":
        return NBA_STAT_KEYS
    if league.upper() == "WNBA":
        return WNBA_STAT_KEYS
    if league.upper() == "MLB":
        cat = (category_name or "").lower()
        key_set = {str(key) for key in (keys or [])}
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
    return {}


def _parse_ip_to_outs(value: str) -> float | None:
    """Convert MLB innings-pitched string like '6.2' (6 innings, 2 outs) to outs."""
    try:
        whole_str, _, frac_str = str(value).partition(".")
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


def parse_box_score(
    payload: dict[str, Any],
    league: str,
) -> dict[str, dict[str, float]]:
    """Parse an ESPN summary payload into ``{player_norm → {stat_type → value}}``.

    Player names are normalized via :func:`normalize_player_name`. Stat types
    use the omega prop_type vocabulary (NBA_STAT_KEYS / MLB_BATTING_KEYS /
    MLB_PITCHING_KEYS). Unknown categories are skipped, unmapped stat types
    are skipped (logged at DEBUG).
    """
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
                if league.upper() == "NBA":
                    pts = player_stats.get("pts") or player_stats.get("points")
                    reb = player_stats.get("reb") or player_stats.get("rebounds")
                    ast = player_stats.get("ast") or player_stats.get("assists")
                    if pts is not None and reb is not None and ast is not None:
                        player_stats.setdefault("pra", float(pts) + float(reb) + float(ast))
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
    url_base = _SUMMARY_URLS.get(league.upper())
    if not url_base:
        raise ValueError(f"No ESPN summary endpoint configured for league={league!r}")
    url = f"{url_base}?{urllib.parse.urlencode({'event': event_id})}"
    logger.debug("fetching ESPN box score: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def supported_prop_type(league: str, prop_type: str, category_hint: str = "") -> bool:
    """Return True if this league/prop_type pair is graded by the box-score parser."""
    if league.upper() == "NBA":
        return prop_type.lower() in NBA_STAT_KEYS
    if league.upper() == "MLB":
        pt = prop_type.lower()
        # Pitching stats and batting stats both live under MLB
        return pt in MLB_BATTING_KEYS or pt in MLB_PITCHING_KEYS
    return False
