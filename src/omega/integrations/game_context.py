"""
omega.integrations.game_context — resolve the situational "context pack" for a matchup.

This is an input-prep tool, in the same spirit as ``odds_resolver``: it prepares
deterministic situational facts and a structured evidence worksheet for the LLM.
It does NOT compute any protected Omega output (probability, edge, EV, Kelly,
units, tiers, trace IDs).

Three layers, with honest graceful degradation:

  Layer 1 — deterministic ``game_context`` keys the engine consumes:
      rest_days / home_rest_days / away_rest_days / is_b2b_* (schedule math via the
      free ESPN scoreboard, Odds API ``/scores`` fallback), is_playoff (per-league
      date heuristic), park_factor (static MLB table). Anything unresolvable lands
      in ``needs_manual`` rather than being fabricated.

  Layer 2 — ``applicable_evidence``: the EvidenceSignal signal_types that apply to
      this league/sport (from SIGNAL_REGISTRY via signal_applies_to_league), with a
      ``markov_eligible`` flag. Tells the LLM exactly which matchup/situational/
      team_form/player_form signals to assess and emit.

  Layer 3 — ``suggested_evidence``: semantic context with no wired data source
      (rivalry, opponent defensive rank, seeding/motivation, vs-former-team). These
      are surfaced as recommended signal_types (rivalry/revenge reuse
      ``motivation_edge``) for the LLM to fill — never auto-fabricated.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from omega.core.contracts.evidence import SIGNAL_REGISTRY, signal_applies_to_league
from omega.core.simulation.evidence_to_modifier import MAPPED_SIGNAL_TYPES
from omega.integrations import espn_mlb, espn_nba, espn_wnba
from omega.integrations._guards import assert_not_replay_mode

logger = logging.getLogger("omega.integrations.game_context")

# League -> ESPN scoreboard module (free, by-date). Leagues absent here fall back
# to the Odds API /scores endpoint for schedule facts.
_ESPN_SCOREBOARD = {
    "NBA": espn_nba,
    "WNBA": espn_wnba,
    "MLB": espn_mlb,
}

# Vintage of the hand-maintained static tables below (playoff windows + park
# factors). Surfaced in provenance so a consumer can judge staleness rather than
# trusting an undated approximation; bump when the tables are reviewed/refreshed.
STATIC_DATA_VINTAGE = "2026"

# Per-league playoff date windows (inclusive), (start_month, start_day)..(end). A
# coarse heuristic — flagged as such in provenance; unknown leagues -> needs_manual.
_PLAYOFF_WINDOWS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "NBA": ((4, 15), (6, 30)),
    "WNBA": ((9, 14), (10, 31)),
    "NHL": ((4, 15), (6, 30)),
    "MLB": ((10, 1), (11, 15)),
    "NFL": ((1, 10), (2, 15)),
}

# Static, approximate MLB park factors keyed by canonical home-team name (>1 =
# hitter-friendly). Source: widely-cited multi-year averages; refresh as needed.
MLB_PARK_FACTORS: dict[str, float] = {
    "Colorado Rockies": 1.15,
    "Boston Red Sox": 1.08,
    "Cincinnati Reds": 1.07,
    "Kansas City Royals": 1.05,
    "Arizona Diamondbacks": 1.04,
    "Baltimore Orioles": 1.03,
    "Philadelphia Phillies": 1.03,
    "Texas Rangers": 1.02,
    "Toronto Blue Jays": 1.02,
    "Atlanta Braves": 1.01,
    "Chicago Cubs": 1.01,
    "Los Angeles Angels": 1.00,
    "Minnesota Twins": 1.00,
    "Washington Nationals": 1.00,
    "Houston Astros": 1.00,
    "Milwaukee Brewers": 1.00,
    "New York Yankees": 1.00,
    "Pittsburgh Pirates": 0.99,
    "Chicago White Sox": 0.99,
    "St. Louis Cardinals": 0.98,
    "New York Mets": 0.98,
    "Los Angeles Dodgers": 0.98,
    "Cleveland Guardians": 0.97,
    "Tampa Bay Rays": 0.97,
    "Detroit Tigers": 0.97,
    "Athletics": 0.96,
    "San Francisco Giants": 0.95,
    "Seattle Mariners": 0.94,
    "Miami Marlins": 0.94,
    "San Diego Padres": 0.94,
}

# Static rivalry pairs (canonical names) per league. Matched order-independently;
# a hit suggests a motivation_edge signal (magnitude left to the LLM).
_RIVALRIES: dict[str, list[frozenset[str]]] = {
    "NBA": [
        frozenset({"Los Angeles Lakers", "Boston Celtics"}),
        frozenset({"Golden State Warriors", "Cleveland Cavaliers"}),
        frozenset({"Chicago Bulls", "Detroit Pistons"}),
        frozenset({"Miami Heat", "New York Knicks"}),
        frozenset({"Los Angeles Lakers", "LA Clippers"}),
    ],
    "MLB": [
        frozenset({"New York Yankees", "Boston Red Sox"}),
        frozenset({"Los Angeles Dodgers", "San Francisco Giants"}),
        frozenset({"Chicago Cubs", "St. Louis Cardinals"}),
        frozenset({"New York Mets", "Philadelphia Phillies"}),
    ],
}

# Semantic context dimensions we cannot auto-resolve (no wired standings/roster/
# transaction source). Surfaced as suggested evidence for the LLM to assess.
_UNWIRED_SEMANTIC: tuple[tuple[str, str], ...] = (
    ("opponent_def_rank", "opponent_stat_rank"),
    ("seeding_motivation", "motivation_edge"),
    ("vs_former_team", "motivation_edge"),
)

ScoreboardFn = Callable[[str, str], list]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _canonical(league: str, name: str) -> str:
    """Canonicalize a team name via the league's ESPN module, lowercased."""
    mod = _ESPN_SCOREBOARD.get(league.upper())
    if mod is not None:
        resolved = mod.canonical_team(name)
        if resolved:
            return resolved.lower()
    return (name or "").strip().lower()


def _default_scoreboard_fn(league: str, date_str: str) -> list:
    """Fetch one date's final games from the league's ESPN scoreboard, or []."""
    mod = _ESPN_SCOREBOARD.get(league.upper())
    if mod is None:
        return []
    return mod.fetch_scoreboard(date_str)


def _is_final(game: Any) -> bool:
    return "final" in str(getattr(game, "status", "") or "").lower()


def _game_involves(game: Any, team_lower: str) -> bool:
    home = str(getattr(game, "home_team", "") or "").lower()
    away = str(getattr(game, "away_team", "") or "").lower()
    return team_lower in (home, away)


def _last_game_date(
    league: str,
    team: str,
    game_date: str,
    lookback_days: int,
    scoreboard_fn: ScoreboardFn,
    odds_client: Any | None,
) -> tuple[str | None, str | None]:
    """Most recent completed game date for ``team`` before ``game_date``.

    Returns ``(iso_date, source)``; ``(None, None)`` if nothing is found within
    the lookback window. Prefers the (free) scoreboard source, falling back to
    the Odds API /scores endpoint.
    """
    target = _parse_date(game_date)
    if target is None:
        return None, None

    # 1. Scoreboard source (ESPN by default; an injected fake in tests). Scan
    # backward day-by-day; the first day with a completed game is the most recent.
    team_c = _canonical(league, team)
    for delta in range(1, lookback_days + 1):
        day = (target - timedelta(days=delta)).isoformat()
        try:
            games = scoreboard_fn(league, day)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the pack
            logger.warning("scoreboard fetch failed for %s %s: %s", league, day, exc)
            games = []
        if any(_is_final(g) and _game_involves(g, team_c) for g in games):
            return day, "scoreboard"

    # 2. Odds API /scores fallback (leagues without an ESPN module).
    scores = _safe_fetch_scores(league, lookback_days, odds_client)
    best: date | None = None
    team_l = (team or "").strip().lower()
    for s in scores:
        if not getattr(s, "completed", False):
            continue
        names = {
            str(getattr(s, "home_team", "")).lower(),
            str(getattr(s, "away_team", "")).lower(),
        }
        if team_l not in names:
            continue
        sd = _parse_date(getattr(s, "commence_time", ""))
        if sd and sd < target and (best is None or sd > best):
            best = sd
    if best is not None:
        return best.isoformat(), "odds_api_scores"
    return None, None


def _safe_fetch_scores(league: str, lookback_days: int, odds_client: Any | None) -> list:
    client = odds_client
    if client is None:
        try:
            from omega.integrations.odds_api import OddsApiClient

            client = OddsApiClient()
        except Exception as exc:  # noqa: BLE001
            logger.debug("odds /scores fallback unavailable: %s", exc)
            return []
    try:
        return client.fetch_scores(league, days_from=min(lookback_days, 3))
    except Exception as exc:  # noqa: BLE001
        logger.debug("odds /scores fetch failed for %s: %s", league, exc)
        return []


def _rest_days_between(last_iso: str | None, game_date: str) -> int | None:
    """Days of rest before ``game_date`` given the previous game date.

    The engine convention is ``rest_days == 0`` means the team played the
    previous night (back-to-back), so rest = calendar delta minus one, floored
    at zero (a same-day or future stray record never yields a negative).
    """
    last = _parse_date(last_iso)
    target = _parse_date(game_date)
    if last is None or target is None:
        return None
    return max((target - last).days - 1, 0)


def _is_playoff(league: str, game_date: str) -> bool | None:
    window = _PLAYOFF_WINDOWS.get(league.upper())
    d = _parse_date(game_date)
    if window is None or d is None:
        return None
    (sm, sd), (em, ed) = window
    start = date(d.year, sm, sd)
    end = date(d.year, em, ed)
    return start <= d <= end


def _mlb_park_factor(home_team: str) -> float | None:
    canonical = _canonical("MLB", home_team)
    for name, factor in MLB_PARK_FACTORS.items():
        if name.lower() == canonical:
            return factor
    return None


def _is_rivalry(league: str, home_team: str, away_team: str) -> bool:
    pairs = _RIVALRIES.get(league.upper())
    if not pairs:
        return False
    matchup = {_canonical(league, home_team), _canonical(league, away_team)}
    return any({n.lower() for n in pair} == matchup for pair in pairs)


def _applicable_evidence(league: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for signal_type, spec in SIGNAL_REGISTRY.items():
        if not signal_applies_to_league(signal_type, league):
            continue
        out.append(
            {
                "signal_type": signal_type,
                "category": spec.category,
                "plane": spec.plane,
                "default_window": spec.default_window,
                "markov_eligible": signal_type in MAPPED_SIGNAL_TYPES,
                "description": spec.description,
            }
        )
    out.sort(key=lambda e: (e["category"], e["signal_type"]))
    return out


def resolve_game_context(
    league: str,
    home_team: str,
    away_team: str,
    game_date: str,
    *,
    lookback_days: int = 5,
    scoreboard_fn: ScoreboardFn | None = None,
    odds_client: Any | None = None,
) -> dict[str, Any]:
    """Resolve the situational context pack for a matchup. See module docstring."""
    assert_not_replay_mode("game context resolver")
    league_u = league.upper()
    sb = scoreboard_fn or _default_scoreboard_fn
    needs_manual: list[str] = []
    provenance: dict[str, Any] = {}
    game_context: dict[str, Any] = {}

    # Layer 1 — rest days (schedule math).
    home_last, home_src = _last_game_date(
        league_u, home_team, game_date, lookback_days, sb, odds_client
    )
    away_last, away_src = _last_game_date(
        league_u, away_team, game_date, lookback_days, sb, odds_client
    )
    home_rest = _rest_days_between(home_last, game_date)
    away_rest = _rest_days_between(away_last, game_date)

    if home_rest is not None:
        game_context["home_rest_days"] = home_rest
        game_context["is_b2b_home"] = home_rest == 0
        game_context["rest_days"] = home_rest  # home-team reference for the engine
    else:
        needs_manual.append("home_rest_days")
    if away_rest is not None:
        game_context["away_rest_days"] = away_rest
        game_context["is_b2b_away"] = away_rest == 0
    else:
        needs_manual.append("away_rest_days")
    provenance["rest_days_source"] = home_src or away_src

    # Layer 1 — playoff phase (date heuristic).
    playoff = _is_playoff(league_u, game_date)
    if playoff is None:
        needs_manual.append("is_playoff")
        provenance["is_playoff_source"] = None
    else:
        game_context["is_playoff"] = playoff
        provenance["is_playoff_source"] = f"date_heuristic:{STATIC_DATA_VINTAGE}"

    # Layer 1 — MLB park factor (static table).
    if league_u == "MLB":
        park = _mlb_park_factor(home_team)
        if park is not None:
            game_context["park_factor"] = park
            provenance["park_factor_source"] = f"static_approximate:{STATIC_DATA_VINTAGE}"
        else:
            needs_manual.append("park_factor")

    # Layer 2 — evidence worksheet.
    applicable_evidence = _applicable_evidence(league_u)

    # Layer 3 — semantic suggestions (no wired source -> LLM fills the value).
    suggested_evidence: list[dict[str, Any]] = []
    if _is_rivalry(league_u, home_team, away_team):
        suggested_evidence.append(
            {
                "signal_type": "motivation_edge",
                "reason": f"rivalry: {away_team} @ {home_team}",
            }
        )
    for field, signal_type in _UNWIRED_SEMANTIC:
        suggested_evidence.append(
            {
                "signal_type": signal_type,
                "reason": f"{field}: no_wired_source — assess from your own reasoning",
            }
        )

    return {
        "status": "success",
        "league": league_u,
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date,
        "game_context": game_context,
        "provenance": provenance,
        "applicable_evidence": applicable_evidence,
        "suggested_evidence": suggested_evidence,
        "needs_manual": needs_manual,
    }
