"""Per-league Markov tuning constants.

These are the sport-specific baselines that the Markov simulation backends fall
back on when a caller omits team context. They are deliberately coarse — anchored
to recent league averages — so an analysis still produces a deterministic,
calibration-eligible trace when full context is unavailable. Real per-team
context, when supplied, always overrides these defaults.

Phase 7 introduces this module so sport backends can share one source of truth
for pace/efficiency constants instead of scattering magic numbers across the
engine. No simulation math lives here — only constants and small accessors.
"""

from __future__ import annotations

from typing import Any

# possessions_per_game_baseline is per-team possessions; the Markov loop sums
# home + away to get the total iteration count (see markov_engine
# _resolve_base_possessions). off/def efficiency are points per 100 possessions.
LEAGUE_BASELINES: dict[str, dict[str, Any]] = {
    "WNBA": {
        # Anchored to 2025 WNBA league averages.
        "possessions_per_game_baseline": 80.0,   # per team
        "off_efficiency_baseline": 100.0,        # pts / 100 poss
        "def_efficiency_baseline": 100.0,        # pts / 100 poss allowed
        # Forward-looking rate priors. Not yet consumed by the IID Markov PPP
        # model (which derives points purely from off/def rating); reserved for
        # a future shot-mix refinement. Documented here so the tuning target is
        # explicit rather than hidden in a backend.
        "three_point_rate": 0.33,
        "free_throw_rate": 0.20,
        "turnover_rate": 0.16,
    },
}


def get_league_baselines(league: str) -> dict[str, Any] | None:
    """Return the tuning-constant dict for *league*, or None if not defined."""
    return LEAGUE_BASELINES.get(league.upper())


def basketball_context_defaults(league: str) -> dict[str, float]:
    """Return a basketball team-context default dict from league baselines.

    Keys match the basketball archetype's ``required_team_keys``
    (``off_rating``, ``def_rating``, ``pace``) so the Markov backend can fill a
    missing context and pass key validation. Returns an empty dict when the
    league has no defined baselines (caller then fails closed as today).
    """
    baselines = get_league_baselines(league)
    if not baselines:
        return {}
    return {
        "off_rating": float(baselines["off_efficiency_baseline"]),
        "def_rating": float(baselines["def_efficiency_baseline"]),
        "pace": float(baselines["possessions_per_game_baseline"]),
    }
