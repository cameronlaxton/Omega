"""
Sanity validator -- checks if extracted values fall within plausible ranges.

Catches obviously wrong data (e.g., offensive rating of 500, negative pace).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from omega.evidence.models import SportsFact

logger = logging.getLogger("omega.data.validators.sanity")


# Plausible ranges: (min, max) for common stat keys
_PLAUSIBLE_RANGES: Dict[str, Tuple[float, float]] = {
    # Basketball ratings
    "off_rating": (80.0, 135.0),
    "def_rating": (80.0, 135.0),
    "pace": (85.0, 115.0),
    # Basketball shooting (decimal form)
    "fg_pct": (0.0, 1.0),
    "three_pt_pct": (0.0, 1.0),
    "ft_pct": (0.0, 1.0),
    "ts_pct": (0.0, 1.0),
    "efg_pct": (0.0, 1.0),
    # Basketball per-game
    "pts_per_game": (0.0, 150.0),
    "reb_per_game": (0.0, 70.0),
    "ast_per_game": (0.0, 40.0),
    "tov_per_game": (0.0, 30.0),
    # Player averages
    "pts_mean": (0.0, 60.0),
    "reb_mean": (0.0, 25.0),
    "ast_mean": (0.0, 20.0),
    "min_mean": (0.0, 48.0),
    "pts_std": (0.0, 20.0),
    # Football
    "yds_per_game": (0.0, 600.0),
    "pass_yds_per_game": (0.0, 500.0),
    "rush_yds_per_game": (0.0, 300.0),
    # Baseball
    "era": (0.0, 15.0),
    "batting_avg": (0.0, 1.0),
    "ops": (0.0, 2.0),
    "runs_per_game": (0.0, 20.0),
    # Hockey
    "goals_per_game": (0.0, 10.0),
    "goals_against_per_game": (0.0, 10.0),
    "save_pct": (0.0, 1.0),
    "power_play_pct": (0.0, 1.0),
    "penalty_kill_pct": (0.0, 1.0),
    # Odds
    "moneyline_home": (-10000.0, 10000.0),
    "moneyline_away": (-10000.0, 10000.0),
    "spread_home": (-60.0, 60.0),
    "spread_away": (-60.0, 60.0),
    "total": (0.0, 400.0),
    # Record
    "wins": (0.0, 200.0),
    "losses": (0.0, 200.0),
    "games_played": (0.0, 200.0),
}


def validate_sanity(facts: List[SportsFact]) -> List[SportsFact]:
    """Filter out facts with values outside plausible ranges.

    Args:
        facts: List of facts to validate.

    Returns:
        List of facts that pass sanity checks.
    """
    valid: List[SportsFact] = []

    for fact in facts:
        if _is_sane(fact.key, fact.value):
            valid.append(fact)
        else:
            logger.debug(
                "Sanity check failed: key=%s value=%s entity=%s",
                fact.key, fact.value, fact.entity,
            )

    return valid


def _is_sane(key: str, value: Any) -> bool:
    """Check if a value falls within plausible range for its key."""
    if value is None:
        return True  # None values are allowed (missing data)

    bounds = _PLAUSIBLE_RANGES.get(key)
    if bounds is None:
        return True  # No range defined -- pass through

    try:
        numeric = float(value)
        return bounds[0] <= numeric <= bounds[1]
    except (ValueError, TypeError):
        return True  # Non-numeric values aren't range-checked
