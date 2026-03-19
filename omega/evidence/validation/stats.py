"""
Stat normalizer -- normalize statistical values to consistent formats.

Handles common issues:
- Percentages expressed as 0-100 vs 0-1
- String values that should be numeric
- Per-game vs total stats
"""

from __future__ import annotations

from typing import Any


# Percentage fields that should be in decimal form (0.0-1.0)
_PCT_KEYS = {
    "fg_pct", "three_pt_pct", "ft_pct", "ts_pct",
    "efg_pct", "batting_avg", "save_pct",
    "power_play_pct", "penalty_kill_pct",
}


def normalize_stat_value(key: str, value: Any, league: str = "") -> Any:
    """Normalize a stat value based on its key and context.

    Args:
        key: The stat key (e.g., "fg_pct", "pts_per_game").
        value: The raw value to normalize.
        league: Optional league context.

    Returns:
        Normalized value.
    """
    if value is None:
        return None

    # Convert string to number if possible
    if isinstance(value, str):
        value = value.strip().rstrip("%")
        try:
            value = float(value)
        except ValueError:
            return value

    # Normalize percentage fields to decimal form
    if key in _PCT_KEYS and isinstance(value, (int, float)):
        if value > 1.0:
            value = value / 100.0

    return value
