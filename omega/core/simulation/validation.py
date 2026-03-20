"""
Simulation input validation — type-checks, coerces, and bounds-checks
context dicts before they enter the Monte Carlo engine.

Drops invalid values (letting the engine fall through to archetype defaults)
rather than raising, so partial data still produces usable results.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Set, Tuple

from omega.core.simulation.archetypes import get_archetype

logger = logging.getLogger("omega.core.simulation.validation")


# ---------------------------------------------------------------------------
# Sanity bounds per key — (min, max) inclusive
# Keys without explicit bounds get a generic finite-number check only.
# ---------------------------------------------------------------------------

SIM_INPUT_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Basketball
    "off_rating": (80.0, 140.0),
    "def_rating": (80.0, 140.0),
    "pace": (85.0, 115.0),
    "fg_pct": (0.30, 0.65),
    "three_pt_pct": (0.20, 0.55),
    "ft_pct": (0.40, 1.0),
    "turnover_rate": (0.05, 0.30),
    "off_reb_pct": (0.15, 0.45),
    "home_court_adj": (-5.0, 10.0),
    # American Football
    "pass_eff": (0.0, 15.0),
    "rush_eff": (0.0, 10.0),
    "turnover_diff": (-3.0, 3.0),
    "red_zone_pct": (0.30, 0.90),
    "third_down_pct": (0.20, 0.65),
    # Baseball
    "era": (0.5, 10.0),
    "batting_avg": (0.100, 0.400),
    "obp": (0.200, 0.500),
    "slg": (0.250, 0.700),
    "whip": (0.70, 2.50),
    "bullpen_era": (0.5, 10.0),
    "park_factor": (0.80, 1.20),
    # Hockey
    "shots_per_game": (20.0, 45.0),
    "save_pct": (0.850, 0.970),
    "pp_pct": (0.05, 0.40),
    "pk_pct": (0.60, 1.0),
    "xgf_per_60": (1.0, 5.0),
    "xga_per_60": (1.0, 5.0),
    "goalie_sv_pct": (0.850, 0.970),
    "goalie_gsax": (-30.0, 40.0),
    # Soccer
    "xg_for": (0.3, 4.0),
    "xg_against": (0.3, 4.0),
    "possession_pct": (25.0, 75.0),
    "shots_on_target": (1.0, 15.0),
    "corners_per_game": (1.0, 15.0),
    # Tennis
    "serve_win_pct": (0.40, 0.90),
    "return_win_pct": (0.20, 0.65),
    "ace_rate": (0.0, 0.40),
    "double_fault_rate": (0.0, 0.20),
    "first_serve_pct": (0.40, 0.85),
    "break_point_conversion": (0.20, 0.70),
    "surface_adj": (-0.10, 0.10),
    "fatigue_factor": (0.0, 1.0),
    # Golf
    "strokes_gained_total": (-3.0, 5.0),
    "sg_off_tee": (-2.0, 3.0),
    "sg_approach": (-2.0, 3.0),
    "sg_around_green": (-2.0, 3.0),
    "sg_putting": (-2.0, 3.0),
    "course_fit": (0.0, 1.0),
    "recent_form": (-3.0, 5.0),
    "gir_pct": (0.40, 0.90),
    # Fighting
    "win_pct": (0.0, 1.0),
    "finish_rate": (0.0, 1.0),
    "ko_tko_rate": (0.0, 1.0),
    "submission_rate": (0.0, 1.0),
    "decision_rate": (0.0, 1.0),
    "sig_strikes_per_min": (0.0, 15.0),
    "sig_strike_accuracy": (0.0, 1.0),
    "takedown_avg": (0.0, 10.0),
    "takedown_defense": (0.0, 1.0),
    "activity_rate": (0.0, 1.0),
    # Esports
    "map_win_rate": (0.0, 1.0),
    "avg_round_diff": (-15.0, 15.0),
    "first_blood_rate": (0.0, 1.0),
    "roster_stability": (0.0, 1.0),
    # Shared / generic
    "elo_rating": (500.0, 3500.0),
}


def _coerce_numeric(value: Any) -> Optional[float]:
    """Try to coerce a value to float. Returns None if impossible."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return None


def validate_sim_context(
    context: Optional[Dict[str, Any]],
    league: str,
    side: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate and sanitize a simulation context dict.

    - Keeps only keys recognized by the archetype
    - Coerces string-numerics to float
    - Drops values that are non-numeric, NaN, Inf, or out of sanity bounds
    - Returns a clean dict (never mutates the input)

    When strict=True:
    - Collects all violations and raises ValueError listing them all
    - Requires at least 2 valid keys to survive (prevents near-empty contexts)

    Args:
        context: Raw context dict (may be None or empty).
        league: League code (e.g. "NBA", "NFL").
        side: "home" or "away" (for logging).
        strict: When True, raises ValueError on any invalid data instead of dropping.

    Returns:
        Sanitized dict with only valid numeric values for known keys.

    Raises:
        ValueError: In strict mode, if any values are invalid or insufficient data remains.
    """
    if not context:
        if strict:
            raise ValueError(f"{side}: no context data provided")
        return {}

    archetype = get_archetype(league)
    if archetype is None:
        logger.warning("No archetype for league %s — passing context through", league)
        return dict(context)

    known_keys: Set[str] = set(
        archetype.critical_team_keys
        + archetype.required_team_keys
        + archetype.optional_team_keys
    )

    cleaned: Dict[str, Any] = {}
    violations: list[str] = []

    for key, value in context.items():
        if key not in known_keys:
            continue

        numeric = _coerce_numeric(value)
        if numeric is None:
            msg = f"{side}.{key}: non-numeric value {value!r}"
            if strict:
                violations.append(msg)
            else:
                logger.warning("Dropped %s", msg)
            continue

        if math.isnan(numeric) or math.isinf(numeric):
            msg = f"{side}.{key}: NaN/Inf value"
            if strict:
                violations.append(msg)
            else:
                logger.warning("Dropped %s", msg)
            continue

        bounds = SIM_INPUT_BOUNDS.get(key)
        if bounds is not None:
            lo, hi = bounds
            if numeric < lo or numeric > hi:
                msg = f"{side}.{key}: value {numeric:.4f} outside bounds [{lo:.4f}, {hi:.4f}]"
                if strict:
                    violations.append(msg)
                else:
                    logger.warning("Dropped %s", msg)
                continue

        cleaned[key] = numeric

    if strict:
        if violations:
            raise ValueError(
                f"Strict validation failed for {side} ({len(violations)} violation(s)): "
                + "; ".join(violations)
            )
        if len(cleaned) < 2:
            raise ValueError(
                f"Strict validation: insufficient valid data for {side} "
                f"(only {len(cleaned)} key(s) survived, minimum 2 required)"
            )

    return cleaned
