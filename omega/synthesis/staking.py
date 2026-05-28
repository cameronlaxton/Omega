"""Centralized staking policy for Omega sessions.

Provides consistent unit/dollar math so that stake_units and stake_dollars
cannot contradict. Bankroll is fixed at $1000 until automated bankroll
tracking is implemented.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_BANKROLL: float = 1000.0
DEFAULT_UNIT_PCT: float = 0.01  # 1 unit = 1% of bankroll


def unit_to_dollars(
    units: float,
    bankroll: float = DEFAULT_BANKROLL,
    unit_pct: float = DEFAULT_UNIT_PCT,
) -> float:
    """Convert units to dollars. 1 unit = bankroll * unit_pct."""
    return round(units * bankroll * unit_pct, 2)


def dollars_to_units(
    dollars: float,
    bankroll: float = DEFAULT_BANKROLL,
    unit_pct: float = DEFAULT_UNIT_PCT,
) -> float:
    """Convert dollars to units. Inverse of unit_to_dollars."""
    base = bankroll * unit_pct
    if base <= 0:
        raise ValueError(f"bankroll * unit_pct must be positive, got {base}")
    return round(dollars / base, 4)


def calculate_stake(
    bankroll: float,
    unit_pct: float,
    is_static_fallback: bool,
) -> tuple[float, float]:
    """Derive stake sizing. Returns (stake_units, stake_dollars).

    Args:
        bankroll: Current bankroll in dollars.
        unit_pct: Fraction of bankroll per unit (e.g. 0.01 = 1%).
        is_static_fallback: True when calibration is the uncalibrated static prior.

    Returns:
        (stake_units, stake_dollars) — consistent pair, never contradicting.
        Caps at 1u when is_static_fallback is True.
    """
    if is_static_fallback:
        logger.warning("Static fallback active; capping stake at 1.0u.")
        stake_units = 1.0
    else:
        stake_units = 1.0  # default; Kelly sizing is applied by the engine

    stake_dollars = unit_to_dollars(stake_units, bankroll, unit_pct)
    return stake_units, stake_dollars
