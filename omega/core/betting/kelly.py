"""
Kelly Criterion staking module.

Computes fractional Kelly stakes with confidence-tier scaling to manage
bankroll risk across different confidence levels.
"""

from __future__ import annotations

from typing import Dict

from omega.core.betting.odds import american_to_decimal


# Fraction of full Kelly to use per confidence tier.
# Full Kelly is mathematically optimal but volatile;
# fractional Kelly trades expected growth for lower variance.
_TIER_MULTIPLIERS: Dict[str, float] = {
    "A": 0.50,   # High confidence: half Kelly
    "B": 0.25,   # Medium confidence: quarter Kelly
    "C": 0.10,   # Low confidence: tenth Kelly
}


def kelly_fraction(true_prob: float, odds: float) -> float:
    """Compute the raw (full) Kelly fraction for a bet.

    f* = (p * b - q) / b
    where p = true probability, q = 1 - p, b = net decimal payout.

    Returns 0.0 when the bet has negative or zero expected value.

    Args:
        true_prob: Model's estimated win probability (0-1).
        odds: American odds for the wager.

    Returns:
        float: Kelly fraction (0-1 range, clamped to 0 if negative EV).
    """
    decimal = american_to_decimal(odds)
    b = decimal - 1  # net payout per unit risked
    if b <= 0:
        return 0.0
    q = 1.0 - true_prob
    f = (true_prob * b - q) / b
    return max(0.0, f)


def recommend_stake(
    true_prob: float,
    odds: float,
    bankroll: float,
    confidence_tier: str = "B",
) -> Dict[str, float]:
    """Recommend a stake size using fractional Kelly.

    Args:
        true_prob: Model's estimated win probability (0-1).
        odds: American odds for the wager.
        bankroll: Current bankroll in dollars.
        confidence_tier: "A", "B", or "C" — scales the Kelly fraction.

    Returns:
        dict with:
            - "units": Recommended wager in bankroll units (1 unit = 1% of bankroll).
            - "kelly_fraction": The scaled Kelly fraction applied.
    """
    raw_kelly = kelly_fraction(true_prob, odds)
    tier_mult = _TIER_MULTIPLIERS.get(confidence_tier.upper(), _TIER_MULTIPLIERS["B"])
    scaled_kelly = raw_kelly * tier_mult

    # Convert to units (1 unit = 1% of bankroll), cap at 5 units
    units = min(scaled_kelly * 100, 5.0)

    return {
        "units": round(units, 2),
        "kelly_fraction": round(scaled_kelly, 4),
    }
