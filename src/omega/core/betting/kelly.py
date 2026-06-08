"""
Kelly Criterion staking module.

Computes fractional Kelly stakes with confidence-tier scaling to manage
bankroll risk across different confidence levels.
"""

from __future__ import annotations

from omega.core.betting.odds import american_to_decimal


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
) -> dict[str, float]:
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

    This delegates to the default staking policy
    (:class:`omega.core.betting.staking_policy.FractionalKellyByTier`) so that all
    sizing flows through one shared path. The numeric result is unchanged.
    """
    # Imported lazily to avoid an import cycle: staking_policy imports
    # ``kelly_fraction`` from this module at import time.
    from omega.core.betting.staking_policy import FractionalKellyByTier, StakingContext

    decision = FractionalKellyByTier().size(
        StakingContext(
            true_prob=true_prob,
            odds=odds,
            bankroll=bankroll,
            confidence_tier=confidence_tier,
        )
    )
    return decision.to_recommend_stake_dict()
