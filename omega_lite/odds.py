"""
Odds evaluation utilities.

Pure math functions for converting odds formats, computing implied probabilities,
edge percentages, and expected value.
"""

from __future__ import annotations


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds.

    Examples:
        -150 -> 1.667  (risk 150 to win 100)
        +130 -> 2.300  (risk 100 to win 130)
    """
    if american_odds >= 100:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def implied_probability(odds: float) -> float:
    """Derive implied probability from American odds.

    Removes the vig/juice conceptually — this is raw implied probability
    from the line, not a no-vig fair probability.

    Returns:
        float: Probability between 0 and 1.
    """
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def edge_percentage(true_prob: float, market_prob: float) -> float:
    """Compute edge as percentage points: (true - market) * 100.

    A positive value means the model sees more value than the market.

    Args:
        true_prob: Model's estimated probability (0-1).
        market_prob: Market-implied probability (0-1).

    Returns:
        float: Edge in percentage points (e.g., 5.2 means 5.2% edge).
    """
    return (true_prob - market_prob) * 100.0


def expected_value_percent(prob: float, odds: float) -> float:
    """Calculate expected value as a percentage of the stake.

    EV% = (prob * decimal_payout) - 1, expressed as a percentage.

    Args:
        prob: True probability of winning (0-1).
        odds: American odds for the wager.

    Returns:
        float: EV as a percentage (e.g., 8.5 means +8.5% EV).
    """
    decimal = american_to_decimal(odds)
    return (prob * decimal - 1) * 100.0
