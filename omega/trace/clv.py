"""
omega.trace.clv — Closing Line Value math.

CLV is the difference between the price/line you took at decision time and the
market's price/line at close. Beating the close is a leading indicator of EV
that resolves within hours of game start, far faster than realized ROI.

This module owns the math only. Persistence lives in `omega.trace.store`
(`attach_closing_line`, `get_closing_lines`); fetching lives in
`omega.integrations.odds_api`.

Conventions:
- American odds in / decimal odds out (when computing CLV in basis points).
- "Beat the close" means odds_taken implies a worse (longer-priced) implied
  probability than closing_odds — i.e., you got a better price than the market
  ultimately settled at. We express CLV in implied-probability cents (negative
  is bad, positive is good).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds (e.g. -110, +135) to decimal odds (e.g. 1.909, 2.35).

    Decimal odds are always > 1.0. Handles -100 / +100 as 2.0 (true even money).
    """
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    if american_odds < 0:
        return 1.0 + (100.0 / abs(american_odds))
    raise ValueError("American odds cannot be zero")


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability (with vig)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100.0)
    raise ValueError("American odds cannot be zero")


@dataclass(frozen=True)
class CLVResult:
    """Output of compute_clv()."""
    odds_taken: float
    closing_odds: float
    implied_taken: float            # 0..1
    implied_closing: float          # 0..1
    clv_cents: float                # implied_closing - implied_taken, in implied-prob percentage points
    decimal_taken: float
    decimal_closing: float
    decimal_gain: float             # decimal_taken - decimal_closing (positive = beat close)
    line_taken: Optional[float]
    closing_line: Optional[float]
    line_value: Optional[float]     # for spread/total, signed line movement in your favor
    beat_close: bool                # True iff implied_taken < implied_closing


def compute_clv(
    odds_taken: float,
    closing_odds: float,
    line_taken: Optional[float] = None,
    closing_line: Optional[float] = None,
    side: Optional[str] = None,
) -> CLVResult:
    """Compute CLV for a single bet vs its closing snapshot.

    Args:
        odds_taken: American odds the user took.
        closing_odds: American odds at close, same selection.
        line_taken: Point/total at which the user bet (None for moneyline).
        closing_line: Point/total at close (None for moneyline).
        side: For spread/total, "over"/"under"/"home"/"away" — used to sign the
              line movement. None disables `line_value` computation.

    Returns:
        CLVResult.
    """
    implied_taken = american_to_implied_prob(odds_taken)
    implied_closing = american_to_implied_prob(closing_odds)
    decimal_taken = american_to_decimal(odds_taken)
    decimal_closing = american_to_decimal(closing_odds)

    line_value: Optional[float] = None
    if line_taken is not None and closing_line is not None and side:
        # Positive line_value = the line moved in your favor.
        delta = closing_line - line_taken  # raw movement
        s = side.lower()
        if s in ("over", "home"):
            # For Over: lower line at close is favorable (you took the higher number).
            line_value = -delta
        elif s in ("under", "away"):
            # For Under: higher line at close is favorable.
            line_value = delta
        else:
            line_value = None

    # CLV in percentage points of implied probability
    clv_cents = (implied_closing - implied_taken) * 100.0

    return CLVResult(
        odds_taken=odds_taken,
        closing_odds=closing_odds,
        implied_taken=implied_taken,
        implied_closing=implied_closing,
        clv_cents=clv_cents,
        decimal_taken=decimal_taken,
        decimal_closing=decimal_closing,
        decimal_gain=decimal_taken - decimal_closing,
        line_taken=line_taken,
        closing_line=closing_line,
        line_value=line_value,
        beat_close=implied_taken < implied_closing,
    )
