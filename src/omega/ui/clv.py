"""Read-only closing-line-value (CLV) helper for the operator console (B.3).

CLV is a *market/settlement* metric — how the price you took compares to the
market's closing price — not recommendation math. It is computed here (never in
templates) by reusing the implied-probability arithmetic from
:mod:`omega.ui.normalizers`. The helper is pure: no I/O, no mutation.

The probability terms are *raw implied* (they include the book's vig); this is the
simple, honest console metric and is labelled as such in the UI. Positive
``clv_points`` means you took a longer price than the close (you "beat the close").
"""

from __future__ import annotations

from dataclasses import dataclass

from omega.ui.normalizers import implied_probability_from_american

__all__ = ["ClvResult", "closing_line_value"]


@dataclass(frozen=True)
class ClvResult:
    """Closing-line value for one taken-vs-closing American-odds pair."""

    taken_implied: float | None
    closing_implied: float | None
    clv_points: float | None  # closing_implied - taken_implied (prob points; >0 = beat close)
    beat_close: bool | None


def closing_line_value(taken_odds: object, closing_odds: object) -> ClvResult:
    """CLV from a taken price vs the closing price (both American odds).

    Returns all-``None`` derived fields when either odds value is missing or not
    valid American odds — no guessing.
    """
    taken_implied = implied_probability_from_american(taken_odds)
    closing_implied = implied_probability_from_american(closing_odds)
    if taken_implied is None or closing_implied is None:
        return ClvResult(taken_implied, closing_implied, None, None)
    points = closing_implied - taken_implied
    return ClvResult(
        taken_implied=taken_implied,
        closing_implied=closing_implied,
        clv_points=points,
        beat_close=points > 0,
    )
