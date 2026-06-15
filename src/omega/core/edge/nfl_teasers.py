"""NFL Wong-teaser leg evaluation from the discrete margin / total pmf.

Phase 7 Milestone 4. A teaser moves a spread or total by a fixed number of points
in the bettor's favor; each leg is then an ordinary threshold bet at the moved
line. "Wong teasers" are the well-known 6-point legs that cross both key NFL
numbers (3 and 7): favorites of -7.5..-8.5 teased to -1.5..-2.5, and underdogs of
+1.5..+2.5 teased to +7.5..+8.5. Crossing 3 and 7 is where the non-linear value
sits, because NFL margins cluster on those numbers.

This module computes one leg's cover distribution from the backend's discrete
margin pmf (``margin_counts``) or total pmf (``total_counts``) and reuses the
sport-neutral threshold evaluator + EV bridge in ``omega.core.edge.line_markets``
— a teaser leg is the same primitive as a soccer Asian-handicap line. The NFL
edge consumer (``omega/core/edge/nfl_consumer.py``) turns these into EdgeDetail
rows; per-leg EVs combine multiplicatively into a full teaser-card price, which
is a staking-layer concern handled outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping

from omega.core.edge.line_markets import LineBetEvaluation, evaluate_threshold_bet

# Standard 2-team Wong teaser shift, in points.
WONG_TEASER_POINTS = 6.0


def tease_spread(leg_line: float, points: float = WONG_TEASER_POINTS) -> float:
    """Move a side's own spread *leg_line* by *points* in the bettor's favor.

    A spread leg always teases toward the bettor — the covered region widens — so
    the side's own line increases regardless of favorite/underdog. A -8.5
    favorite teased 6 becomes -2.5; a +1.5 underdog teased 6 becomes +7.5.
    """
    return leg_line + points


def tease_total(line: float, side: str, points: float = WONG_TEASER_POINTS) -> float:
    """Move a total *line* by *points* toward the bettor for an over/under leg."""
    if side == "over":
        return line - points
    if side == "under":
        return line + points
    raise ValueError(f"side must be 'over' or 'under', got {side!r}")


def evaluate_teaser_spread_leg(
    margin_counts: Mapping[str | int | float, int],
    leg_line: float,
    side: str,
) -> LineBetEvaluation:
    """Evaluate one *already-teased* spread leg against the home-minus-away pmf.

    ``leg_line`` is that side's signed teased spread number (e.g. -2.5 for a
    teased favorite, +7.5 for a teased underdog), matching the engine's
    ``spread_home`` convention (negative = favored). Home covers when
    ``margin + leg_line > 0``; away covers when ``-margin + leg_line > 0``.
    """
    if side == "home":
        return evaluate_threshold_bet(margin_counts, -leg_line, direction="over")
    if side == "away":
        return evaluate_threshold_bet(margin_counts, leg_line, direction="under")
    raise ValueError(f"side must be 'home' or 'away', got {side!r}")


def evaluate_teaser_total_leg(
    total_counts: Mapping[str | int | float, int],
    leg_line: float,
    side: str,
) -> LineBetEvaluation:
    """Evaluate one *already-teased* total leg against the total pmf."""
    if side not in {"over", "under"}:
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")
    return evaluate_threshold_bet(total_counts, leg_line, direction=side)
