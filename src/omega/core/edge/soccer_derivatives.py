"""Asian-handicap and first-half-total evaluation from soccer score pmfs.

Phase 7 M2 (design Part 4): the soccer backend emits empirical pmfs of the
goal margin (``margin_counts``), full-time total (``total_counts``) and the
thinned first-half total (``fh_total_counts``). This module evaluates handicap
and total lines — including quarter-ball lines — against those pmfs. The
backend stays line-unaware; the soccer edge consumer
(``omega/core/edge/soccer_consumer.py``) turns the evaluations into EdgeDetail rows.

The generic threshold-bet primitive (quarter-ball split, push/half-stake
buckets, the EV bridge) lives in ``omega.core.edge.line_markets`` and is shared
with NFL Wong teasers. ``LineBetEvaluation`` and ``evaluate_threshold_bet`` are
re-exported here for backward compatibility with existing soccer callers/tests.
"""

from __future__ import annotations

from collections.abc import Mapping

from omega.core.edge.line_markets import (
    LineBetEvaluation,
    evaluate_threshold_bet,
)

__all__ = [
    "LineBetEvaluation",
    "evaluate_threshold_bet",
    "evaluate_asian_handicap",
    "evaluate_total",
]


def evaluate_asian_handicap(
    margin_counts: Mapping[str | int | float, int],
    home_line: float,
    side: str,
) -> LineBetEvaluation:
    """Evaluate the home/away side of a home-quoted Asian handicap.

    ``margin_counts`` is the empirical pmf of (home_goals - away_goals).
    Home with line ``h`` wins when ``margin + h > 0`` i.e. margin > -h; away
    (line ``-h``) wins when margin < -h. Pushes land on margin == -h.
    """
    if side not in {"home", "away"}:
        raise ValueError(f"side must be 'home' or 'away', got {side!r}")
    direction = "over" if side == "home" else "under"
    return evaluate_threshold_bet(margin_counts, -home_line, direction=direction)


def evaluate_total(
    total_counts: Mapping[str | int | float, int],
    line: float,
    side: str,
) -> LineBetEvaluation:
    """Evaluate an over/under at *line* against an empirical totals pmf."""
    if side not in {"over", "under"}:
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")
    return evaluate_threshold_bet(total_counts, line, direction=side)
