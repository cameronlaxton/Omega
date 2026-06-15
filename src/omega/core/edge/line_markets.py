"""Sport-neutral line-bet evaluation from an outcome pmf.

The same primitive prices any over/under-a-line bet against an empirical pmf of
some integer-valued outcome — soccer Asian handicaps and first-half totals
(``soccer_derivatives``) and NFL Wong-teaser legs (``nfl_teasers``) are all the
same threshold bet on a margin/total pmf. This module owns that primitive so the
two sports share one implementation rather than duplicating the quarter-ball
split, push and half-stake semantics.

Quarter-ball semantics: a line ending in .25/.75 splits the stake across the two
adjacent half/integer lines. Each half-stake resolves independently (win =
half-stake at full odds, push = half-stake returned, lose = half-stake lost),
producing the standard five outcome buckets.

EV bridging: the existing edge framework prices binary win/lose bets from a
single probability. ``equivalent_win_prob`` converts the exact five-bucket EV
into the binary-equivalent probability ``q = (EV + 1) / decimal_odds`` so the
downstream edge/EV/Kelly math reproduces the exact line-bet EV without changing
``EdgeDetail``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from omega.core.betting.odds import american_to_decimal


def _sub_lines(line: float) -> list[float]:
    """Split a quarter-ball line into its two component lines; else identity."""
    if (line * 4) % 2 == 1.0:  # .25 or .75 fraction -> split into adjacent lines
        return [line - 0.25, line + 0.25]
    return [line]


@dataclass(frozen=True)
class LineBetEvaluation:
    """Outcome buckets of one (possibly split-stake) line bet."""

    p_win: float
    p_half_win: float
    p_push: float
    p_half_loss: float
    p_loss: float

    def ev_per_unit(self, american: float) -> float:
        """Exact expected profit per unit stake at *american* odds."""
        profit = american_to_decimal(american) - 1.0
        return (
            self.p_win * profit
            + self.p_half_win * profit / 2.0
            - self.p_half_loss * 0.5
            - self.p_loss
        )

    def equivalent_win_prob(self, american: float) -> float:
        """Binary-equivalent win probability reproducing ``ev_per_unit``.

        Solving ``q*(d-1) - (1-q) = EV`` gives ``q = (EV+1)/d``. Clamped to
        [0, 1] so degenerate quotes cannot produce out-of-range probabilities.
        """
        q = (self.ev_per_unit(american) + 1.0) / american_to_decimal(american)
        return max(0.0, min(1.0, q))


def _normalize_counts(counts: Mapping[str | int | float, int]) -> list[tuple[float, float]]:
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("empty value counts; backend emitted no score samples")
    return [(float(value), n / total) for value, n in counts.items()]


def evaluate_threshold_bet(
    counts: Mapping[str | int | float, int],
    line: float,
    *,
    direction: str,
) -> LineBetEvaluation:
    """Evaluate a bet that wins when value is above/below *line*.

    ``direction="over"``: wins when value > line; ``"under"``: value < line.
    Equality on a sub-line is a push (only possible on integer/half lines that
    integer-valued samples can hit). Quarter lines split into two half-stakes.
    """
    if direction not in {"over", "under"}:
        raise ValueError(f"direction must be 'over' or 'under', got {direction!r}")
    pmf = _normalize_counts(counts)
    subs = _sub_lines(line)
    weight = 1.0 / len(subs)

    win = half_win = push = half_loss = loss = 0.0
    for value, p in pmf:
        outcomes = []
        for sub in subs:
            if value == sub:
                outcomes.append(0)
            elif (value > sub) == (direction == "over"):
                outcomes.append(1)
            else:
                outcomes.append(-1)
        score = sum(outcomes) * weight  # in {-1, -0.5, 0, 0.5, 1}
        if score == 1.0:
            win += p
        elif score == 0.5:
            half_win += p
        elif score == 0.0:
            push += p
        elif score == -0.5:
            half_loss += p
        else:
            loss += p
    return LineBetEvaluation(
        p_win=win, p_half_win=half_win, p_push=push, p_half_loss=half_loss, p_loss=loss
    )
