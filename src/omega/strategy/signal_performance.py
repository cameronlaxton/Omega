"""
Retrospective evidence-signal scoring (Phase 6i — Phase C).

After outcomes attach to traces, this module measures whether each structured
reasoning signal was actually predictive: did its stated ``direction`` match the
realized result, and did its stated ``confidence`` match its empirical accuracy?

Aggregates are keyed by ``(signal_type, source, obs_window, league)`` so the
agent can learn, for example, that opponent-rank signals from box-score data are
directionally correct 71% of the time while agent-judgment outlier calls are
correct 48% of the time (below random) — and weight its evidence accordingly.

This module is pure (no I/O, no DB). ``omega-score-evidence-signals`` does
the trace/outcome/evidence JOIN and persistence; this file owns only the math.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Directions that make a player-prop claim vs. a game claim.
_PROP_DIRECTIONS = frozenset({"over", "under"})
_GAME_DIRECTIONS = frozenset({"home", "away", "draw"})


@dataclass(frozen=True)
class MarketMove:
    """Closing-line move for one trace+plane, prepared by the ops layer.

    ``favored_direction`` is the side the line moved toward by close — the side
    that beat the close ('over'/'under' for the prop plane, 'home'/'away' for the
    game plane). ``clv_cents`` is the magnitude of that move in implied-prob cents
    (always >= 0). ``favored_direction`` is None when the line did not move, so
    CLV alignment is undefined and the signal is scored on direction only.
    """

    favored_direction: str | None
    clv_cents: float


@dataclass(frozen=True)
class ScoredSignal:
    """One evidence signal scored against its trace's realized outcome and close.

    ``direction_correct`` is None when the realized outcome was unavailable or a
    push — the signal can still contribute CLV when a closing-line move existed.
    ``clv_aligned`` is None when no closing-line move was available for the
    signal's plane; otherwise True iff following the signal's direction would have
    captured positive closing-line value. ``clv_cents`` is the signed CLV (implied-
    prob cents) of following the signal (positive = beat the close).
    """

    signal_type: str
    source: str
    obs_window: str
    league: str
    confidence: float
    direction_correct: bool | None
    clv_aligned: bool | None = None
    clv_cents: float | None = None


@dataclass(frozen=True)
class SignalPerformanceRow:
    """Aggregated retrospective performance for one signal key.

    Carries two parallel measures: the legacy ``direction_*`` block (kept as an
    audit column — realized direction is what the closing line already contains)
    and the CLV block (``clv_aligned`` rate, ``clv_cents_when_followed``,
    ``clv_sample``) which is the sample-efficient, market-relative trust driver
    (issue #28). CLV fields are None / 0 where closing-line coverage is absent.
    """

    signal_type: str
    source: str
    obs_window: str
    league: str
    sample_size: int
    direction_correct: int
    direction_accuracy: float
    mean_confidence: float
    realized_hit_rate: float
    calibration_gap: float
    brier: float
    clv_aligned: float | None = None
    clv_cents_when_followed: float | None = None
    clv_sample: int = 0
    # Sample SD of the per-observation CLV cents (issue #28). Stored as a sufficient
    # statistic so the fit can POOL (n, mean, std) across this signal type's rows
    # (source/window/league) and compute a signal_type-level confidence bound +
    # p-value without re-reading raw observations. None when clv_sample < 2.
    clv_cents_std: float | None = None


# ---------------------------------------------------------------------------
# Realized-outcome resolution
# ---------------------------------------------------------------------------


def realized_prop_direction(prop_outcomes: list[dict[str, Any]] | None) -> str | None:
    """Resolve a trace's realized prop direction ('over' / 'under').

    Returns None when there is no prop outcome or the result was a push. A prop
    trace normally carries a single prop outcome; the first row is used.
    """
    if not prop_outcomes:
        return None
    row = prop_outcomes[0]
    try:
        stat_value = float(row["stat_value"])
        line = float(row["line"])
    except (KeyError, TypeError, ValueError):
        return None
    if stat_value > line:
        return "over"
    if stat_value < line:
        return "under"
    return None  # push — no directional outcome


def realized_game_direction(outcome: dict[str, Any] | None) -> str | None:
    """Resolve a trace's realized game direction ('home' / 'away' / 'draw').

    Returns None only for a missing/unrecognised outcome. A tie resolves to
    'draw' so 3-way (soccer, hockey regulation) draw-direction signals can be
    scored; non-draw sports never store ``result == 'draw'`` for graded games.
    """
    if not outcome:
        return None
    result = str(outcome.get("result") or "")
    if result == "home_win":
        return "home"
    if result == "away_win":
        return "away"
    if result == "draw":
        return "draw"
    return None


# ---------------------------------------------------------------------------
# Per-trace scoring
# ---------------------------------------------------------------------------


def score_trace_signals(
    trace: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    *,
    prop_move: MarketMove | None = None,
    game_move: MarketMove | None = None,
) -> list[ScoredSignal]:
    """Score every directional evidence signal on one trace.

    Scores two parallel measures per signal:
      * direction-correctness vs the realized outcome (audit only — restating
        public info that the closing line already contains);
      * CLV alignment vs the closing-line move (``prop_move`` / ``game_move``,
        prepared by the ops layer): did following the signal's direction beat the
        close? Sample-efficient and market-relative — the trust driver.

    Args:
        trace: A trace dict from ``query_traces`` — may carry ``_outcome``
            (game) and/or ``_prop_outcomes`` (prop).
        evidence_rows: ``evidence_signals`` rows for the trace.
        prop_move/game_move: the closing-line move for each plane, or None when
            no closing line was available (CLV degrades to direction-only).

    Returns:
        One ScoredSignal per directional signal that could be scored on at least
        one of the two measures. Signals with no direction (or 'neutral') are
        skipped, as are signals with neither a realized outcome nor a close move.
    """
    prop_dir = realized_prop_direction(trace.get("_prop_outcomes"))
    game_dir = realized_game_direction(trace.get("_outcome"))

    scored: list[ScoredSignal] = []
    for row in evidence_rows:
        direction = row.get("direction")
        if direction in _PROP_DIRECTIONS:
            realized = prop_dir
            move = prop_move
        elif direction in _GAME_DIRECTIONS:
            realized = game_dir
            move = game_move
        else:
            continue  # neutral / None — no directional claim to score

        direction_correct = None if realized is None else (direction == realized)

        clv_aligned: bool | None = None
        clv_cents: float | None = None
        if move is not None and move.favored_direction is not None:
            aligned = direction == move.favored_direction
            clv_aligned = aligned
            # Following the signal captures +move.clv_cents if it pointed the way
            # the line moved, else the (symmetric) opposite cost.
            clv_cents = move.clv_cents if aligned else -move.clv_cents

        # Nothing to score: neither a realized direction (push / ungraded) nor a
        # closing-line move. Skip so empty observations do not dilute aggregates.
        if direction_correct is None and clv_aligned is None:
            continue

        confidence = row.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5

        scored.append(
            ScoredSignal(
                signal_type=str(row.get("signal_type") or "unknown"),
                source=str(row.get("source") or "unknown"),
                obs_window=str(row.get("obs_window") or "unknown"),
                league=str(row.get("league") or "unknown"),
                confidence=confidence,
                direction_correct=direction_correct,
                clv_aligned=clv_aligned,
                clv_cents=clv_cents,
            )
        )
    return scored


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def accumulate_signal_performance(
    scored: Iterable[ScoredSignal],
) -> list[SignalPerformanceRow]:
    """Aggregate scored signals into per-key performance rows.

    ``calibration_gap`` is ``mean_confidence - realized_hit_rate`` — positive
    means the agent was overconfident in that signal. ``brier`` treats the
    stated confidence as a probabilistic forecast of direction-correctness.

    Rows are returned sorted by key for deterministic, reproducible output.
    """
    buckets: dict[tuple[str, str, str, str], list[ScoredSignal]] = {}
    for s in scored:
        buckets.setdefault((s.signal_type, s.source, s.obs_window, s.league), []).append(s)

    rows: list[SignalPerformanceRow] = []
    for (signal_type, source, obs_window, league), group in sorted(buckets.items()):
        # Direction block (audit): only observations with a realized outcome.
        dir_scored = [s for s in group if s.direction_correct is not None]
        n_dir = len(dir_scored)
        n_correct = sum(1 for s in dir_scored if s.direction_correct)
        accuracy = (n_correct / n_dir) if n_dir else 0.0
        mean_conf = (sum(s.confidence for s in dir_scored) / n_dir) if n_dir else 0.0
        brier = (
            sum((s.confidence - (1.0 if s.direction_correct else 0.0)) ** 2 for s in dir_scored)
            / n_dir
            if n_dir
            else 0.0
        )

        # CLV block (trust driver): only observations with a closing-line move.
        clv_scored = [s for s in group if s.clv_aligned is not None]
        clv_sample = len(clv_scored)
        clv_aligned_rate: float | None = None
        clv_cents_when_followed: float | None = None
        clv_cents_std: float | None = None
        if clv_sample:
            cents = [float(s.clv_cents or 0.0) for s in clv_scored]
            clv_aligned_rate = sum(1 for s in clv_scored if s.clv_aligned) / clv_sample
            clv_cents_when_followed = statistics.fmean(cents)
            clv_cents_std = statistics.stdev(cents) if clv_sample >= 2 else 0.0

        rows.append(
            SignalPerformanceRow(
                signal_type=signal_type,
                source=source,
                obs_window=obs_window,
                league=league,
                sample_size=n_dir,
                direction_correct=n_correct,
                direction_accuracy=accuracy,
                mean_confidence=mean_conf,
                realized_hit_rate=accuracy,
                calibration_gap=mean_conf - accuracy,
                brier=brier,
                clv_aligned=clv_aligned_rate,
                clv_cents_when_followed=clv_cents_when_followed,
                clv_sample=clv_sample,
                clv_cents_std=clv_cents_std,
            )
        )
    return rows
