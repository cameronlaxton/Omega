"""
Retrospective evidence-signal scoring (Phase 6i — Phase C).

After outcomes attach to traces, this module measures whether each structured
reasoning signal was actually predictive: did its stated ``direction`` match the
realized result, and did its stated ``confidence`` match its empirical accuracy?

Aggregates are keyed by ``(signal_type, source, obs_window, league)`` so the
agent can learn, for example, that opponent-rank signals from box-score data are
directionally correct 71% of the time while agent-judgment outlier calls are
correct 48% of the time (below random) — and weight its evidence accordingly.

This module is pure (no I/O, no DB). ``scripts/score_evidence_signals.py`` does
the trace/outcome/evidence JOIN and persistence; this file owns only the math.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Directions that make a player-prop claim vs. a game claim.
_PROP_DIRECTIONS = frozenset({"over", "under"})
_GAME_DIRECTIONS = frozenset({"home", "away", "draw"})


@dataclass(frozen=True)
class ScoredSignal:
    """One evidence signal scored against its trace's realized outcome."""

    signal_type: str
    source: str
    obs_window: str
    league: str
    confidence: float
    direction_correct: bool


@dataclass(frozen=True)
class SignalPerformanceRow:
    """Aggregated retrospective performance for one signal key."""

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
) -> list[ScoredSignal]:
    """Score every directional evidence signal on one graded trace.

    Args:
        trace: A trace dict from ``query_traces`` — may carry ``_outcome``
            (game) and/or ``_prop_outcomes`` (prop).
        evidence_rows: ``evidence_signals`` rows for the trace, as returned by
            ``TraceStore.get_evidence_signals``.

    Returns:
        One ScoredSignal per directional signal whose claim could be resolved.
        Signals with no direction (or 'neutral'), or whose plane's outcome is
        unavailable or a push, are skipped. A 3-way game draw resolves to the
        'draw' direction and is scored.
    """
    prop_dir = realized_prop_direction(trace.get("_prop_outcomes"))
    game_dir = realized_game_direction(trace.get("_outcome"))

    scored: list[ScoredSignal] = []
    for row in evidence_rows:
        direction = row.get("direction")
        if direction in _PROP_DIRECTIONS:
            realized = prop_dir
        elif direction in _GAME_DIRECTIONS:
            realized = game_dir
        else:
            continue  # neutral / None — no directional claim to score
        if realized is None:
            continue  # outcome unavailable or a push

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
                direction_correct=(direction == realized),
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
        n = len(group)
        n_correct = sum(1 for s in group if s.direction_correct)
        accuracy = n_correct / n
        mean_conf = sum(s.confidence for s in group) / n
        brier = sum(
            (s.confidence - (1.0 if s.direction_correct else 0.0)) ** 2 for s in group
        ) / n
        rows.append(
            SignalPerformanceRow(
                signal_type=signal_type,
                source=source,
                obs_window=obs_window,
                league=league,
                sample_size=n,
                direction_correct=n_correct,
                direction_accuracy=accuracy,
                mean_confidence=mean_conf,
                realized_hit_rate=accuracy,
                calibration_gap=mean_conf - accuracy,
                brier=brier,
            )
        )
    return rows
