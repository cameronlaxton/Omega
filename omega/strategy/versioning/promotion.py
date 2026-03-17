"""
Promotion workflow — automate strategy lifecycle transitions.

Workflow:
    1. Register strategy (→ CANDIDATE)
    2. Run backtest (→ BACKTESTING → STAGING or REJECTED)
    3. Review (manual or auto-promote based on criteria)
    4. Promote to PRODUCTION (archives previous production version)

Auto-promotion criteria (all must be met):
    - Backtest passed (backtest engine's own criteria)
    - ROI ≥ min_roi_pct
    - Win rate ≥ min_win_rate
    - Sample size ≥ min_sample_size
    - CLV ≥ min_clv (if available)
    - Max drawdown ≤ max_drawdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from omega.strategy.models import BacktestResult, StrategyEntry, StrategyStatus
from omega.strategy.versioning.registry import StrategyRegistry

logger = logging.getLogger("omega.strategy.promotion")


@dataclass
class PromotionCriteria:
    """Configurable criteria for auto-promotion."""
    min_roi_pct: float = 2.0
    min_win_rate: float = 0.48
    min_sample_size: int = 30
    min_clv: float = 0.0
    max_drawdown: float = 20.0


def evaluate_for_promotion(
    result: BacktestResult,
    criteria: Optional[PromotionCriteria] = None,
) -> tuple[bool, List[str]]:
    """Evaluate whether a backtest result meets promotion criteria.

    Returns (should_promote, rejection_reasons).
    """
    criteria = criteria or PromotionCriteria()
    reasons: List[str] = []

    if not result.passed:
        reasons.extend(result.rejection_reasons)

    if result.total_bets_placed < criteria.min_sample_size:
        reasons.append(
            f"Sample too small: {result.total_bets_placed} < {criteria.min_sample_size}"
        )

    if result.roi_pct < criteria.min_roi_pct:
        reasons.append(
            f"ROI too low: {result.roi_pct:.1f}% < {criteria.min_roi_pct:.1f}%"
        )

    if result.win_rate < criteria.min_win_rate:
        reasons.append(
            f"Win rate too low: {result.win_rate:.1%} < {criteria.min_win_rate:.1%}"
        )

    if result.max_drawdown_units > criteria.max_drawdown:
        reasons.append(
            f"Drawdown too high: {result.max_drawdown_units:.1f} > {criteria.max_drawdown:.1f}"
        )

    if result.avg_closing_line_value < criteria.min_clv:
        reasons.append(
            f"CLV too low: {result.avg_closing_line_value:.1f}% < {criteria.min_clv:.1f}%"
        )

    should_promote = len(reasons) == 0
    return should_promote, reasons


def auto_promote_or_reject(
    registry: StrategyRegistry,
    strategy_id: str,
    version: int,
    result: BacktestResult,
    criteria: Optional[PromotionCriteria] = None,
    decided_by: str = "auto_promoter",
) -> StrategyEntry:
    """Record backtest, then auto-promote or reject based on criteria.

    Returns the updated StrategyEntry.
    """
    # Record backtest result in registry
    entry = registry.record_backtest(strategy_id, version, result)

    if entry.status == StrategyStatus.REJECTED:
        logger.info(
            "Strategy %s v%d rejected by backtest engine: %s",
            strategy_id, version, result.rejection_reasons,
        )
        return entry

    # Evaluate promotion criteria
    should_promote, reasons = evaluate_for_promotion(result, criteria)

    if should_promote:
        entry = registry.promote(
            strategy_id=strategy_id,
            version=version,
            reason=f"Auto-promoted: ROI={result.roi_pct:.1f}%, WR={result.win_rate:.1%}, CLV={result.avg_closing_line_value:.1f}%",
            decided_by=decided_by,
            backtest_run_id=result.run_id,
        )
        logger.info("Auto-promoted %s v%d to production", strategy_id, version)
    else:
        entry = registry.reject(
            strategy_id=strategy_id,
            version=version,
            reason=f"Failed promotion criteria: {'; '.join(reasons)}",
            decided_by=decided_by,
        )
        logger.info("Rejected %s v%d: %s", strategy_id, version, reasons)

    return entry
