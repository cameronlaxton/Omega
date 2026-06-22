"""Backtest metrics.

Probability accuracy is delegated to ``CalibrationFitter.evaluate`` (the single
ECE/Brier/log-loss implementation) — applied once to raw predictions and once to
already-calibrated predictions, so raw-vs-calibrated share one code path. Betting
performance is computed separately from ``ReplayCandidateSelection`` rows graded
against outcomes via the engine's ``settle_game_bet`` / ``compute_pnl``. The two
are never mixed: probability quality and ROI are reported in distinct blocks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from omega.core.betting.odds import american_to_decimal
from omega.core.calibration.fitter import CalibrationFitter
from omega.core.calibration.profiles import CalibrationProfile
from omega.historical.contracts import (
    BettingBlock,
    HealthBlock,
    MetricBlock,
    ReplayCandidateSelection,
    ReplayEventRecord,
)
from omega.trace.bet_settlement import compute_pnl, settle_game_bet
from omega.trace.ledger_bet import LedgerStatus

_NONE_PROFILE = CalibrationProfile(
    profile_id="__identity__",
    version=1,
    method="none",
    league="*",
    training_window="none",
    sample_size=0,
    dataset_hash="none",
)


def probability_metrics(
    raw_preds: Sequence[float],
    calibrated_preds: Sequence[float],
    outcomes: Sequence[int],
) -> MetricBlock:
    """Raw-vs-calibrated Brier/ECE/log-loss via the shared fitter evaluation.

    ``calibrated_preds`` are already-calibrated probabilities; passing them
    through the identity ("none") profile reuses the exact same metric code.
    """
    if not raw_preds:
        return MetricBlock(n=0)
    fitter = CalibrationFitter()
    raw = fitter.evaluate(_NONE_PROFILE, list(raw_preds), list(outcomes))
    cal = fitter.evaluate(_NONE_PROFILE, list(calibrated_preds), list(outcomes))
    return MetricBlock(
        raw_brier=raw["brier_score"],
        calibrated_brier=cal["brier_score"],
        raw_ece=raw["calibration_error"],
        calibrated_ece=cal["calibration_error"],
        raw_log_loss=raw["log_loss"],
        calibrated_log_loss=cal["log_loss"],
        n=len(raw_preds),
    )


def _clv(sel: ReplayCandidateSelection, closing_list: list[dict]) -> float | None:
    """Closing-line value as decision_decimal / closing_decimal - 1 (reporting only)."""
    if sel.decision_odds is None:
        return None
    for c in closing_list:
        if (
            c.get("market") == sel.market
            and c.get("selection_descriptor") == sel.selection_descriptor
        ):
            close_odds = c.get("closing_odds")
            if close_odds is None:
                continue
            close_dec = american_to_decimal(close_odds)
            if close_dec <= 0:
                continue
            return round(american_to_decimal(sel.decision_odds) / close_dec - 1.0, 4)
    return None


def _max_drawdown(nets: Iterable[float]) -> float:
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for n in nets:
        cum += n
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return round(mdd, 2)


def betting_metrics(
    selections: list[ReplayCandidateSelection],
    outcomes_by_event: dict[str, dict],
    closing_by_trace: dict[str, list[dict]] | None = None,
) -> BettingBlock:
    """Grade candidate selections against outcomes into ROI/PnL/CLV/etc.

    ``outcomes_by_event`` maps event_id → {"home_score", "away_score"}.
    ``closing_by_trace`` maps trace_id → list of closing-line dicts (CLV only).
    """
    closing_by_trace = closing_by_trace or {}
    ordered = sorted(selections, key=lambda s: s.decision_time)

    nets: list[float] = []
    total_staked = 0.0
    wins = 0
    decided = 0
    gross_win = 0.0
    gross_loss = 0.0
    clvs: list[float] = []

    for sel in ordered:
        oc = outcomes_by_event.get(sel.event_id)
        if not oc or oc.get("home_score") is None or oc.get("away_score") is None:
            continue
        if sel.decision_odds is None:
            continue
        side = sel.selection_descriptor.split("_", 1)[0]
        status = settle_game_bet(
            sel.market, side, sel.decision_line, oc["home_score"], oc["away_score"]
        )
        stake = sel.stake_amount or 0.0
        _payout, net = compute_pnl(status, sel.decision_odds, stake)
        if net is None:
            continue
        nets.append(net)
        total_staked += stake
        if status == LedgerStatus.WON:
            wins += 1
        if status in (LedgerStatus.WON, LedgerStatus.LOST):
            decided += 1
        if net > 0:
            gross_win += net
        elif net < 0:
            gross_loss += -net
        clv = _clv(sel, closing_by_trace.get(sel.trace_id, []))
        if clv is not None:
            clvs.append(clv)

    if not nets:
        return BettingBlock(n_bets=0)

    net_total = round(sum(nets), 2)
    return BettingBlock(
        roi=round(net_total / total_staked, 4) if total_staked else None,
        net_pnl=net_total,
        hit_rate=round(wins / decided, 4) if decided else None,
        profit_factor=round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        max_drawdown=_max_drawdown(nets),
        avg_clv=round(sum(clvs) / len(clvs), 4) if clvs else None,
        n_bets=len(nets),
    )


def health_metrics(
    records: list[ReplayEventRecord], *, fallback_profile_rate: float = 0.0
) -> HealthBlock:
    """Visibility rates over a set of replay event records."""
    n = len(records) or 1
    return HealthBlock(
        missing_odds_rate=round(sum(1 for r in records if r.missing_odds) / n, 4),
        leakage_skip_count=sum(1 for r in records if r.leakage_status in ("skipped", "failed")),
        identity_failure_count=sum(1 for r in records if r.identity_status == "missing"),
        fallback_profile_rate=round(fallback_profile_rate, 4),
        default_context_rate=round(sum(1 for r in records if r.context_source == "default") / n, 4),
        stale_context_rate=round(sum(1 for r in records if r.is_stale) / n, 4),
    )
