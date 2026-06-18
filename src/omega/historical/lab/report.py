"""Deterministic Markdown renderer for a lab run.

``render`` is a **pure function of its model arguments** — no wall-clock, cwd, or
environment reads — so golden snapshot tests are stable. It does not recompute any
metric: probability/betting numbers are read from the typed ``BacktestReport``
blocks (produced by the single ``omega.historical.metrics`` path) and the variant
ledger. The report's job is to stitch and to surface the winner's-curse framing
that nothing else reports.
"""

from __future__ import annotations

from typing import Any

from omega.historical.lab.schemas import (
    AttemptedVariantLedger,
    HistoricalLabRun,
    PromotionEvidenceBundle,
)


def _f(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def render(
    lab_run: HistoricalLabRun,
    ledger: AttemptedVariantLedger,
    evidence: PromotionEvidenceBundle,
    backtest_report: Any | None = None,
) -> str:
    """Render REPORT.md for a lab run. Pure → deterministic for golden tests."""
    out: list[str] = []
    w = out.append

    w(f"# Historical Validation Lab — {lab_run.lab_run_id}")
    w("")
    w("## Provenance")
    w(f"- created_at: {lab_run.created_at}")
    w(f"- code_version: {lab_run.code_version}")
    w(f"- git_commit: {lab_run.git_commit}")
    w(f"- working_tree_dirty: {lab_run.working_tree_dirty}")
    w(f"- league / plane / market: {lab_run.league} / {lab_run.plane} / {lab_run.market}")
    w(f"- dataset_manifest_id: {lab_run.dataset_manifest_id}")
    w(f"- dataset_hash: {lab_run.dataset_hash}")
    w(f"- replay_id: {lab_run.replay_id}")
    w(f"- replay_config_hash: {lab_run.replay_config_hash}")
    w(f"- profile_grid_hash: {lab_run.profile_grid_hash}")
    w("")

    w("## Windows")
    w(f"- train:      {lab_run.train_window.start} .. {lab_run.train_window.end}")
    w(f"- validation: {lab_run.validation_window.start} .. {lab_run.validation_window.end}")
    w(f"- holdout:    {lab_run.holdout_window.start} .. {lab_run.holdout_window.end}")
    w(f"- holdout_sealed: {lab_run.holdout_sealed} (access_count={lab_run.holdout_access_count})")
    w("")

    w("## Promotion")
    w(f"- status: **{lab_run.promotion_status}**")
    w(f"- auto_promote_armed: {lab_run.auto_promote_armed}")
    w(f"- candidate_id: {_f(evidence.candidate_id)}")
    w(f"- incumbent_id: {_f(evidence.incumbent_id)}")
    w(f"- recommended: {evidence.recommended}")
    w("")
    w("### Parity verdicts")
    w(f"- backtest_parity: {_f(evidence.backtest_parity_verdict)}")
    w(f"- clv_walk_forward: {_f(evidence.clv_walk_forward_verdict)}")
    w(f"- historical_live_parity: {_f(evidence.historical_live_parity_verdict)}")
    gp = evidence.gate_report
    gate = "PASS" if (gp and gp.get("passed")) else ("FAIL" if gp else "not evaluated")
    w(f"- promotion_gate: {gate}")
    w("")

    wc = evidence.winners_curse
    w("## Winner's-curse")
    if wc is not None:
        w(f"- attempted variants (N): {wc.n_variants}")
        w(f"- validation→holdout ECE delta: {_f(wc.val_to_holdout_ece_delta)}")
        w(f"- risk: **{wc.risk}**")
    else:
        w("- (no winner selected)")
    w("")

    w("## Attempted variants")
    w("")
    w("| variant_id | family | slice | n_train | n_val | val_brier | val_ece | cv_ece | holdout_ece | status |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for v in ledger.variants:
        w(
            f"| {v.variant_id} | {v.profile_family} | {v.context_slice or 'base'} | "
            f"{v.sample_size} | {v.n_validation} | {_f(v.brier)} | {_f(v.ece)} | "
            f"{_f(v.cv_ece)} | {_f(v.holdout_ece)} | {v.status} |"
        )
    w("")

    if backtest_report is not None:
        w("## Probability quality (walk-forward aggregate)")
        w("")
        w("| market | raw_brier | cal_brier | raw_ece | cal_ece | n |")
        w("|---|---|---|---|---|---|")
        for mk in sorted(backtest_report.aggregate_metrics_by_market):
            mb = backtest_report.aggregate_metrics_by_market[mk]
            w(
                f"| {mk} | {_f(mb.raw_brier)} | {_f(mb.calibrated_brier)} | "
                f"{_f(mb.raw_ece)} | {_f(mb.calibrated_ece)} | {mb.n} |"
            )
        w("")
        bb = backtest_report.aggregate_betting
        w("## Betting diagnostics (walk-forward aggregate)")
        if bb is not None:
            w(
                f"- n_bets: {bb.n_bets} | roi: {_f(bb.roi)} | "
                f"avg_clv: {_f(bb.avg_clv)} | max_drawdown: {_f(bb.max_drawdown)}"
            )
        else:
            w("- no graded bets")
        w("")

    return "\n".join(out) + "\n"
