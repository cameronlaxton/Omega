"""Deterministic golden snapshot of the lab Markdown report.

``render`` is a pure function of its models, so the same fixtures always produce
the same Markdown. Run this module as a script to (re)generate the golden file
after an intentional format change:  ``python tests/historical/lab/test_report.py``
"""

from __future__ import annotations

from pathlib import Path

from omega.historical.lab.report import render
from omega.historical.lab.schemas import (
    AttemptedVariant,
    AttemptedVariantLedger,
    HistoricalLabRun,
    PromotionEvidenceBundle,
    Window,
    WinnersCurse,
)

GOLDEN = Path(__file__).parent / "fixtures" / "golden_report.md"

_TRAIN = Window(start="2023-01-01", end="2023-04-30")
_VALID = Window(start="2023-05-01", end="2023-06-15")
_HOLD = Window(start="2023-06-16", end="2023-07-31")


def _build():
    lab_run = HistoricalLabRun(
        lab_run_id="lab_golden",
        created_at="2023-08-01T00:00:00+00:00",
        code_version="omega-test",
        git_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        working_tree_dirty=False,
        dataset_manifest_id="mfest",
        dataset_hash="dshash",
        league="FIFA_INTL",
        plane="draw",
        replay_id="rep1",
        replay_db_path="/tmp/lab_golden.db",
        replay_config_hash="cfg",
        profile_grid_hash="grid",
        attempted_variant_count=3,
        train_window=_TRAIN,
        validation_window=_VALID,
        holdout_window=_HOLD,
        holdout_sealed=True,
        holdout_access_count=1,
        auto_promote_armed=False,
        promotion_status="shadow_only",
    )
    variants = [
        AttemptedVariant(
            variant_id="draw_isotonic_base",
            profile_family="isotonic",
            plane="draw",
            train_window=_TRAIN,
            validation_window=_VALID,
            holdout_window=_HOLD,
            sample_size=200,
            n_validation=60,
            brier=0.21,
            ece=0.03,
            cv_ece=0.032,
            holdout_brier=0.215,
            holdout_ece=0.035,
            n_holdout=55,
            status="selected",
        ),
        AttemptedVariant(
            variant_id="draw_shrinkage_base",
            profile_family="shrinkage",
            plane="draw",
            train_window=_TRAIN,
            validation_window=_VALID,
            holdout_window=_HOLD,
            sample_size=200,
            n_validation=60,
            brier=0.225,
            ece=0.05,
            cv_ece=0.052,
            status="rejected",
            rejection_reason="not selected",
        ),
        AttemptedVariant(
            variant_id="draw_isotonic_playoff",
            profile_family="isotonic",
            plane="draw",
            context_slice="playoff",
            train_window=_TRAIN,
            validation_window=_VALID,
            holdout_window=_HOLD,
            sample_size=10,
            n_validation=2,
            status="skipped",
            rejection_reason="insufficient pairs",
        ),
    ]
    ledger = AttemptedVariantLedger(
        lab_run_id="lab_golden", plane="draw", profile_grid_hash="grid", variants=variants
    )
    evidence = PromotionEvidenceBundle(
        lab_run_id="lab_golden",
        plane="draw",
        market="draw",
        gate_inputs={
            "sample_size": 200,
            "brier_score": 0.215,
            "calibration_error": 0.032,
            "cv_calibration_error": 0.032,
            "cv_n_folds": 20,
            "log_loss": 0.5,
        },
        backtest_parity_verdict="no_incumbent",
        clv_walk_forward_verdict="INCONCLUSIVE",
        historical_live_parity_verdict="INCONCLUSIVE",
        holdout_sealed=True,
        attempted_variant_count=3,
        winners_curse=WinnersCurse(n_variants=3, val_to_holdout_ece_delta=0.005, risk="elevated"),
        working_tree_dirty=False,
        recommended=False,
        decision="shadow_only",
    )
    return lab_run, ledger, evidence


def test_render_matches_golden():
    assert render(*_build()) == GOLDEN.read_text(encoding="utf-8")


def test_render_is_deterministic():
    assert render(*_build()) == render(*_build())


def test_render_includes_scorecard_and_model_vs_market_with_backtest_report():
    """The new fused sections render only when a BacktestReport is supplied."""
    from omega.historical.contracts import (
        BacktestReport,
        BettingBlock,
        MarginalValueBlock,
        MetricBlock,
        ModelVsMarketBlock,
        ScorecardRow,
        WalkForwardConfig,
    )

    lab_run, ledger, evidence = _build()
    bt = BacktestReport(
        manifest_id="mfest",
        replay_id="rep1",
        league="FIFA_INTL",
        walk_forward_config=WalkForwardConfig(markets=["game"]),
        aggregate_metrics_by_market={
            "game": MetricBlock(raw_ece=0.10, calibrated_ece=0.05, n=100)
        },
        aggregate_betting=BettingBlock(n_bets=20, roi=0.05, avg_clv=1.0, max_drawdown=2.0),
        aggregate_model_vs_market_by_market={
            "game": ModelVsMarketBlock(
                n=20, n_divergent=8, mean_signed_divergence=0.03, mean_abs_divergence=0.04,
                clv_when_divergent=-0.1, divergent_beat_close_rate=0.3,
            )
        },
        aggregate_marginal_value=[
            MarginalValueBlock(signal_type="recent_form_residual", brier_delta=0.01, n=20)
        ],
        scorecard=[
            ScorecardRow(
                market="game", n_calibrated=100, raw_ece=0.10, calibrated_ece=0.05,
                n_bets=20, roi=0.05, avg_clv=1.0, n_divergent=8,
                mean_signed_divergence=0.03, clv_when_divergent=-0.1,
                divergent_beat_close_rate=0.3, clv_coherent=False,
            )
        ],
    )
    out = render(lab_run, ledger, evidence, bt)
    assert "## Model vs market (incremental edge)" in out
    assert "## Scorecard (per plane)" in out
    assert "## Marginal signal value" in out
    assert "recent_form_residual" in out
    # The incoherent plane surfaces its False flag in the scorecard row.
    assert "| game |" in out


if __name__ == "__main__":
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(render(*_build()), encoding="utf-8")
    print("wrote", GOLDEN)
