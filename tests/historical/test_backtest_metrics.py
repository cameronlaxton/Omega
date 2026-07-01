"""Metrics: raw-vs-calibrated separation, betting ROI/CLV, health rates."""

from __future__ import annotations

from omega.historical.contracts import (
    BettingBlock,
    MetricBlock,
    ModelVsMarketBlock,
    ReplayCandidateSelection,
    ReplayEventRecord,
)
from omega.historical.metrics import (
    betting_metrics,
    build_scorecard,
    clv_coherent,
    health_metrics,
    model_vs_market_from_selections,
    probability_metrics,
    selection_plane,
)


def test_probability_metrics_separates_raw_and_calibrated():
    outs = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    raw = [0.9] * 10  # overconfident, ignores the zeros
    cal = [0.5] * 10  # honest about a coin flip
    m = probability_metrics(raw, cal, outs)
    assert m.n == 10
    assert m.raw_brier is not None and m.calibrated_brier is not None
    assert m.calibrated_brier < m.raw_brier
    # raw and calibrated are reported independently
    assert m.raw_ece != m.calibrated_ece


def _sel(event_id, trace_id, side, odds, dt, stake=100.0) -> ReplayCandidateSelection:
    return ReplayCandidateSelection(
        replay_id="r",
        event_id=event_id,
        trace_id=trace_id,
        market="moneyline",
        selection_descriptor=side,
        raw_prob=0.6,
        decision_odds=odds,
        decision_time=dt,
        stake_amount=stake,
    )


def test_betting_metrics_roi_pnl_clv():
    sels = [
        _sel("e1", "t1", "home", -110, "2023-01-01T00:00:00+00:00"),
        _sel("e2", "t2", "away", 150, "2023-01-02T00:00:00+00:00"),
    ]
    outcomes = {
        "e1": {"home_score": 24, "away_score": 17},  # home wins → home bet WON
        "e2": {"home_score": 24, "away_score": 17},  # home wins → away bet LOST
    }
    # decision -110 beat the -150 close → positive CLV
    closing = {
        "t1": [{"market": "moneyline", "selection_descriptor": "home", "closing_odds": -150}]
    }

    b = betting_metrics(sels, outcomes, closing)
    assert b.n_bets == 2
    assert b.hit_rate == 0.5
    assert b.net_pnl is not None
    assert b.roi is not None
    assert b.max_drawdown is not None
    assert b.avg_clv is not None and b.avg_clv > 0


def test_betting_metrics_empty_when_no_outcomes():
    sels = [_sel("e1", "t1", "home", -110, "2023-01-01T00:00:00+00:00")]
    b = betting_metrics(sels, outcomes_by_event={})
    assert b.n_bets == 0


def test_betting_metrics_skips_prop_selections_not_misgrade():
    """Prop selections are skipped (not settled as moneyline) — game-only grading guard."""
    sels = [
        _sel("e1", "t1", "home", -110, "2023-01-01T00:00:00+00:00"),
        ReplayCandidateSelection(
            replay_id="r",
            event_id="e1",
            trace_id="t2",
            market="prop",
            selection_descriptor="over",
            raw_prob=0.6,
            decision_odds=-110,
            decision_time="2023-01-01T00:00:00+00:00",
            stake_amount=100.0,
        ),
    ]
    outcomes = {"e1": {"home_score": 24, "away_score": 17}}
    b = betting_metrics(sels, outcomes)
    assert b.n_bets == 1  # moneyline graded; prop skipped rather than mis-settled


def _rec(event_id, **kw) -> ReplayEventRecord:
    base = dict(
        event_id=event_id,
        decision_time="2023-01-01T00:00:00+00:00",
        leakage_status="clean",
        identity_status="complete",
        context_source="provided",
        is_stale=False,
        missing_odds=False,
    )
    base.update(kw)
    return ReplayEventRecord(**base)


def test_health_metrics_rates():
    records = [
        _rec("e1"),
        _rec(
            "e2",
            leakage_status="skipped",
            leakage_reasons=["post_event_features"],
            identity_status="missing",
            context_source="default",
            is_stale=True,
            missing_odds=True,
        ),
    ]
    h = health_metrics(records, fallback_profile_rate=0.25)
    assert h.leakage_skip_count == 1
    assert h.identity_failure_count == 1
    assert h.missing_odds_rate == 0.5
    assert h.default_context_rate == 0.5
    assert h.stale_context_rate == 0.5
    assert h.fallback_profile_rate == 0.25


class TestMarginalAndModelVsMarket:
    """Issue #28 WS1/WS4 scorecard foundation."""

    def test_marginal_value_rewards_a_helpful_signal(self):
        from omega.historical.metrics import marginal_value

        outcomes = [1, 1, 0, 0]
        # WITH the signal the forecast is sharper toward the truth than WITHOUT.
        preds_with = [0.8, 0.7, 0.2, 0.3]
        preds_without = [0.55, 0.55, 0.45, 0.45]
        blk = marginal_value("recent_form_residual", preds_with, preds_without, outcomes)
        assert blk.signal_type == "recent_form_residual"
        assert blk.brier_delta > 0  # signal improves (lowers) Brier
        assert blk.n == 4

    def test_marginal_value_penalizes_a_harmful_signal(self):
        from omega.historical.metrics import marginal_value

        outcomes = [1, 1, 0, 0]
        preds_with = [0.4, 0.45, 0.6, 0.55]  # pushed the wrong way
        preds_without = [0.55, 0.55, 0.45, 0.45]
        blk = marginal_value("noise_signal", preds_with, preds_without, outcomes)
        assert blk.brier_delta < 0  # signal hurts

    def test_marginal_value_empty(self):
        from omega.historical.metrics import marginal_value

        blk = marginal_value("x", [], [], [])
        assert blk.n == 0 and blk.brier_delta is None

    def test_model_vs_market_rewards_vindicated_divergence(self):
        from omega.historical.metrics import model_vs_market

        # Three decisions diverge from market; CLV positive on those.
        model = [0.60, 0.40, 0.55, 0.50]
        market = [0.50, 0.50, 0.50, 0.50]
        clv = [3.0, 2.0, 1.0, None]
        blk = model_vs_market(model, market, clv, divergence_threshold=0.02)
        assert blk.n == 4
        assert blk.n_divergent == 3  # the 0.50==0.50 one is not divergent
        assert blk.clv_when_divergent == 2.0  # mean of 3,2,1
        assert blk.divergent_beat_close_rate == 1.0

    def test_model_vs_market_empty(self):
        from omega.historical.metrics import model_vs_market

        blk = model_vs_market([], [], [])
        assert blk.n == 0


def _sel_full(
    *, market, descriptor, calibrated_prob, decision_odds, trace_id="t1", event_id="e1"
) -> ReplayCandidateSelection:
    return ReplayCandidateSelection(
        replay_id="r",
        event_id=event_id,
        trace_id=trace_id,
        market=market,
        selection_descriptor=descriptor,
        raw_prob=0.5,
        calibrated_prob=calibrated_prob,
        decision_odds=decision_odds,
        decision_time="2023-01-01T00:00:00+00:00",
        stake_amount=100.0,
    )


class TestScorecardFusion:
    """Issue #28 WS5: the per-plane scorecard that fuses calibration + betting + edge."""

    def test_selection_plane_bridges_market_to_calibration_plane(self):
        cases = {
            ("moneyline", "home"): "game",
            ("moneyline", "away"): "game",
            ("moneyline", "draw"): "draw",
            ("spread", "home_-3.5"): "cover",
            ("total", "over_45.5"): "over",
            ("total", "under_45.5"): "under",
            ("prop", "over"): "prop",
        }
        for (market, desc), plane in cases.items():
            sel = _sel_full(market=market, descriptor=desc, calibrated_prob=0.5, decision_odds=-110)
            assert selection_plane(sel) == plane

    def test_model_vs_market_from_selections_counts_divergence_and_clv(self):
        sels = [
            # diverges from the +100 (0.50) close, and beat the close (decision +100 vs -150)
            _sel_full(
                market="moneyline", descriptor="home", calibrated_prob=0.62, decision_odds=100
            ),
            # sits on the market — not divergent
            _sel_full(
                market="moneyline", descriptor="home", calibrated_prob=0.50, decision_odds=100,
                trace_id="t2",
            ),
        ]
        closing = {
            "t1": [{"market": "moneyline", "selection_descriptor": "home", "closing_odds": -150}]
        }
        blk = model_vs_market_from_selections(sels, closing)
        assert blk.n == 2
        assert blk.n_divergent == 1
        assert blk.clv_when_divergent is not None and blk.clv_when_divergent > 0
        assert blk.divergent_beat_close_rate == 1.0

    def test_clv_coherent_flags_unearned_divergence(self):
        # Materially divergent yet did not beat the close → incoherent.
        incoherent = ModelVsMarketBlock(n=20, n_divergent=8, clv_when_divergent=-0.4)
        assert clv_coherent(incoherent) is False
        # Divergent AND beat the close → coherent.
        earned = ModelVsMarketBlock(n=20, n_divergent=8, clv_when_divergent=0.3)
        assert clv_coherent(earned) is True

    def test_clv_coherent_true_on_thin_or_missing_evidence(self):
        thin = ModelVsMarketBlock(n=20, n_divergent=2, clv_when_divergent=-0.9)
        assert clv_coherent(thin) is True  # too few divergent decisions to disprove
        no_clv = ModelVsMarketBlock(n=20, n_divergent=8, clv_when_divergent=None)
        assert clv_coherent(no_clv) is True  # no closing-line data

    def test_build_scorecard_fuses_blocks_and_excludes_slices(self):
        metrics = {
            "game": MetricBlock(
                raw_ece=0.10, calibrated_ece=0.05, raw_brier=0.25, calibrated_brier=0.20, n=100
            ),
            "game:playoff": MetricBlock(n=10),  # slice key — must NOT become a scorecard row
        }
        betting = {"game": BettingBlock(n_bets=20, roi=0.05, avg_clv=1.0, hit_rate=0.55, max_drawdown=3.0)}
        mvm = {
            "game": ModelVsMarketBlock(
                n=20, n_divergent=8, mean_signed_divergence=0.02, clv_when_divergent=-0.1,
                divergent_beat_close_rate=0.3,
            )
        }
        rows = build_scorecard(metrics, betting, mvm)
        assert [r.market for r in rows] == ["game"]  # slice excluded
        row = rows[0]
        assert row.n_calibrated == 100
        assert row.calibrated_ece == 0.05
        assert row.n_bets == 20
        assert row.roi == 0.05
        assert row.clv_coherent is False  # diverged materially, did not beat the close
