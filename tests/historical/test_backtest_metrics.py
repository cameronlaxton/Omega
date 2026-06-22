"""Metrics: raw-vs-calibrated separation, betting ROI/CLV, health rates."""

from __future__ import annotations

from omega.historical.contracts import ReplayCandidateSelection, ReplayEventRecord
from omega.historical.metrics import betting_metrics, health_metrics, probability_metrics


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
