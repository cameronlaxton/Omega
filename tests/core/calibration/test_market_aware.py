"""
Market-aware calibration (issue #28 WS4): the blend math, coarse liquidity-
weighted deference, the calibrate_probability dispatch + fail-safe, the
fit_market_aware grid search + guard, and the apply path with liquidity scaling.
"""

from __future__ import annotations

import pytest

from omega.core.calibration.fitter import CalibrationFitter
from omega.core.calibration.probability import (
    apply_calibration_audited,
    calibrate_probability,
    liquidity_deference_factor,
    market_aware_calibration,
)


class TestBlendMath:
    def test_weights(self):
        assert market_aware_calibration(0.70, 0.50, 0.0) == 0.70  # keep the model
        assert market_aware_calibration(0.70, 0.50, 1.0) == 0.50  # fully defer
        assert market_aware_calibration(0.70, 0.50, 0.5) == pytest.approx(0.60)

    def test_inputs_clipped(self):
        assert market_aware_calibration(1.5, -0.2, 2.0) == 0.0  # w clipped to 1, market clipped to 0


class TestDeference:
    def test_game_high_liquidity_full(self):
        assert liquidity_deference_factor("NBA", "game") == 1.0

    def test_prop_half(self):
        assert liquidity_deference_factor("NBA", "prop") == 0.5

    def test_low_liquidity_league_shrinks(self):
        assert liquidity_deference_factor("WNBA", "game") == 0.5
        assert liquidity_deference_factor("WNBA", "prop") == 0.25


class TestDispatch:
    def test_no_market_is_failsafe_identity(self):
        out = calibrate_probability(0.7, method="market_aware")
        assert out["calibrated"] == 0.7
        assert out["method"] == "market_aware_no_market"

    def test_with_market_blends(self):
        out = calibrate_probability(0.7, method="market_aware", market_prob=0.5, market_weight=0.5)
        assert out["calibrated"] == pytest.approx(0.6)
        assert out["method"] == "market_aware"


class TestFit:
    def test_defers_when_market_predicts_better(self):
        # Model is uninformative (0.5); market nails it -> w should be > 0.
        n = 60
        model = [0.5] * n
        market = [0.85 if i % 2 == 0 else 0.15 for i in range(n)]
        outcomes = [1 if i % 2 == 0 else 0 for i in range(n)]
        prof = CalibrationFitter().fit_market_aware(model, market, outcomes, "NBA")
        assert prof.method == "market_aware"
        assert prof.params["market_weight"] > 0.0

    def test_keeps_model_when_market_is_noise(self):
        # Model nails it; market is a constant coin flip -> deferring only hurts -> w=0.
        n = 60
        outcomes = [1 if i % 2 == 0 else 0 for i in range(n)]
        model = [0.9 if o else 0.1 for o in outcomes]
        market = [0.5] * n
        prof = CalibrationFitter().fit_market_aware(model, market, outcomes, "NBA")
        assert prof.params["market_weight"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            CalibrationFitter().fit_market_aware([0.5] * 60, [0.5] * 59, [1] * 60, "NBA")


def _market_aware_profile(market_weight: float):
    from omega.core.calibration.profiles import CalibrationProfile

    return CalibrationProfile(
        profile_id="market_nba_v1",
        method="market_aware",
        league="NBA",
        status="production",
        version=1,
        dataset_hash="x",
        params={"market_weight": market_weight},
        metrics={},
        training_window="365d",
        sample_size=1000,
    )


class TestApplyPath:
    def test_blends_with_liquidity_scaling(self, monkeypatch):
        from omega.core.calibration import probability

        monkeypatch.setattr(
            probability, "_get_active_profile", lambda *a, **k: _market_aware_profile(0.8)
        )
        # game market on NBA -> liquidity factor 1.0 -> w_eff = 0.8.
        cal, audit = apply_calibration_audited(0.70, league="NBA", market="game", market_prob=0.50)
        assert cal == pytest.approx(0.2 * 0.70 + 0.8 * 0.50)  # 0.54
        assert audit["method_resolved"] == "market_aware"

    def test_prop_market_defers_less(self, monkeypatch):
        from omega.core.calibration import probability

        monkeypatch.setattr(
            probability, "_get_active_profile", lambda *a, **k: _market_aware_profile(0.8)
        )
        # prop market -> liquidity factor 0.5 -> w_eff = 0.4.
        cal, _ = apply_calibration_audited(0.70, league="NBA", market="prop", market_prob=0.50)
        assert cal == pytest.approx(0.6 * 0.70 + 0.4 * 0.50)  # 0.62

    def test_failsafe_without_market_prob(self, monkeypatch):
        from omega.core.calibration import probability

        monkeypatch.setattr(
            probability, "_get_active_profile", lambda *a, **k: _market_aware_profile(0.8)
        )
        # No market_prob (existing callers) -> identity, model unchanged.
        cal, _ = apply_calibration_audited(0.70, league="NBA", market="game")
        assert cal == 0.70
