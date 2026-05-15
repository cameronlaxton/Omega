"""
Tests for omega.trace.clv — CLV computation math.

Covers:
- American/decimal/implied-probability conversions on both sides of even.
- compute_clv on a "beat the close" case (got a better price).
- compute_clv on a "lost the close" case.
- Line movement scoring for Over/Under and home/away spread.
- Sanity: pushing identical inputs returns clv_cents == 0.
"""
from __future__ import annotations

import math

import pytest

from omega.trace.clv import (
    american_to_decimal,
    american_to_implied_prob,
    compute_clv,
)


class TestConversions:
    def test_positive_american_to_decimal(self):
        assert american_to_decimal(100) == pytest.approx(2.0)
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_negative_american_to_decimal(self):
        assert american_to_decimal(-100) == pytest.approx(2.0)
        assert american_to_decimal(-110) == pytest.approx(1.9090909, abs=1e-5)
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_implied_prob(self):
        assert american_to_implied_prob(-110) == pytest.approx(0.5238095, abs=1e-5)
        assert american_to_implied_prob(+110) == pytest.approx(0.47619, abs=1e-5)
        assert american_to_implied_prob(-200) == pytest.approx(2 / 3, abs=1e-5)
        assert american_to_implied_prob(+100) == pytest.approx(0.5)

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            american_to_decimal(0)
        with pytest.raises(ValueError):
            american_to_implied_prob(0)


class TestComputeCLV:
    def test_beat_close(self):
        # You took +120 (implied 45.45%). Market closed at -120 (implied 54.55%).
        # You beat the close: lower implied prob taken, higher implied at close.
        result = compute_clv(odds_taken=+120, closing_odds=-120)
        assert result.beat_close is True
        assert result.clv_cents > 0
        assert result.implied_taken < result.implied_closing

    def test_lost_close(self):
        # You took -120. Market closed at +120. Bad bet.
        result = compute_clv(odds_taken=-120, closing_odds=+120)
        assert result.beat_close is False
        assert result.clv_cents < 0

    def test_identical_close(self):
        result = compute_clv(odds_taken=-110, closing_odds=-110)
        assert math.isclose(result.clv_cents, 0.0, abs_tol=1e-9)
        assert result.beat_close is False  # not strictly better

    def test_decimal_gain(self):
        # +150 decimal = 2.5; -150 decimal = 1.6667
        result = compute_clv(odds_taken=+150, closing_odds=-150)
        assert result.decimal_taken == pytest.approx(2.5)
        assert result.decimal_closing == pytest.approx(1.6667, abs=1e-3)
        assert result.decimal_gain > 0


class TestLineValue:
    def test_over_line_dropped_is_favorable(self):
        # Took Over 226.5, line closed at 224.5 → favorable for Over (lower line)
        result = compute_clv(
            odds_taken=-110, closing_odds=-110,
            line_taken=226.5, closing_line=224.5,
            side="over",
        )
        assert result.line_value == pytest.approx(2.0)

    def test_under_line_rose_is_favorable(self):
        # Took Under 226.5, line closed at 228.5 → favorable for Under (higher line)
        result = compute_clv(
            odds_taken=-110, closing_odds=-110,
            line_taken=226.5, closing_line=228.5,
            side="under",
        )
        assert result.line_value == pytest.approx(2.0)

    def test_home_spread_negative_close_is_favorable(self):
        # Home took -3.5; closed at -5.5 → home line got steeper (worse for home),
        # so favorable line_value is the magnitude shift you BEAT. Took -3.5 vs
        # closing -5.5: you took the shorter spread (better) → favorable.
        result = compute_clv(
            odds_taken=-110, closing_odds=-110,
            line_taken=-3.5, closing_line=-5.5,
            side="home",
        )
        # delta = -5.5 - (-3.5) = -2.0; for "home" side our convention is -delta
        # so line_value = 2.0 (favorable: you beat the 2 points)
        assert result.line_value == pytest.approx(2.0)

    def test_line_value_requires_side(self):
        result = compute_clv(
            odds_taken=-110, closing_odds=-110,
            line_taken=226.5, closing_line=224.5,
            side=None,
        )
        assert result.line_value is None
