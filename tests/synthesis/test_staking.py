"""Tests for omega.synthesis.staking — unit/dollar math consistency."""

from __future__ import annotations

import pytest

from omega.synthesis.staking import (
    DEFAULT_BANKROLL,
    DEFAULT_UNIT_PCT,
    calculate_stake,
    dollars_to_units,
    unit_to_dollars,
)


class TestUnitToDollars:
    def test_default_bankroll_default_pct(self):
        # bankroll=1000, unit_pct=0.01 → 1u = $10
        assert unit_to_dollars(1.0) == 10.0

    def test_bankroll_1000_unit_pct_001(self):
        result = unit_to_dollars(1.0, bankroll=1000.0, unit_pct=0.01)
        assert result == 10.0

    def test_bankroll_1000_unit_pct_005(self):
        # bankroll=1000, unit_pct=0.05 → 1u = $50
        result = unit_to_dollars(1.0, bankroll=1000.0, unit_pct=0.05)
        assert result == 50.0

    def test_fractional_units(self):
        # 0.5u at $10/u = $5
        assert unit_to_dollars(0.5, bankroll=1000.0, unit_pct=0.01) == 5.0

    def test_zero_units(self):
        assert unit_to_dollars(0.0) == 0.0

    def test_five_units(self):
        # 5u at $10/u = $50 (max Kelly cap)
        assert unit_to_dollars(5.0, bankroll=1000.0, unit_pct=0.01) == 50.0


class TestDollarsToUnits:
    def test_roundtrip_default(self):
        dollars = unit_to_dollars(2.5)
        units = dollars_to_units(dollars)
        assert abs(units - 2.5) < 1e-9

    def test_bankroll_1000_unit_pct_001(self):
        # $10 / ($1000 * 0.01) = 1u
        result = dollars_to_units(10.0, bankroll=1000.0, unit_pct=0.01)
        assert result == 1.0

    def test_zero_unit_pct_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            dollars_to_units(10.0, bankroll=1000.0, unit_pct=0.0)


class TestCalculateStake:
    def test_static_fallback_caps_at_1u(self):
        units, dollars = calculate_stake(
            bankroll=1000.0, unit_pct=0.01, is_static_fallback=True
        )
        assert units == 1.0
        assert units <= 1.0, "Static fallback must cap at 1u"

    def test_static_fallback_dollars_consistent(self):
        units, dollars = calculate_stake(
            bankroll=1000.0, unit_pct=0.01, is_static_fallback=True
        )
        expected_dollars = unit_to_dollars(units, 1000.0, 0.01)
        assert dollars == expected_dollars, "stake_units and stake_dollars must not contradict"

    def test_non_fallback_returns_1u(self):
        units, dollars = calculate_stake(
            bankroll=1000.0, unit_pct=0.01, is_static_fallback=False
        )
        assert units == 1.0
        assert dollars == 10.0

    def test_stake_dollars_5pct_unit(self):
        # bankroll=1000, unit_pct=0.05 → 1u = $50
        units, dollars = calculate_stake(
            bankroll=1000.0, unit_pct=0.05, is_static_fallback=False
        )
        assert units == 1.0
        assert dollars == 50.0

    def test_units_and_dollars_never_contradict(self):
        for unit_pct in [0.01, 0.02, 0.05, 0.10]:
            for is_static in [True, False]:
                units, dollars = calculate_stake(
                    bankroll=DEFAULT_BANKROLL, unit_pct=unit_pct, is_static_fallback=is_static
                )
                expected = unit_to_dollars(units, DEFAULT_BANKROLL, unit_pct)
                assert dollars == expected, (
                    f"Contradiction: {units}u != ${dollars} at unit_pct={unit_pct}"
                )
