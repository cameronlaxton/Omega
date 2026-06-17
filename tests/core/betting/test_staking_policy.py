"""Tests for the staking-policy shim (Stage C PR1).

The headline guarantee: ``recommend_stake`` now delegates to
``FractionalKellyByTier`` but its numeric result is **bit-identical** to the
legacy inline implementation. The reference implementation below is frozen here
so that any drift in the delegated path is caught by the suite.
"""

from __future__ import annotations

import pytest

from omega.core.betting.kelly import kelly_fraction, recommend_stake
from omega.core.betting.staking_policy import (
    DEFAULT_TIER_MULTIPLIERS,
    CappedFractionalKelly,
    FlatKelly,
    FractionalKellyByTier,
    StakingContext,
    StakingDecision,
    StakingPolicy,
)

# --- Frozen reference: the EXACT legacy recommend_stake math --------------------
_LEGACY_TIER = {"A": 0.50, "B": 0.25, "C": 0.10}


def _legacy_recommend_stake(true_prob, odds, bankroll, tier="B"):
    raw = kelly_fraction(true_prob, odds)
    mult = _LEGACY_TIER.get(tier.upper(), _LEGACY_TIER["B"])
    scaled = raw * mult
    units = min(scaled * 100, 5.0)
    return {"units": round(units, 2), "kelly_fraction": round(scaled, 4)}


_PROBS = [0.0, 0.40, 0.50, 0.5238, 0.55, 0.62, 0.75, 0.90, 0.99, 1.0]
_ODDS = [-100000, -200, -150, -110, -101, 100, 105, 120, 150, 250, 100000]
# Includes lowercase and unknown tiers — both must fall back to the B multiplier.
_TIERS = ["A", "B", "C", "a", "b", "c", "Pass", "Z", ""]


@pytest.mark.parametrize("true_prob", _PROBS)
@pytest.mark.parametrize("odds", _ODDS)
@pytest.mark.parametrize("tier", _TIERS)
def test_recommend_stake_bit_identical_to_legacy(true_prob, odds, tier):
    """recommend_stake (now delegating) matches the frozen legacy math exactly."""
    assert recommend_stake(true_prob, odds, 1000.0, tier) == _legacy_recommend_stake(
        true_prob, odds, 1000.0, tier
    )


@pytest.mark.parametrize("bankroll", [100.0, 1000.0, 50000.0])
def test_recommend_stake_dict_independent_of_bankroll(bankroll):
    """units / kelly_fraction do not depend on bankroll (only stake_amount does)."""
    assert recommend_stake(0.62, -110, bankroll, "A") == recommend_stake(0.62, -110, 1000.0, "A")


@pytest.mark.parametrize("true_prob", _PROBS)
@pytest.mark.parametrize("odds", _ODDS)
@pytest.mark.parametrize("tier", _TIERS)
def test_policy_matches_recommend_stake(true_prob, odds, tier):
    """FractionalKellyByTier.size mirrors recommend_stake's dict output."""
    decision = FractionalKellyByTier().size(
        StakingContext(true_prob=true_prob, odds=odds, bankroll=1000.0, confidence_tier=tier)
    )
    assert decision.to_recommend_stake_dict() == recommend_stake(true_prob, odds, 1000.0, tier)


def test_default_tier_multipliers_match_legacy():
    assert DEFAULT_TIER_MULTIPLIERS == _LEGACY_TIER


def test_stake_amount_is_units_pct_of_bankroll():
    decision = FractionalKellyByTier().size(
        StakingContext(true_prob=0.62, odds=-110, bankroll=2000.0, confidence_tier="A")
    )
    assert decision.stake_amount == pytest.approx(round(decision.units * 2000.0 / 100.0, 2))


def test_unit_cap_records_capped_by():
    # A large edge on plus-money at half Kelly blows past the 5-unit cap.
    decision = FractionalKellyByTier().size(
        StakingContext(true_prob=0.95, odds=250, bankroll=1000.0, confidence_tier="A")
    )
    assert decision.units == 5.0
    assert "unit_cap" in decision.capped_by


def test_negative_ev_sizes_to_zero():
    # true_prob below the break-even implied by the odds → Kelly 0 → 0 units.
    decision = FractionalKellyByTier().size(
        StakingContext(true_prob=0.40, odds=-110, bankroll=1000.0, confidence_tier="A")
    )
    assert decision.units == 0.0
    assert decision.kelly_fraction == 0.0
    assert decision.capped_by == ()


def test_policy_decision_is_deterministic():
    ctx = StakingContext(true_prob=0.62, odds=-110, bankroll=1000.0, confidence_tier="B")
    a = FractionalKellyByTier().size(ctx)
    b = FractionalKellyByTier().size(ctx)
    assert a == b
    assert a.policy_id == "fractional_kelly_by_tier"
    assert a.policy_version == 1


def test_staking_policy_is_abstract():
    with pytest.raises(TypeError):
        StakingPolicy()  # type: ignore[abstract]


def test_flat_kelly_uses_single_multiplier():
    ctx = StakingContext(true_prob=0.62, odds=-110, bankroll=1000.0, confidence_tier="A")
    flat = FlatKelly(multiplier=0.25).size(ctx)
    # FlatKelly(0.25) on any tier equals FractionalKellyByTier on a B-tier (0.25) bet.
    fk_b = FractionalKellyByTier().size(
        StakingContext(true_prob=0.62, odds=-110, bankroll=1000.0, confidence_tier="B")
    )
    assert flat.kelly_fraction == fk_b.kelly_fraction
    assert flat.units == fk_b.units
    assert flat.policy_id == "flat_kelly"


def test_capped_fractional_kelly_clamps_kelly_fraction():
    ctx = StakingContext(true_prob=0.95, odds=250, bankroll=1000.0, confidence_tier="A")
    uncapped = FractionalKellyByTier().size(ctx)
    capped = CappedFractionalKelly(max_kelly_fraction=0.02).size(ctx)
    assert uncapped.kelly_fraction > 0.02
    assert capped.kelly_fraction == 0.02
    assert "max_kelly_fraction" in capped.capped_by


def test_staking_decision_round_trip_dict():
    d = StakingDecision(units=1.23, kelly_fraction=0.0123, stake_amount=12.30)
    assert d.to_recommend_stake_dict() == {"units": 1.23, "kelly_fraction": 0.0123}
