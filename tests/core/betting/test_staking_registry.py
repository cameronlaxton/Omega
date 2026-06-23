"""Tests for the staking-policy registry (Stage C PR2)."""

from __future__ import annotations

import pytest

from omega.core.betting.staking_policy import (
    CappedFractionalKelly,
    FlatKelly,
    FractionalKellyByTier,
    StakingContext,
)
from omega.core.betting.staking_registry import (
    ANY,
    STATUS_PRODUCTION,
    StakingPolicyEntry,
    StakingRegistry,
    build_policy,
    default_policy,
)


@pytest.fixture
def registry(tmp_path):
    return StakingRegistry(path=str(tmp_path / "staking_policies.json"))


def _entry(
    policy_id="flat_kelly", league="NBA", market="prop", version=1, params=None, status="candidate"
):
    return StakingPolicyEntry(
        entry_id=StakingPolicyEntry.make_entry_id(policy_id, league, market, version),
        policy_id=policy_id,
        version=version,
        league=league,
        market=market,
        status=status,
        params=params or {},
    )


# --- default / empty registry --------------------------------------------------
def test_empty_registry_returns_default_policy(registry):
    policy = registry.get_production("NBA", "prop")
    assert isinstance(policy, FractionalKellyByTier)
    # behaves identically to the module default
    ctx = StakingContext(true_prob=0.62, odds=-110, bankroll=1000.0, confidence_tier="A")
    assert policy.size(ctx) == default_policy().size(ctx)


# --- build_policy factory ------------------------------------------------------
def test_build_policy_known_ids():
    assert isinstance(build_policy("fractional_kelly_by_tier"), FractionalKellyByTier)
    assert isinstance(build_policy("flat_kelly", {"multiplier": 0.3}), FlatKelly)
    assert isinstance(
        build_policy("capped_fractional_kelly", {"max_kelly_fraction": 0.02}), CappedFractionalKelly
    )


def test_build_policy_unknown_id_raises():
    with pytest.raises(ValueError, match="Unknown staking policy_id"):
        build_policy("does_not_exist")


def test_register_rejects_bad_policy(registry):
    bad = _entry(policy_id="nonsense")
    with pytest.raises(ValueError, match="Unknown staking policy_id"):
        registry.register(bad)


# --- register + activate + select ----------------------------------------------
def test_register_then_activate_selects_policy(registry):
    e = _entry(policy_id="flat_kelly", league="NBA", market="prop", params={"multiplier": 0.3})
    registry.register(e)
    # still candidate -> get_production falls back to default
    assert isinstance(registry.get_production("NBA", "prop"), FractionalKellyByTier)
    registry.activate(e.entry_id)
    selected = registry.get_production("NBA", "prop")
    assert isinstance(selected, FlatKelly)


def test_duplicate_entry_id_rejected(registry):
    e = _entry()
    registry.register(e)
    with pytest.raises(ValueError, match="already exists"):
        registry.register(_entry())


def test_activate_archives_prior_production_same_slot(registry):
    e1 = _entry(policy_id="flat_kelly", version=1)
    e2 = _entry(policy_id="capped_fractional_kelly", version=2)
    registry.register(e1)
    registry.register(e2)
    registry.activate(e1.entry_id)
    registry.activate(e2.entry_id)
    prod = registry.list_entries(league="NBA", market="prop", status=STATUS_PRODUCTION)
    assert len(prod) == 1  # one-production invariant
    assert prod[0].entry_id == e2.entry_id
    assert registry.get_entry(e1.entry_id).status == "archived"


def test_activate_does_not_touch_other_slots(registry):
    nba = _entry(policy_id="flat_kelly", league="NBA", market="prop", version=1)
    mlb = _entry(policy_id="flat_kelly", league="MLB", market="prop", version=1)
    registry.register(nba)
    registry.register(mlb)
    registry.activate(nba.entry_id)
    registry.activate(mlb.entry_id)
    # both remain production in their own slots
    assert registry.get_entry(nba.entry_id).status == STATUS_PRODUCTION
    assert registry.get_entry(mlb.entry_id).status == STATUS_PRODUCTION


# --- fallback chain ------------------------------------------------------------
def test_market_wildcard_fallback(registry):
    e = _entry(policy_id="flat_kelly", league="NBA", market=ANY, version=1)
    registry.register(e)
    registry.activate(e.entry_id)
    # no exact (NBA, game) -> falls back to (NBA, ANY)
    assert isinstance(registry.get_production("NBA", "game"), FlatKelly)


def test_exact_slot_beats_wildcard(registry):
    wild = _entry(policy_id="flat_kelly", league="NBA", market=ANY, version=1)
    exact = _entry(policy_id="capped_fractional_kelly", league="NBA", market="prop", version=1)
    for e in (wild, exact):
        registry.register(e)
        registry.activate(e.entry_id)
    assert isinstance(registry.get_production("NBA", "prop"), CappedFractionalKelly)
    assert isinstance(registry.get_production("NBA", "game"), FlatKelly)


def test_global_wildcard_fallback(registry):
    e = _entry(policy_id="flat_kelly", league=ANY, market=ANY, version=1)
    registry.register(e)
    registry.activate(e.entry_id)
    assert isinstance(registry.get_production("ANYLEAGUE", "anymarket"), FlatKelly)


# --- reject + serialization ----------------------------------------------------
def test_reject_records_reason(registry):
    e = _entry()
    registry.register(e)
    registry.reject(e.entry_id, "failed backtest")
    got = registry.get_entry(e.entry_id)
    assert got.status == "rejected"
    assert got.reject_reason == "failed backtest"


def test_entry_params_round_trip(registry):
    params = {"tier_multipliers": {"A": 0.4, "B": 0.2, "C": 0.05}, "unit_cap": 3.0}
    e = _entry(policy_id="fractional_kelly_by_tier", params=params)
    registry.register(e)
    got = registry.get_entry(e.entry_id)
    assert got.params == params
    # reconstructed policy honors persisted params (unit cap of 3, not 5)
    registry.activate(e.entry_id)
    policy = registry.get_production("NBA", "prop")
    decision = policy.size(
        StakingContext(true_prob=0.99, odds=250, bankroll=1000.0, confidence_tier="A")
    )
    assert decision.units == 3.0
    assert "unit_cap" in decision.capped_by


def test_persistence_across_instances(registry, tmp_path):
    e = _entry(policy_id="flat_kelly")
    registry.register(e)
    registry.activate(e.entry_id)
    # a fresh registry reading the same file sees the production entry
    reopened = StakingRegistry(path=str(tmp_path / "staking_policies.json"))
    assert isinstance(reopened.get_production("NBA", "prop"), FlatKelly)


def test_league_lookup_is_case_insensitive(registry):
    # Mirrors CalibrationRegistry: stored "NBA" must match a "nba"/"Nba" query,
    # else PR5 wiring would silently fall back to the default policy.
    e = _entry(policy_id="flat_kelly", league="NBA", market="prop")
    registry.register(e)
    registry.activate(e.entry_id)
    for q in ("nba", "NBA", "Nba"):
        assert isinstance(registry.get_production(q, "prop"), FlatKelly)
    assert len(registry.list_entries(league="nba")) == 1
