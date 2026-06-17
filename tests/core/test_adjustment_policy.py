"""
Tests for the versioned engine adjustment policy artifact (Phase B).

Covers the AdjustmentPolicy model, the JSON-file registry + promotion
workflow, and the hand-seeded v1 production policy.
"""

from __future__ import annotations

import tempfile

import pytest

from omega.core.calibration.adjustment_policy import (
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.calibration.profiles import ProfileStatus


def _tmp_registry() -> AdjustmentPolicyRegistry:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    # Start from an empty file path so the registry is pristine.
    import os

    os.unlink(tmp.name)
    return AdjustmentPolicyRegistry(path=tmp.name)


def _policy(policy_id: str, **overrides) -> AdjustmentPolicy:
    base = dict(
        policy_id=policy_id,
        version=1,
        coefficients={"usage_spike": {"scale": 1.0, "cap": 0.2}},
    )
    base.update(overrides)
    return AdjustmentPolicy(**base)


class TestSeedPolicy:
    """The checked-in adjustment_policies.json seed."""

    def test_seed_loads_as_production(self):
        policy = AdjustmentPolicyRegistry().get_production_policy()
        assert policy is not None
        assert policy.policy_id == "adj_v1_seed"
        assert policy.version == 1
        assert policy.status is ProfileStatus.PRODUCTION

    def test_seed_is_shadow_mode(self):
        # Behavior-neutral: shadow means nothing reaches live predictions.
        policy = AdjustmentPolicyRegistry().get_production_policy()
        assert policy.mode == "shadow"

    def test_seed_b2b_transcribes_legacy_constants(self):
        # b2b_fatigue must match the legacy _B2B_FATIGUE table verbatim.
        policy = AdjustmentPolicyRegistry().get_production_policy()
        coeffs = policy.coeffs_for("b2b_fatigue")
        assert coeffs["by_league"]["NBA"] == 0.94
        assert coeffs["by_league"]["NHL"] == 0.95

    def test_seed_covers_all_registry_signals(self):
        from omega.core.contracts.evidence import SIGNAL_REGISTRY

        policy = AdjustmentPolicyRegistry().get_production_policy()
        for signal_type in SIGNAL_REGISTRY:
            assert signal_type in policy.coefficients, (
                f"seed policy missing coefficients for {signal_type!r}"
            )

    def test_every_coefficient_has_a_cap(self):
        policy = AdjustmentPolicyRegistry().get_production_policy()
        for signal_type, coeffs in policy.coefficients.items():
            assert "cap" in coeffs, f"{signal_type!r} coefficient missing 'cap'"
            assert coeffs["cap"] > 0


class TestPolicyModel:
    def test_coeffs_for_unknown_returns_empty(self):
        policy = _policy("p1")
        assert policy.coeffs_for("not_a_signal") == {}

    def test_coeffs_for_returns_copy(self):
        policy = _policy("p1")
        c = policy.coeffs_for("usage_spike")
        c["scale"] = 999
        assert policy.coeffs_for("usage_spike")["scale"] == 1.0

    def test_version_must_be_positive(self):
        with pytest.raises(ValueError):
            AdjustmentPolicy(policy_id="bad", version=0)


class TestRegistryWorkflow:
    def test_register_and_get(self):
        reg = _tmp_registry()
        reg.register(_policy("p1"))
        assert reg.get_policy("p1") is not None

    def test_duplicate_id_rejected(self):
        reg = _tmp_registry()
        reg.register(_policy("p1"))
        with pytest.raises(ValueError):
            reg.register(_policy("p1"))

    def test_no_production_when_all_candidates(self):
        reg = _tmp_registry()
        reg.register(_policy("p1"))
        assert reg.get_production_policy() is None

    def test_promote_makes_production(self):
        reg = _tmp_registry()
        reg.register(_policy("p1"))
        reg.promote("p1")
        prod = reg.get_production_policy()
        assert prod is not None and prod.policy_id == "p1"
        assert prod.promoted_at is not None

    def test_promote_archives_incumbent(self):
        reg = _tmp_registry()
        reg.register(_policy("p1"))
        reg.promote("p1")
        reg.register(_policy("p2", version=2))
        reg.promote("p2")
        assert reg.get_production_policy().policy_id == "p2"
        assert reg.get_policy("p1").status is ProfileStatus.ARCHIVED

    def test_cannot_promote_non_candidate(self):
        reg = _tmp_registry()
        reg.register(_policy("p1"))
        reg.promote("p1")
        with pytest.raises(ValueError):
            reg.promote("p1")  # already production

    def test_reject_with_reason(self):
        reg = _tmp_registry()
        reg.register(_policy("p1"))
        reg.reject("p1", "fails backtest parity")
        rejected = reg.get_policy("p1")
        assert rejected.status is ProfileStatus.REJECTED
        assert rejected.reject_reason == "fails backtest parity"

    def test_list_filters_by_status(self):
        reg = _tmp_registry()
        reg.register(_policy("p1"))
        reg.register(_policy("p2", version=2))
        reg.promote("p1")
        prod = reg.list_policies(status="production")
        assert [p.policy_id for p in prod] == ["p1"]

    def test_round_trip_serialization(self):
        reg = _tmp_registry()
        original = _policy("p1", mode="live", notes="round trip", sample_size=42)
        reg.register(original)
        loaded = reg.get_policy("p1")
        assert loaded.mode == "live"
        assert loaded.notes == "round trip"
        assert loaded.sample_size == 42
        assert loaded.coefficients == original.coefficients
