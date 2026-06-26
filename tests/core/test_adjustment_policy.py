"""
Tests for the versioned engine adjustment policy artifact (Phase B).

Covers the AdjustmentPolicy model, the JSON-file registry + promotion
workflow, and the hand-seeded v1 production policy.
"""

from __future__ import annotations

import tempfile

import pytest

from omega.core.calibration.adjustment_policy import (
    SEED_UNFITTED_RELIABILITY_PRIOR,
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

    def test_seed_is_bounded_live_mode(self):
        # Graduated-apply default: the shipped seed applies evidence under hard
        # caps (scaled by reliability), not the legacy record-only score_only.
        policy = AdjustmentPolicyRegistry().get_production_policy()
        assert policy.mode == "bounded_live"

    def test_seed_unfitted_prior_is_a_sliver(self):
        # Unscored signals move a live prediction only by this fraction of their
        # handler factor until omega-fit-adjustment-policy measures them. Pin the
        # exact shipped seed value so drift to another fraction fails loudly.
        policy = AdjustmentPolicyRegistry().get_production_policy()
        assert policy.unfitted_reliability_prior == SEED_UNFITTED_RELIABILITY_PRIOR
        assert 0.0 < SEED_UNFITTED_RELIABILITY_PRIOR < 1.0

    def test_seed_curated_signals_carry_full_reliability(self):
        # The hand-validated directional signals apply their capped prior in full
        # (reliability_weight=1.0), not the conservative unfitted prior.
        policy = AdjustmentPolicyRegistry().get_production_policy()
        for signal_type in ("b2b_fatigue", "rest_advantage", "def_matchup_weak", "def_matchup_strong"):
            assert policy.coeffs_for(signal_type).get("reliability_weight") == 1.0

    def test_seed_evidence_metrics_do_not_pass_gate(self):
        # Unfitted priors (sample_size=0, metrics={}) must not clear the gate
        # that would let bounded_live evidence lift a rec to A.
        policy = AdjustmentPolicyRegistry().get_production_policy()
        assert policy.evidence_metrics_passed() is False

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

    def test_legacy_shadow_maps_to_score_only_on_load(self):
        # A policy persisted with the legacy binary mode parses cleanly.
        policy = AdjustmentPolicy(policy_id="legacy", version=1, mode="shadow")
        assert policy.mode == "score_only"

    def test_all_graduated_modes_parse(self):
        for mode in ("disabled", "observe", "score_only", "bounded_live", "live"):
            assert AdjustmentPolicy(policy_id="m", version=1, mode=mode).mode == mode

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            AdjustmentPolicy(policy_id="bad", version=1, mode="turbo")

    def test_default_mode_is_score_only(self):
        assert AdjustmentPolicy(policy_id="d", version=1).mode == "score_only"

    def test_bounded_live_effective_forces_hard_caps(self):
        # No caps configured -> bounded_live supplies its hard defaults and
        # turns family damping on.
        eff = _policy("p1", mode="bounded_live").bounded_live_effective()
        assert eff.single_cap_ceiling == 0.10
        assert eff.family_cap == 0.15
        assert eff.plane_cap == 0.20
        assert eff.enable_correlation_damping is True

    def test_bounded_live_effective_keeps_tighter_existing_caps(self):
        eff = _policy("p1", mode="bounded_live", plane_cap=0.05).bounded_live_effective()
        assert eff.plane_cap == 0.05  # tighter than the 0.20 default is kept

    def test_evidence_metrics_passed_requires_samples_and_predictive_metrics(self):
        # No samples -> fail.
        assert _policy("p1").evidence_metrics_passed() is False
        # Samples but no metrics -> fail.
        assert _policy("p2", sample_size=200).evidence_metrics_passed() is False
        # Plenty of samples but the fitted signals are NOISE (reliability 0) -> fail.
        assert (
            _policy(
                "p3", sample_size=200, metrics={"mean_reliability_weight": 0.0}
            ).evidence_metrics_passed()
            is False
        )
        # Samples but no reliability metric recorded -> fail (cannot verify predictiveness).
        assert (
            _policy("p4", sample_size=200, metrics={"ece": 0.04}).evidence_metrics_passed()
            is False
        )
        # Enough samples AND predictive signals -> pass.
        assert (
            _policy(
                "p5", sample_size=200, metrics={"mean_reliability_weight": 0.3}
            ).evidence_metrics_passed()
            is True
        )


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
