"""
Tests for the versioned engine adjustment policy artifact (Phase B).

Covers the AdjustmentPolicy model, the JSON-file registry + promotion
workflow, and the hand-seeded v1 production policy.
"""

from __future__ import annotations

import json
import os
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
        # adj_v2_seed re-versions adj_v1_seed (PR #35 remediation): the production
        # seed is now v2, while v1 is retained ARCHIVED for trace attribution.
        policy = AdjustmentPolicyRegistry().get_production_policy()
        assert policy is not None
        assert policy.policy_id == "adj_v2_seed"
        assert policy.version == 2
        assert policy.status is ProfileStatus.PRODUCTION

    def test_archived_v1_seed_preserves_score_only_attribution(self):
        # PR #35 remediation: adj_v1_seed is re-versioned rather than mutated in
        # place. The original hand-seed is retained ARCHIVED so traces stamped
        # policy_version='adj_v1_seed' stay attributable to the score_only behavior
        # they actually ran under (not the later bounded_live posture of adj_v2_seed).
        v1 = AdjustmentPolicyRegistry().get_policy("adj_v1_seed")
        assert v1 is not None
        assert v1.status is ProfileStatus.ARCHIVED
        assert v1.version == 1
        assert v1.mode == "score_only"
        # The curated directional signals did NOT carry full reliability in v1 —
        # that is an adj_v2_seed change and must not bleed back into the record.
        for signal_type in ("b2b_fatigue", "rest_advantage", "def_matchup_weak", "def_matchup_strong"):
            assert "reliability_weight" not in v1.coeffs_for(signal_type)
        # The v1 record is field-less on disk (schema 1); the registry backfills the
        # v2 prior on load (inert in score_only, which never applies it).
        assert v1.unfitted_reliability_prior == SEED_UNFITTED_RELIABILITY_PRIOR
        assert v1.schema_version == 2

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


class TestSchemaMigration:
    """Schema v1 -> v2 load-time backfill of ``unfitted_reliability_prior``."""

    @staticmethod
    def _registry_with(policy: dict) -> AdjustmentPolicyRegistry:
        """A registry whose file holds one raw (un-migrated) policy dict."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        with open(tmp.name, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "policies": [policy]}, f)
        return AdjustmentPolicyRegistry(path=tmp.name)

    @staticmethod
    def _raw(**overrides) -> dict:
        base = {
            "policy_id": "old_persisted",
            "schema_version": 1,
            "version": 1,
            "status": "production",
            "mode": "bounded_live",
            "coefficients": {"usage_spike": {"scale": 1.0, "cap": 0.2}},
        }
        base.update(overrides)
        return base

    def test_fieldless_persisted_policy_backfills_prior(self):
        # A policy persisted before the field existed must NOT deserialize at the
        # model's ad-hoc default (1.0, full trust); the registry backfills the
        # conservative seed prior on load and stamps the new schema_version.
        reg = self._registry_with(self._raw())  # no unfitted_reliability_prior
        loaded = reg.get_production_policy()
        assert loaded is not None
        assert loaded.unfitted_reliability_prior == SEED_UNFITTED_RELIABILITY_PRIOR
        assert loaded.schema_version == 2

    def test_fieldless_backfill_applies_via_get_policy_and_list(self):
        # Every read path routes through _load, so get_policy/list_policies migrate too.
        reg = self._registry_with(self._raw(status="candidate"))
        assert reg.get_policy("old_persisted").unfitted_reliability_prior == (
            SEED_UNFITTED_RELIABILITY_PRIOR
        )
        assert reg.list_policies()[0].unfitted_reliability_prior == (
            SEED_UNFITTED_RELIABILITY_PRIOR
        )

    def test_deliberate_prior_at_old_schema_not_clobbered(self):
        # A policy persisted WITH a deliberate prior (even at schema_version 1, e.g.
        # a candidate fitted post-PR#35) keeps its value: the backfill keys on field
        # absence, not on schema_version, so it never resets an intentional prior.
        reg = self._registry_with(self._raw(unfitted_reliability_prior=0.5))
        assert reg.get_production_policy().unfitted_reliability_prior == 0.5

    def test_model_default_unchanged_for_adhoc_policies(self):
        # The in-memory model default stays 1.0 (legacy parity): an ad-hoc / test
        # policy that never round-trips through the registry keeps full trust.
        assert AdjustmentPolicy(policy_id="adhoc", version=1).unfitted_reliability_prior == 1.0

    def test_backfill_persisted_lazily_on_next_save(self):
        # Migration happens in _load, so a mutation (which loads then saves) upgrades
        # the on-disk file: the field is written and schema_version advanced.
        reg = self._registry_with(self._raw(status="candidate"))
        reg.set_mode("old_persisted", "score_only")  # any mutation triggers _save
        with open(reg._path, encoding="utf-8") as f:
            on_disk = json.load(f)
        os.unlink(reg._path)
        persisted = on_disk["policies"][0]
        assert persisted["unfitted_reliability_prior"] == SEED_UNFITTED_RELIABILITY_PRIOR
        assert persisted["schema_version"] == 2
        assert on_disk["schema_version"] == 2

    def test_malformed_top_level_schema_version_defaults_to_current(self):
        reg = self._registry_with(self._raw())
        with open(reg._path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["schema_version"] = "not-an-int"
            f.seek(0)
            json.dump(data, f)
            f.truncate()

        loaded = reg._load()

        assert loaded["schema_version"] == 2
