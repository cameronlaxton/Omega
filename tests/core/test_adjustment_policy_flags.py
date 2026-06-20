"""Backward-compatibility tests for the Issue #22 Phase 1 scaffolding.

Phase 1 adds three feature flags to ``AdjustmentPolicy`` and a ``damping_family``
field to ``SignalSpec``. The guarantee is: a policy persisted before these
fields existed parses unchanged and behaves bit-identically, because every flag
defaults to ``False`` and no engine path reads one yet.
"""

from __future__ import annotations

import warnings

import pytest

from omega.core.calibration.adjustment_policy import (
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.contracts.evidence import (
    SIGNAL_REGISTRY,
    EvidenceSignal,
    damping_family_for,
)
from omega.core.simulation.evidence_handlers import compute_player_adjustment

# ---------------------------------------------------------------------------
# Feature flags default off and parse from legacy (flag-less) JSON
# ---------------------------------------------------------------------------


class TestFeatureFlagDefaults:
    def test_seed_production_policy_has_all_flags_off(self):
        policy = AdjustmentPolicyRegistry().get_production_policy()
        assert policy is not None
        assert policy.enable_confidence_weighting is False
        assert policy.enable_correlation_damping is False
        assert policy.enable_competition_strength_index is False

    def test_legacy_dict_without_flags_parses_with_defaults(self):
        legacy = {
            "policy_id": "adj_legacy",
            "schema_version": 1,
            "version": 1,
            "status": "production",
            "mode": "shadow",
            "coefficients": {"usage_spike": {"scale": 1.0, "cap": 0.22}},
        }
        policy = AdjustmentPolicy(**legacy)
        assert policy.enable_confidence_weighting is False
        assert policy.enable_correlation_damping is False
        assert policy.enable_competition_strength_index is False
        # Phase 3 damping/cap params also default to behaviour-preserving values.
        assert policy.correlation_damping_weight == 0.5
        assert policy.family_cap is None
        assert policy.plane_cap is None

    def test_flags_round_trip_through_model_dump(self):
        policy = AdjustmentPolicy(
            policy_id="adj_flags",
            version=1,
            enable_confidence_weighting=True,
            enable_correlation_damping=True,
        )
        dumped = policy.model_dump()
        assert dumped["enable_confidence_weighting"] is True
        assert dumped["enable_correlation_damping"] is True
        assert dumped["enable_competition_strength_index"] is False
        reloaded = AdjustmentPolicy(**dumped)
        assert reloaded == policy


# ---------------------------------------------------------------------------
# No-behavior-change: the seed policy still computes the legacy factor
# ---------------------------------------------------------------------------


def _usage_signal() -> EvidenceSignal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(
            signal_type="usage_spike",
            category="player_form",
            plane="player",
            value=0.20,
            source="agent_reasoning",
            confidence=0.7,
            window="matchup",
        )


def test_seed_policy_factor_is_unchanged_by_scaffolding():
    policy = AdjustmentPolicyRegistry().get_production_policy()
    adj = compute_player_adjustment(
        player_context={"pts_mean": 25.0},
        evidence=[_usage_signal()],
        league="NBA",
        prop_type="pts",
        policy=policy,
        evidence_mode="live",
    )
    # usage_spike value 0.20, scale 1.0, reliability default 1.0 -> raw 1.20,
    # within the 0.22 cap. Confidence weighting is NOT applied (flag off), so the
    # 0.7 confidence does not shrink the factor.
    assert adj.records[0].factor == pytest.approx(1.20)
    assert adj.mean_factor == pytest.approx(1.20)


# ---------------------------------------------------------------------------
# damping_family registry field
# ---------------------------------------------------------------------------


class TestDampingFamilyRegistry:
    def test_seeded_families_are_grouped(self):
        assert damping_family_for("recent_form") == "player_recency"
        assert damping_family_for("series_avg") == "player_recency"
        assert damping_family_for("home_away_split") == "player_recency"
        assert damping_family_for("pace_up") == "pace"
        assert damping_family_for("pace_down") == "pace"
        assert damping_family_for("def_matchup_weak") == "def_matchup"
        assert damping_family_for("def_matchup_strong") == "def_matchup"
        assert damping_family_for("rest_advantage") == "rest_fatigue"
        assert damping_family_for("b2b_fatigue") == "rest_fatigue"

    def test_ungrouped_signal_is_a_singleton(self):
        assert damping_family_for("usage_spike") is None

    def test_unknown_signal_is_a_singleton(self):
        assert damping_family_for("not_a_real_signal") is None

    def test_default_field_is_none(self):
        # Most specs declare no family; the field exists and defaults to None.
        ungrouped = [
            s for s in SIGNAL_REGISTRY.values() if s.damping_family is None
        ]
        assert ungrouped, "expected most signals to remain singletons in Phase 1"
