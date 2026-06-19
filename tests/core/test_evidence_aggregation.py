"""Unit tests for the pure evidence-aggregation helpers (Issue #22, Phase 1).

These lock the factor-sequence math and the sign-preserving family damping that
later phases wire in behind policy flags. Nothing here exercises a live engine
path — the helpers are pure transforms.
"""

from __future__ import annotations

import math

import pytest

from omega.core.simulation.evidence_aggregation import (
    FamilyMember,
    SignalApplication,
    cap_factor,
    confidence_adjusted_factor,
    damp_family,
    is_finite_factor,
    per_signal_capped_factor,
    plane_aggregate,
    reliability_adjusted_factor,
)

# ---------------------------------------------------------------------------
# cap_factor — must match the legacy evidence_handlers._cap_factor exactly
# ---------------------------------------------------------------------------


class TestCapFactor:
    def test_nonpositive_cap_collapses_to_identity(self):
        assert cap_factor(1.5, 0.0) == 1.0
        assert cap_factor(0.5, -0.1) == 1.0

    def test_clamps_upper_and_lower(self):
        assert cap_factor(1.30, 0.15) == pytest.approx(1.15)
        assert cap_factor(0.50, 0.15) == pytest.approx(0.85)

    def test_passthrough_within_band(self):
        assert cap_factor(1.07, 0.15) == pytest.approx(1.07)

    def test_is_the_same_object_as_handler_alias(self):
        # The handler re-exports this canonical implementation; one source of truth.
        from omega.core.simulation.evidence_handlers import _cap_factor

        assert _cap_factor is cap_factor


# ---------------------------------------------------------------------------
# reliability / confidence weighting (sequence steps 2 and 6)
# ---------------------------------------------------------------------------


class TestReliabilityAdjusted:
    def test_weight_zero_is_identity(self):
        assert reliability_adjusted_factor(1.20, 0.0) == 1.0

    def test_weight_one_is_raw(self):
        assert reliability_adjusted_factor(1.20, 1.0) == pytest.approx(1.20)

    def test_weight_half_is_halfway(self):
        assert reliability_adjusted_factor(1.20, 0.5) == pytest.approx(1.10)

    def test_suppression_sign_preserved(self):
        assert reliability_adjusted_factor(0.80, 0.5) == pytest.approx(0.90)


class TestConfidenceAdjusted:
    def test_confidence_zero_is_identity(self):
        assert confidence_adjusted_factor(1.20, 0.0) == 1.0

    def test_confidence_one_is_unchanged(self):
        assert confidence_adjusted_factor(1.20, 1.0) == pytest.approx(1.20)

    def test_confidence_half_shrinks_toward_one(self):
        assert confidence_adjusted_factor(1.20, 0.5) == pytest.approx(1.10)
        assert confidence_adjusted_factor(0.80, 0.5) == pytest.approx(0.90)


class TestPerSignalCapped:
    def test_composes_reliability_then_cap(self):
        # raw 1.40, reliability 1.0 -> 1.40, capped at 0.15 -> 1.15
        assert per_signal_capped_factor(1.40, 1.0, 0.15) == pytest.approx(1.15)
        # raw 1.40, reliability 0.5 -> 1.20, within 0.25 cap -> 1.20
        assert per_signal_capped_factor(1.40, 0.5, 0.25) == pytest.approx(1.20)


class TestPlaneAggregate:
    def test_empty_is_identity(self):
        assert plane_aggregate([]) == 1.0

    def test_product(self):
        assert plane_aggregate([1.10, 0.90, 1.05]) == pytest.approx(1.10 * 0.90 * 1.05)


# ---------------------------------------------------------------------------
# damp_family — sign-preserving co-occurrence damping (sequence step 4)
# ---------------------------------------------------------------------------


class TestDampFamily:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            damp_family([], damping_weight=0.5)

    def test_singleton_passes_through(self):
        res = damp_family([FamilyMember("recent_form", 1.12)], damping_weight=0.5)
        assert res.family_damped_factor == pytest.approx(1.12)
        assert res.roles == {"recent_form": "singleton"}
        assert res.primary_key == "recent_form"
        assert res.family_size == 1

    def test_primary_selected_by_abs_deviation(self):
        members = [
            FamilyMember("a", 1.05),
            FamilyMember("b", 0.80),  # largest |delta| -> primary
            FamilyMember("c", 1.02),
        ]
        res = damp_family(members, damping_weight=0.5)
        assert res.primary_key == "b"
        assert res.roles == {"a": "secondary", "b": "primary", "c": "secondary"}

    def test_positive_family_stays_a_boost(self):
        members = [FamilyMember("a", 1.10), FamilyMember("b", 1.06)]
        res = damp_family(members, damping_weight=0.5)
        # primary 1.10 (delta .10), secondary .06 * 0.5 -> 1.0 + .10 + .03 = 1.13
        assert res.family_damped_factor == pytest.approx(1.13)
        assert res.family_damped_factor > 1.0

    def test_negative_family_stays_a_suppression(self):
        members = [FamilyMember("a", 0.90), FamilyMember("b", 0.94)]
        res = damp_family(members, damping_weight=0.5)
        # primary 0.90 (delta -.10), secondary -.06 * 0.5 -> 1.0 - .10 - .03 = 0.87
        assert res.family_damped_factor == pytest.approx(0.87)
        assert res.family_damped_factor < 1.0

    def test_damping_weight_one_is_full_signed_sum(self):
        members = [FamilyMember("a", 1.10), FamilyMember("b", 1.06)]
        res = damp_family(members, damping_weight=1.0)
        # 1.0 + .10 + .06 = 1.16 (no damping of the secondary delta)
        assert res.family_damped_factor == pytest.approx(1.16)

    def test_damping_weight_zero_keeps_only_primary(self):
        members = [FamilyMember("a", 1.10), FamilyMember("b", 1.06)]
        res = damp_family(members, damping_weight=0.0)
        assert res.family_damped_factor == pytest.approx(1.10)

    def test_mixed_sign_family_uses_signed_deltas(self):
        # primary is the boost (|.12| > |.05|); the suppressing secondary pulls back.
        members = [FamilyMember("a", 1.12), FamilyMember("b", 0.95)]
        res = damp_family(members, damping_weight=0.5)
        # 1.0 + .12 + (-.05 * 0.5) = 1.095
        assert res.family_damped_factor == pytest.approx(1.095)
        assert res.primary_key == "a"

    def test_abs_is_not_used_as_magnitude(self):
        # A purely suppressing family must never flip positive.
        members = [FamilyMember("a", 0.85), FamilyMember("b", 0.88)]
        res = damp_family(members, damping_weight=1.0)
        assert res.family_damped_factor < 1.0


# ---------------------------------------------------------------------------
# SignalApplication scaffold + misc
# ---------------------------------------------------------------------------


def test_signal_application_as_dict_carries_every_traceability_field():
    app = SignalApplication(
        signal_type="recent_form",
        target="mean",
        applied=True,
        reason="x",
        policy_version="adj_v1_seed",
        evidence_mode="live",
        raw_factor=1.20,
        reliability_weight=1.0,
        reliability_adjusted_factor=1.20,
        per_signal_capped_factor=1.15,
        damping_family="player_recency",
        family_size=2,
        family_role="primary",
        family_damped_factor=1.17,
        family_capped_factor=1.15,
        confidence=0.7,
        confidence_defaulted=False,
        confidence_adjusted_factor=1.105,
        final_applied_factor=1.105,
    )
    d = app.as_dict()
    required = {
        "raw_factor",
        "reliability_weight",
        "reliability_adjusted_factor",
        "per_signal_capped_factor",
        "damping_family",
        "family_size",
        "family_role",
        "family_damped_factor",
        "family_capped_factor",
        "confidence",
        "confidence_defaulted",
        "confidence_adjusted_factor",
        "final_applied_factor",
    }
    assert required.issubset(d.keys())
    assert d["family_role"] == "primary"
    assert d["damping_family"] == "player_recency"


def test_is_finite_factor():
    assert is_finite_factor(1.0)
    assert not is_finite_factor(math.inf)
    assert not is_finite_factor(math.nan)
