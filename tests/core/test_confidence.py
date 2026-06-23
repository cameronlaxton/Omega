"""Honest multi-factor confidence (Issue 5 + remediation proofs).

Pins the contract that confidence is earned from calibration + evidence + trace
quality, never from Monte-Carlo iteration count alone.
"""

from __future__ import annotations

from omega.core.contracts.confidence import (
    assign_confidence,
    cap_tier,
    combine_trace_caps,
    most_restrictive_constraint,
)


def _a_grade_inputs(**over):
    """Inputs that DO earn an A unless a specific factor is degraded."""
    base = dict(
        edge_pct=8.0,
        ev_pct=5.0,
        calibration_path="profile",
        profile_maturity="production",
        profile_sample_size=500,
        profile_ece=0.03,
        n_iterations=5000,
    )
    base.update(over)
    return base


class TestIterationsNeverCreateA:
    def test_1000_iterations_alone_never_creates_a(self):
        # The exact dishonesty being removed: many draws, nothing else.
        res = assign_confidence(
            edge_pct=8.0,
            ev_pct=5.0,
            calibration_path="static_identity",  # no real profile
            profile_maturity=None,
            profile_sample_size=None,
            profile_ece=None,
            n_iterations=1000,
        )
        assert res.tier != "A"

    def test_high_iterations_with_weak_profile_not_a(self):
        res = assign_confidence(**_a_grade_inputs(n_iterations=100000, profile_ece=0.20))
        assert res.tier != "A"
        assert res.cap_reason == "profile_ece_above_floor"

    def test_full_grade_earns_a(self):
        # Sanity: when everything is right, A is reachable (so the gate is not
        # vacuously impossible).
        res = assign_confidence(**_a_grade_inputs())
        assert res.tier == "A"
        assert res.cap_reason is None

    def test_insufficient_iterations_caps_down(self):
        res = assign_confidence(**_a_grade_inputs(n_iterations=100))
        assert res.tier == "B"
        assert res.cap_reason == "insufficient_iterations"


class TestProfileGates:
    def test_no_profile_calibration_caps_to_b(self):
        res = assign_confidence(**_a_grade_inputs(calibration_path="static_calibrated"))
        assert res.tier == "B"
        assert res.cap_reason == "no_production_profile_calibration"

    def test_provisional_maturity_caps_to_b(self):
        res = assign_confidence(**_a_grade_inputs(profile_maturity="provisional"))
        assert res.tier == "B"
        assert res.cap_reason == "profile_maturity_not_production"

    def test_small_sample_caps_to_b(self):
        res = assign_confidence(**_a_grade_inputs(profile_sample_size=40))
        assert res.tier == "B"
        assert res.cap_reason == "profile_sample_size_below_floor"


class TestEdgeAndEv:
    def test_sub_threshold_edge_is_pass(self):
        res = assign_confidence(**_a_grade_inputs(edge_pct=1.0))
        assert res.tier == "Pass"
        assert res.cap_reason == "edge_below_threshold"

    def test_negative_ev_is_pass(self):
        res = assign_confidence(**_a_grade_inputs(ev_pct=-2.0))
        assert res.tier == "Pass"
        assert res.cap_reason == "negative_ev"


class TestStage2TraceCaps:
    def test_trace_quality_cap_binds(self):
        caps = combine_trace_caps(
            trace_confidence_cap="C",
            evidence_mode="score_only",
            evidence_metrics_passed=False,
            imputed_fraction=None,
        )
        binding = most_restrictive_constraint(caps)
        assert binding == ("C", "trace_quality_cap")
        assert cap_tier("A", binding[0]) == "C"

    def test_bounded_live_without_passing_metrics_caps_b(self):
        caps = combine_trace_caps(
            trace_confidence_cap=None,
            evidence_mode="bounded_live",
            evidence_metrics_passed=False,
            imputed_fraction=None,
        )
        binding = most_restrictive_constraint(caps)
        assert binding == ("B", "evidence_metrics_unproven")

    def test_bounded_live_with_passing_metrics_no_cap(self):
        caps = combine_trace_caps(
            trace_confidence_cap=None,
            evidence_mode="bounded_live",
            evidence_metrics_passed=True,
            imputed_fraction=None,
        )
        assert most_restrictive_constraint(caps) is None

    def test_high_imputation_forces_pass(self):
        caps = combine_trace_caps(
            trace_confidence_cap=None,
            evidence_mode="score_only",
            evidence_metrics_passed=False,
            imputed_fraction=0.5,
        )
        binding = most_restrictive_constraint(caps)
        assert binding == ("Pass", "insufficient_real_observations")

    def test_most_restrictive_wins(self):
        caps = combine_trace_caps(
            trace_confidence_cap="C",
            evidence_mode="bounded_live",
            evidence_metrics_passed=False,  # would be B
            imputed_fraction=None,
        )
        # C (rank 1) is more restrictive than B (rank 2).
        assert most_restrictive_constraint(caps)[0] == "C"
