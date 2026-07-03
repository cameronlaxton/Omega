"""Phase 4 (Issue #22) — Markov richer return object + parity.

``compute_transition_modifier_adjustment`` returns modifiers (bit-identical to
the legacy mapping when flags are off), real per-signal applications (no more
fabricated ``factor=None``), and per-key aggregation records. Confidence and
correlation damping affect the modifiers only when their flags are enabled.
"""

from __future__ import annotations

import warnings

import pytest

from omega.core.calibration.adjustment_policy import AdjustmentPolicy
from omega.core.contracts.evidence import SIGNAL_REGISTRY, EvidenceSignal
from omega.core.contracts.service import _merge_markov_applications, _suppression_record
from omega.core.simulation.evidence_to_modifier import (
    compute_transition_modifier_adjustment,
    signals_to_transition_modifiers,
)


def _policy(*, confidence=False, damping=False, weight=0.5) -> AdjustmentPolicy:
    return AdjustmentPolicy(
        policy_id="m",
        version=1,
        enable_confidence_weighting=confidence,
        enable_correlation_damping=damping,
        correlation_damping_weight=weight,
    )


def _sig(signal_type: str, *, direction=None, confidence=0.75) -> EvidenceSignal:
    spec = SIGNAL_REGISTRY[signal_type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(
            signal_type=signal_type,
            category=spec.category,
            plane="game",
            value=True,
            source="test",
            confidence=confidence,
            window=spec.default_window,
            direction=direction,
        )


# ---------------------------------------------------------------------------
# Modifiers preserve old output when flags are disabled
# ---------------------------------------------------------------------------


class TestModifierParity:
    @pytest.mark.parametrize(
        "signals",
        [
            [_sig("rest_advantage", direction="home")],
            [_sig("pace_down"), _sig("pace_down")],
            [_sig("pace_up"), _sig("def_matchup_weak")],
            [_sig("season_record")],  # unmapped -> empty modifiers
        ],
    )
    def test_rich_modifiers_match_legacy_wrapper(self, signals):
        legacy = signals_to_transition_modifiers(signals, home_team="Lakers")
        rich = compute_transition_modifier_adjustment(signals, "Lakers").modifiers
        assert rich == legacy

    def test_flags_off_policy_is_bit_identical_to_no_policy(self):
        signals = [_sig("pace_down"), _sig("pace_down")]
        no_policy = compute_transition_modifier_adjustment(signals, "Lakers").modifiers
        flags_off = compute_transition_modifier_adjustment(
            signals, "Lakers", policy=_policy()
        ).modifiers
        assert flags_off == no_policy

    def test_two_usage_signals_clamped_when_off(self):
        signals = [_sig("usage_role_change"), _sig("usage_role_change")]
        mods = compute_transition_modifier_adjustment(signals, "Lakers").modifiers
        # 0.93*0.93 = 0.8649 below the floor 1/1.15 -> clamped
        assert mods["home_score_rate_scalar"] == pytest.approx(1.0 / 1.15, abs=1e-9)


# ---------------------------------------------------------------------------
# Real applications + aggregation records
# ---------------------------------------------------------------------------


class TestApplications:
    def test_mapped_signal_carries_real_factor(self):
        adj = compute_transition_modifier_adjustment(
            [_sig("rest_advantage", direction="home")], "Lakers"
        )
        app = adj.applications[0]
        assert app["applied"] is True
        assert app["factor"] is not None
        assert app["factor"] == pytest.approx(1.04)
        assert app["effective_scalar"] == pytest.approx(1.04)
        assert app["raw_scalar"] == pytest.approx(1.04)
        assert app["modifier_key"] == "home_score_rate_scalar"
        assert app["target"] == "markov_transition"

    def test_unmapped_signal_is_skip(self):
        adj = compute_transition_modifier_adjustment([_sig("season_record")], "Lakers")
        app = adj.applications[0]
        assert app["applied"] is False
        assert app["factor"] == pytest.approx(1.0)
        assert app["target"] == "skip"

    def test_aggregation_records_describe_each_key(self):
        adj = compute_transition_modifier_adjustment(
            [_sig("usage_role_change"), _sig("usage_role_change")], "Lakers"
        )
        assert len(adj.aggregation_records) == 1
        rec = adj.aggregation_records[0]
        assert rec["modifier_key"] == "home_score_rate_scalar"
        assert rec["family_size"] == 2
        assert rec["damped"] is False
        assert rec["raw_value"] == pytest.approx(0.93 * 0.93)
        assert rec["clamped_value"] == pytest.approx(1.0 / 1.15, abs=1e-9)


# ---------------------------------------------------------------------------
# Flags enabled change the modifier math
# ---------------------------------------------------------------------------


class TestFlagsEnabled:
    def test_confidence_scales_scalar(self):
        adj = compute_transition_modifier_adjustment(
            [_sig("rest_advantage", direction="home", confidence=0.5)],
            "Lakers",
            policy=_policy(confidence=True),
        )
        # 1 + 0.5*(1.04 - 1) = 1.02
        assert adj.modifiers["home_score_rate_scalar"] == pytest.approx(1.02)
        assert adj.applications[0]["effective_scalar"] == pytest.approx(1.02)

    def test_damping_replaces_product_for_same_key(self):
        signals = [_sig("usage_role_change"), _sig("usage_role_change")]
        adj = compute_transition_modifier_adjustment(
            signals, "Lakers", policy=_policy(damping=True)
        )
        # damp_family: 1 + (-0.07) + (-0.07*0.5) = 0.895 (above the 0.8696 floor)
        assert adj.modifiers["home_score_rate_scalar"] == pytest.approx(0.895)
        rec = adj.aggregation_records[0]
        assert rec["damped"] is True
        assert rec["raw_value"] == pytest.approx(0.895)

    def test_damping_preserves_sign(self):
        signals = [_sig("def_matchup_weak"), _sig("def_matchup_weak")]  # 1.05 each
        adj = compute_transition_modifier_adjustment(
            signals, "Lakers", policy=_policy(damping=True)
        )
        assert adj.modifiers["away_score_rate_scalar"] > 1.0


# ---------------------------------------------------------------------------
# Service merge: suppression records realigned with real applications
# ---------------------------------------------------------------------------


def test_merge_markov_applications_realigns_with_suppression():
    a, b, c = _sig("rest_advantage"), _sig("pace_up"), _sig("def_matchup_weak")
    evidence = [a, b, c]
    suppressed = {1}
    active = [a, c]
    rich = compute_transition_modifier_adjustment(active, "Lakers")
    merged = _merge_markov_applications(evidence, suppressed, rich.applications)

    assert len(merged) == 3
    assert merged[0]["signal_type"] == "rest_advantage"
    assert merged[0]["factor"] is not None
    # index 1 was suppressed -> suppression record, not pulled from active apps
    expected_suppressed = _suppression_record(b, "markov_transition").as_application()
    assert merged[1]["reason"] == expected_suppressed["reason"]
    assert merged[1]["applied"] is False
    assert merged[2]["signal_type"] == "def_matchup_weak"
