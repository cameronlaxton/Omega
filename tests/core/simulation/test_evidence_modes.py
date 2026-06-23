"""Graduated evidence application modes (Issues 1 & 2 + remediation proofs).

Proves the rollout ladder is honest: non-applying modes record but never move
the math; bounded_live moves the math but only within hard caps.
"""

from __future__ import annotations

import warnings

from omega.core.calibration.adjustment_policy import (
    BOUNDED_LIVE_SINGLE_CAP,
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.contracts.evidence import EvidenceSignal
from omega.core.simulation.evidence_handlers import compute_player_adjustment

_POLICY = AdjustmentPolicyRegistry().get_production_policy()


def _player_signal(**overrides) -> EvidenceSignal:
    base = dict(
        signal_type="usage_spike",
        category="player_form",
        plane="player",
        value=0.5,
        source="agent_reasoning",
        confidence=0.9,
        window="matchup",
    )
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(**base)


def _adjust(mode: str, policy: AdjustmentPolicy | None = None):
    pol = policy or _POLICY
    if mode == "bounded_live":
        pol = pol.bounded_live_effective()
    return compute_player_adjustment(
        player_context={"pts_mean": 25.0, "pts_std": 6.0},
        evidence=[_player_signal()],
        league="NBA",
        prop_type="pts",
        policy=pol,
        evidence_mode=mode,
    )


class TestNonApplyingModesRecordButDoNotAlterMath:
    def test_score_only_does_not_alter_math(self):
        adj = _adjust("score_only")
        assert adj.mean_factor == 1.0  # math unchanged
        rec = adj.records[0]
        assert rec.applied is False
        assert rec.factor != 1.0  # counterfactual still recorded
        assert rec.raw_factor != 1.0

    def test_observe_does_not_alter_math(self):
        adj = _adjust("observe")
        assert adj.mean_factor == 1.0
        assert adj.records[0].applied is False

    def test_disabled_records_present_but_unapplied(self):
        adj = _adjust("disabled")
        assert adj.mean_factor == 1.0
        assert adj.records[0].applied is False


class TestBoundedLiveChangesMathWithinCaps:
    def test_bounded_live_moves_the_mean_factor(self):
        adj = _adjust("bounded_live")
        assert adj.mean_factor != 1.0  # the prediction math actually moved
        assert adj.records[0].applied is True

    def test_bounded_live_respects_hard_single_signal_cap(self):
        adj = _adjust("bounded_live")
        # No single signal may deviate more than the bounded_live single cap.
        assert abs(adj.mean_factor - 1.0) <= BOUNDED_LIVE_SINGLE_CAP + 1e-9
        rec = adj.records[0]
        # The raw factor exceeded the cap, and the recorded capped factor is bounded.
        assert abs(rec.per_signal_capped_factor - 1.0) <= BOUNDED_LIVE_SINGLE_CAP + 1e-9

    def test_bounded_live_records_raw_vs_capped_vs_reason(self):
        adj = _adjust("bounded_live")
        rec = adj.records[0]
        assert rec.raw_factor is not None
        assert rec.per_signal_capped_factor is not None
        assert rec.final_applied_factor is not None
        assert rec.reason  # human-readable provenance string
        # The raw factor is larger than the capped one (cap actually bit).
        assert abs(rec.raw_factor - 1.0) >= abs(rec.per_signal_capped_factor - 1.0)


class TestLegacyShadowMapping:
    def test_legacy_shadow_policy_loads_as_score_only(self):
        # A policy persisted with the binary "shadow" parses as score_only and,
        # being non-applying, never moves the math.
        pol = AdjustmentPolicy(
            policy_id="legacy",
            version=1,
            mode="shadow",
            coefficients=_POLICY.coefficients,
        )
        assert pol.mode == "score_only"
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0, "pts_std": 6.0},
            evidence=[_player_signal()],
            league="NBA",
            prop_type="pts",
            policy=pol,
            evidence_mode=pol.mode,
        )
        assert adj.mean_factor == 1.0
        assert adj.records[0].applied is False
