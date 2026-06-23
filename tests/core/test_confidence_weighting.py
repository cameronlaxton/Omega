"""Phase 2 (Issue #22) — confidence weighting behind a policy flag.

When ``enable_confidence_weighting`` is off the factor is bit-identical to the
pre-phase per-signal-capped value; when on, each signal's deviation from 1.0 is
scaled by the agent's stated confidence (sequence step 6). Reliability and
confidence stay independently traceable.
"""

from __future__ import annotations

import warnings

import pytest

from omega.core.calibration.adjustment_policy import AdjustmentPolicy
from omega.core.contracts.evidence import EvidenceSignal
from omega.core.simulation.evidence_aggregation import (
    DEFAULT_CONFIDENCE,
    resolve_confidence,
)
from omega.core.simulation.evidence_handlers import (
    compute_game_adjustment,
    compute_player_adjustment,
)


def _policy(confidence_flag: bool, cap: float = 0.5) -> AdjustmentPolicy:
    return AdjustmentPolicy(
        policy_id="p_conf",
        version=1,
        enable_confidence_weighting=confidence_flag,
        coefficients={
            "usage_spike": {"scale": 1.0, "cap": cap},
            "motivation_edge": {"per_unit": 0.05, "cap": cap},
        },
    )


def _player_sig(confidence: float = 0.7) -> EvidenceSignal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(
            signal_type="usage_spike",
            category="player_form",
            plane="player",
            value=0.20,  # raw factor 1.20
            source="agent_reasoning",
            confidence=confidence,
            window="matchup",
        )


def _player_adj(policy: AdjustmentPolicy, confidence: float = 0.7):
    return compute_player_adjustment(
        player_context={"pts_mean": 25.0},
        evidence=[_player_sig(confidence)],
        league="NBA",
        prop_type="pts",
        policy=policy,
        evidence_mode="live",
    )


# ---------------------------------------------------------------------------
# Flag off — bit-identical to the legacy per-signal-capped factor
# ---------------------------------------------------------------------------


class TestConfidenceDisabledPreservesBehavior:
    def test_factor_ignores_confidence_when_flag_off(self):
        adj = _player_adj(_policy(False), confidence=0.7)
        assert adj.records[0].factor == pytest.approx(1.20)
        assert adj.mean_factor == pytest.approx(1.20)

    def test_confidence_sweep_does_not_move_factor_when_flag_off(self):
        for c in (0.1, 0.5, 0.9):
            adj = _player_adj(_policy(False), confidence=c)
            assert adj.records[0].factor == pytest.approx(1.20)

    def test_confidence_adjusted_factor_equals_family_capped_when_off(self):
        rec = _player_adj(_policy(False), confidence=0.3).records[0]
        # Step 6 is a pass-through when disabled: it equals the family-capped value.
        assert rec.confidence_adjusted_factor == pytest.approx(rec.family_capped_factor)
        assert rec.confidence_adjusted_factor == pytest.approx(1.20)


# ---------------------------------------------------------------------------
# Flag on — deviation scaled by confidence
# ---------------------------------------------------------------------------


class TestConfidenceEnabled:
    def test_factor_scaled_by_confidence(self):
        rec = _player_adj(_policy(True), confidence=0.7).records[0]
        # 1.0 + 0.7 * (1.20 - 1.0) = 1.14
        assert rec.factor == pytest.approx(1.14)
        assert rec.confidence_adjusted_factor == pytest.approx(1.14)
        assert rec.final_applied_factor == pytest.approx(1.14)

    def test_mean_factor_reflects_confidence_weighting(self):
        adj = _player_adj(_policy(True), confidence=0.7)
        assert adj.mean_factor == pytest.approx(1.14)

    def test_confidence_one_equals_flag_off(self):
        on = _player_adj(_policy(True), confidence=1.0).records[0]
        off = _player_adj(_policy(False), confidence=1.0).records[0]
        assert on.factor == pytest.approx(off.factor) == pytest.approx(1.20)

    def test_zero_confidence_neutralizes_signal(self):
        adj = _player_adj(_policy(True), confidence=0.0)
        # 1.0 + 0 * (.20) = 1.0 -> a no-op, so it is not applied.
        assert adj.records[0].factor == pytest.approx(1.0)
        assert adj.records[0].applied is False
        assert adj.mean_factor == pytest.approx(1.0)

    def test_reliability_and_confidence_are_separate_stages(self):
        # reliability 0.5 halves the raw delta (1.20 -> 1.10); confidence 0.5
        # then halves again (1.10 -> 1.05). Two independent, traceable stages.
        policy = AdjustmentPolicy(
            policy_id="p2",
            version=1,
            enable_confidence_weighting=True,
            coefficients={"usage_spike": {"scale": 1.0, "cap": 0.5, "reliability_weight": 0.5}},
        )
        rec = _player_adj(policy, confidence=0.5).records[0]
        assert rec.reliability_adjusted_factor == pytest.approx(1.10)
        assert rec.per_signal_capped_factor == pytest.approx(1.10)
        assert rec.confidence_adjusted_factor == pytest.approx(1.05)
        assert rec.factor == pytest.approx(1.05)

    def test_game_plane_confidence_weighting(self):
        sig = EvidenceSignal(
            signal_type="motivation_edge",
            category="situational",
            plane="game",
            value=1.0,  # per_unit 0.05 -> raw 1.05
            source="agent_reasoning",
            confidence=0.6,
            window="matchup",
            direction="home",
        )
        adj = compute_game_adjustment(
            evidence=[sig], league="NBA", policy=_policy(True), evidence_mode="live"
        )
        # 1.0 + 0.6 * (1.05 - 1.0) = 1.03
        assert adj.records[0].factor == pytest.approx(1.03)
        assert adj.home_factor == pytest.approx(1.03)
        assert adj.away_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Trace enrichment + confidence defaulting
# ---------------------------------------------------------------------------


class TestTraceEnrichment:
    REQUIRED = (
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
        "factor",  # legacy alias preserved for the V9 explode + consumers
    )

    def test_application_carries_all_fields(self):
        app = _player_adj(_policy(True), confidence=0.7).records[0].as_application()
        for key in self.REQUIRED:
            assert key in app, f"missing {key}"
        assert app["factor"] == pytest.approx(app["final_applied_factor"])
        assert app["family_role"] == "singleton"
        assert app["damping_family"] is None  # usage_spike is ungrouped

    def test_live_confidence_is_not_defaulted(self):
        rec = _player_adj(_policy(True), confidence=0.7).records[0]
        assert rec.confidence == pytest.approx(0.7)
        assert rec.confidence_defaulted is False

    def test_skip_record_still_records_confidence_and_family(self):
        # A game-plane signal on a prop analysis is skipped but stays attributable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            game_sig = EvidenceSignal(
                signal_type="pace_up",
                category="matchup",
                plane="game",
                value=1.05,
                source="agent_reasoning",
                confidence=0.4,
                window="matchup",
            )
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=[game_sig],
            league="NBA",
            prop_type="pts",
            policy=_policy(True),
            evidence_mode="live",
        )
        rec = adj.records[0]
        assert rec.target == "skip"
        assert rec.factor == pytest.approx(1.0)
        assert rec.confidence == pytest.approx(0.4)
        assert rec.damping_family == "pace"  # declared family preserved on skip


class TestResolveConfidence:
    def test_present_value_not_defaulted(self):
        assert resolve_confidence(0.7) == (0.7, False)

    def test_missing_legacy_confidence_is_defaulted_true(self):
        conf, defaulted = resolve_confidence(None)
        assert conf == DEFAULT_CONFIDENCE
        assert defaulted is True

    def test_default_confidence_is_a_weighting_noop(self):
        # Defaulting to 1.0 means confidence weighting does not move the factor.
        from omega.core.simulation.evidence_aggregation import confidence_adjusted_factor

        assert confidence_adjusted_factor(1.20, DEFAULT_CONFIDENCE) == pytest.approx(1.20)
