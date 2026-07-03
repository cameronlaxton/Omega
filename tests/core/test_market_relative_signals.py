"""
Market-relative / microstructure signals (issue #28 WS2): the new probation
specs, their handlers, the stale_line toxicity gate (coarse liquidity proxy +
doubled N_min), and the Markov-path lifecycle gate.
"""

from __future__ import annotations

import warnings

import omega.ops.fit_adjustment_policy as fit_mod
from omega.core.calibration.adjustment_policy import AdjustmentPolicy, AdjustmentPolicyRegistry
from omega.core.contracts.evidence import SIGNAL_REGISTRY, EvidenceSignal, declared_lifecycle
from omega.core.simulation.evidence_handlers import (
    HANDLER_REGISTRY,
    compute_game_adjustment,
    compute_player_adjustment,
)
from omega.core.simulation.evidence_to_modifier import compute_transition_modifier_adjustment

_POLICY = AdjustmentPolicyRegistry().get_production_policy()
_NEW_SIGNALS = ("recent_form_residual", "stale_line", "sharp_line_move")


def _active(pol: AdjustmentPolicy, *types: str) -> AdjustmentPolicy:
    """A bounded_live copy of the policy with the given signals forced active."""
    return pol.model_copy(
        update={"signal_lifecycle": {t: "active" for t in types}}
    ).bounded_live_effective()


def _signal(**overrides) -> EvidenceSignal:
    base = dict(
        signal_type="stale_line",
        category="situational",
        plane="game",
        value=1.0,
        source="agent_reasoning",
        confidence=0.8,
        window="matchup",
        direction="home",
    )
    base.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(**base)


class TestNewSpecsRegistered:
    def test_all_present_as_probation_with_handlers_and_coeffs(self):
        for st in _NEW_SIGNALS:
            assert st in SIGNAL_REGISTRY, st
            assert declared_lifecycle(st) == "probation", st
            assert st in HANDLER_REGISTRY, st
            assert _POLICY.coeffs_for(st), st  # seed coefficients exist


class TestHandlersApplyWhenActive:
    def test_recent_form_residual_applies_when_active(self):
        pol = _active(_POLICY, "recent_form_residual")
        sig = _signal(
            signal_type="recent_form_residual",
            category="player_form",
            plane="player",
            value=0.20,
            stat_key="pts",
            direction="over",
        )
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0, "pts_std": 6.0},
            evidence=[sig],
            league="NBA",
            prop_type="pts",
            policy=pol,
            evidence_mode="bounded_live",
        )
        assert adj.mean_factor != 1.0
        assert adj.records[0].applied is True

    def test_recent_form_residual_probation_not_applied(self):
        pol = _POLICY.bounded_live_effective()  # no override -> stays probation
        sig = _signal(
            signal_type="recent_form_residual",
            category="player_form",
            plane="player",
            value=0.20,
            stat_key="pts",
            direction="over",
        )
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0, "pts_std": 6.0},
            evidence=[sig],
            league="NBA",
            prop_type="pts",
            policy=pol,
            evidence_mode="bounded_live",
        )
        assert adj.mean_factor == 1.0
        assert adj.records[0].applied is False

    def test_sharp_line_move_applies_when_active(self):
        pol = _active(_POLICY, "sharp_line_move")
        adj = compute_game_adjustment(
            evidence=[_signal(signal_type="sharp_line_move", value=1.0, direction="home")],
            league="NBA",
            policy=pol,
            evidence_mode="bounded_live",
        )
        assert adj.home_factor != 1.0
        assert adj.records[0].applied is True


class TestStaleLineToxicityGate:
    def test_suppressed_in_high_liquidity_when_active(self):
        pol = _active(_POLICY, "stale_line")
        adj = compute_game_adjustment(
            evidence=[_signal()],  # stale_line, home
            league="NBA",  # not low-liquidity
            policy=pol,
            evidence_mode="bounded_live",
        )
        assert adj.home_factor == 1.0  # suppressed
        rec = adj.records[0]
        assert rec.applied is False
        assert "liquidity not low" in rec.reason

    def test_applies_in_low_liquidity_when_active(self):
        pol = _active(_POLICY, "stale_line")
        adj = compute_game_adjustment(
            evidence=[_signal()],
            league="WNBA",  # low-liquidity tier
            policy=pol,
            evidence_mode="bounded_live",
        )
        assert adj.home_factor != 1.0
        assert adj.records[0].applied is True

    def test_probation_never_applies_even_in_low_liquidity(self):
        pol = _POLICY.bounded_live_effective()  # stale_line stays probation
        adj = compute_game_adjustment(
            evidence=[_signal()],
            league="WNBA",
            policy=pol,
            evidence_mode="bounded_live",
        )
        assert adj.home_factor == 1.0  # probation gate wins
        assert adj.records[0].applied is False


class TestDoubledNMin:
    def test_signal_n_min_doubles_only_toxic(self):
        assert fit_mod._signal_n_min("stale_line", 1000) == 2000
        assert fit_mod._signal_n_min("recent_form_residual", 1000) == 1000

    def test_probation_stats_uses_doubled_floor(self):
        aggs = {
            "stale_line": fit_mod._SignalAgg(
                clv_n=100, clv_aligned_weighted=0.6 * 100, clv_rows=[(100, 2.0, 5.0)]
            )
        }
        stats = fit_mod._probation_stats(aggs, n_min=80)
        assert stats["stale_line"].n_min == 160
        assert stats["stale_line"].meets_n_min is False  # 100 < 160


class TestMarkovLifecycleGate:
    def _pace_signal(self) -> EvidenceSignal:
        return _signal(
            signal_type="pace_up", category="matchup", plane="game", value=1.05, direction=None
        )

    def test_active_signal_applies(self):
        # rest_advantage: declared active, no markov-plane probation -> applies.
        sig = _signal(
            signal_type="rest_advantage",
            category="situational",
            plane="game",
            value=1.0,
            direction="home",
        )
        out = compute_transition_modifier_adjustment([sig], home_team="A")
        assert "home_score_rate_scalar" in out.modifiers

    def test_markov_plane_probation_withholds_pace_until_graduated(self):
        # pace_up maps to a mechanism that was dead until 2026-07-02; it is
        # scored but NOT applied until an operator lifecycle override of
        # "active" graduates it (plan 5.3 gate).
        out = compute_transition_modifier_adjustment([self._pace_signal()], home_team="A")
        assert "pace_scalar" not in out.modifiers
        rec = out.applications[0]
        assert rec["applied"] is False
        assert "markov_plane_probation" in rec["reason"]

        graduated = _POLICY.model_copy(update={"signal_lifecycle": {"pace_up": "active"}})
        out = compute_transition_modifier_adjustment(
            [self._pace_signal()], home_team="A", policy=graduated
        )
        assert "pace_scalar" in out.modifiers

    def test_deprecated_signal_not_applied(self):
        pol = _POLICY.model_copy(update={"signal_lifecycle": {"pace_up": "deprecated"}})
        out = compute_transition_modifier_adjustment(
            [self._pace_signal()], home_team="A", policy=pol
        )
        assert "pace_scalar" not in out.modifiers
        rec = out.applications[0]
        assert rec["applied"] is False
        assert "lifecycle=deprecated" in rec["reason"]
