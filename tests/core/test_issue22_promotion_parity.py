"""Phase 7 (Issue #22) — promotion-readiness gate: replay parity + the lever.

The guardrails (confidence weighting, correlation damping, competition strength)
ship behind ``AdjustmentPolicy`` flags that the seed production policy leaves
OFF. This module is the durable gate that pins three promotion invariants:

1. Production default: every guardrail flag is off, so the live engine is
   unaffected by the merge.
2. Replay parity: with the flags off, the Issue-22 code paths reproduce the
   legacy output bit-identically (handler factors and Markov modifiers).
3. The lever: turning a flag on is the *only* thing that changes output, and the
   change stays bounded by the caps (no silent inflation).

Because the Markov path (confidence/damping) and the soccer path (competition
strength) are NOT evidence-mode gated, flipping a flag on is a real live change
— which is exactly why these invariants gate promotion.
"""

from __future__ import annotations

import warnings

import pytest

from omega.core.calibration.adjustment_policy import (
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.contracts.evidence import SIGNAL_REGISTRY, EvidenceSignal
from omega.core.simulation.evidence_handlers import compute_player_adjustment
from omega.core.simulation.evidence_to_modifier import (
    compute_transition_modifier_adjustment,
    signals_to_transition_modifiers,
)

_SEED = AdjustmentPolicyRegistry().get_production_policy()


def _game_sig(signal_type: str, direction=None, confidence: float = 0.5) -> EvidenceSignal:
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


def _usage_sig(confidence: float = 0.7) -> EvidenceSignal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(
            signal_type="usage_spike",
            category="player_form",
            plane="player",
            value=0.20,
            source="agent_reasoning",
            confidence=confidence,
            window="matchup",
        )


# ---------------------------------------------------------------------------
# Invariant 1 — production default leaves the engine untouched
# ---------------------------------------------------------------------------


def test_seed_production_policy_has_all_guardrail_flags_off():
    assert _SEED is not None
    assert _SEED.enable_confidence_weighting is False
    assert _SEED.enable_correlation_damping is False
    assert _SEED.enable_competition_strength_index is False


# ---------------------------------------------------------------------------
# Invariant 2 — replay parity: flags off == legacy output
# ---------------------------------------------------------------------------


class TestReplayParityFlagsOff:
    def test_markov_modifiers_match_legacy_under_seed_policy(self):
        sigs = [
            _game_sig("rest_advantage", direction="home"),
            _game_sig("pace_down"),
            _game_sig("def_matchup_weak", direction="away"),
        ]
        legacy = signals_to_transition_modifiers(sigs, home_team="Lakers")
        seed = compute_transition_modifier_adjustment(
            sigs, "Lakers", policy=_SEED
        ).modifiers
        assert seed == legacy

    def test_handler_factor_ignores_confidence_under_seed_policy(self):
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=[_usage_sig(confidence=0.7)],
            league="NBA",
            prop_type="pts",
            policy=_SEED,
            evidence_mode="live",
        )
        # raw 1.20 within cap; confidence weighting NOT applied (flag off).
        assert adj.records[0].factor == pytest.approx(1.20)
        assert adj.mean_factor == pytest.approx(1.20)


# ---------------------------------------------------------------------------
# Invariant 3 — the flag is the only lever, and it stays bounded
# ---------------------------------------------------------------------------


class TestFlagsAreTheLever:
    def test_confidence_flag_changes_markov_modifiers(self):
        sigs = [_game_sig("rest_advantage", direction="home", confidence=0.5)]
        off = compute_transition_modifier_adjustment(sigs, "Lakers", policy=_SEED).modifiers
        on_policy = AdjustmentPolicy(
            policy_id="on", version=1, enable_confidence_weighting=True
        )
        on = compute_transition_modifier_adjustment(sigs, "Lakers", policy=on_policy).modifiers
        assert off["home_score_rate_scalar"] == pytest.approx(1.04)
        assert on["home_score_rate_scalar"] == pytest.approx(1.02)  # 1 + 0.5*0.04
        assert on != off

    def test_all_flags_on_stays_within_caps(self):
        # A co-occurring family with confidence + damping + a family cap must
        # never escape the cap, even stacked. (No silent inflation.)
        policy = AdjustmentPolicy(
            policy_id="all-on",
            version=1,
            enable_confidence_weighting=True,
            enable_correlation_damping=True,
            family_cap=0.10,
            plane_cap=0.15,
            coefficients={
                "recent_form": {"weight": 0.35, "cap": 0.18},
                "series_avg": {"weight": 0.30, "cap": 0.18},
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evidence = [
                EvidenceSignal(
                    signal_type="recent_form", category="player_form", plane="player",
                    value=[40.0, 40.0, 40.0], source="t", confidence=1.0,
                    window="last_5", stat_key="pts",
                ),
                EvidenceSignal(
                    signal_type="series_avg", category="player_form", plane="player",
                    value=40.0, source="t", confidence=1.0, window="series", stat_key="pts",
                ),
            ]
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=evidence,
            league="NBA",
            prop_type="pts",
            policy=policy,
            evidence_mode="live",
        )
        # family cap binds at +10%, plane cap at +15%: the mean factor cannot
        # exceed 1.15 no matter how the family stacks.
        assert adj.mean_factor <= 1.15 + 1e-9
        primary = max(adj.records, key=lambda r: r.family_size)
        assert primary.family_capped_factor <= 1.10 + 1e-9
