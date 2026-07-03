"""Unit tests for the deterministic evidence-to-modifier mapping."""

from __future__ import annotations

import pytest

from omega.core.contracts.evidence import SIGNAL_REGISTRY, EvidenceSignal
from omega.core.simulation.evidence_to_modifier import (
    MAPPED_SIGNAL_TYPES,
    signals_to_transition_modifiers,
)


def _sig(signal_type: str, direction: str | None = None):
    """Minimal EvidenceSignal-compatible object for testing."""
    spec = SIGNAL_REGISTRY[signal_type]

    return EvidenceSignal(
        signal_type=signal_type,
        category=spec.category,
        plane="game",
        value=True,
        source="test",
        confidence=0.75,
        window=spec.default_window,
        direction=direction,
    )


def test_empty_signals_returns_empty_dict():
    result = signals_to_transition_modifiers([], home_team="Lakers")
    assert result == {}


def test_unknown_signal_type_is_silently_skipped():
    sig = _sig("season_record")  # in registry but not in SIGNAL_TO_MODIFIER
    result = signals_to_transition_modifiers([sig], home_team="Lakers")
    assert result == {}


def test_single_rest_advantage_applies_home_boost():
    sig = _sig("rest_advantage", direction="home")
    result = signals_to_transition_modifiers([sig], home_team="Lakers")
    assert "home_score_rate_scalar" in result
    assert result["home_score_rate_scalar"] == pytest.approx(1.04, abs=1e-9)


def test_b2b_fatigue_applies_home_suppression():
    sig = _sig("b2b_fatigue")
    result = signals_to_transition_modifiers([sig], home_team="Lakers")
    assert result["home_score_rate_scalar"] == pytest.approx(0.94, abs=1e-9)


def test_directional_flip_away_rest_advantage():
    """rest_advantage with direction='away' should target away_score_rate_scalar."""
    sig = _sig("rest_advantage", direction="away")
    result = signals_to_transition_modifiers([sig], home_team="Lakers")
    assert "away_score_rate_scalar" in result
    assert "home_score_rate_scalar" not in result


def test_cumulative_cap_prevents_compounding_exploits():
    """Three usage_role_change signals (0.93^3 ≈ 0.804) clamp to ~0.870 (1/1.15)."""
    sigs = [_sig("usage_role_change"), _sig("usage_role_change"), _sig("usage_role_change")]
    result = signals_to_transition_modifiers(sigs, home_team="Lakers")
    # Raw product would be 0.93^3 ≈ 0.804, below the floor of 1/1.15 ≈ 0.870
    assert result["home_score_rate_scalar"] == pytest.approx(1.0 / 1.15, abs=1e-6)


def test_cumulative_boost_cap():
    """Three def_matchup_weak signals (1.05^3 ≈ 1.158) must be clamped to 1.15."""
    sigs = [_sig("def_matchup_weak"), _sig("def_matchup_weak"), _sig("def_matchup_weak")]
    result = signals_to_transition_modifiers(sigs, home_team="Lakers")
    assert result["away_score_rate_scalar"] == pytest.approx(1.15, abs=1e-6)


def test_markov_plane_probation_withholds_pace_and_blowout():
    """pace_up/pace_down/blowout_risk map to mechanisms that were dead until
    2026-07-02; they are scored but NOT applied until an operator graduates
    them (markov-plane probation)."""
    for signal_type in ("pace_up", "pace_down", "blowout_risk"):
        result = signals_to_transition_modifiers([_sig(signal_type)], home_team="Lakers")
        assert result == {}, signal_type


def test_two_signals_same_key_multiply():
    """Two non-capping signals for the same key must compound (multiply)."""
    sigs = [_sig("rest_advantage"), _sig("rest_advantage")]
    result = signals_to_transition_modifiers(sigs, home_team="Lakers")
    # 1.04 * 1.04 = 1.0816 — within the 1.15 cap
    assert result["home_score_rate_scalar"] == pytest.approx(1.04 * 1.04, abs=1e-9)


def test_different_keys_are_independent():
    """Signals targeting different modifier keys must not interfere."""
    sigs = [_sig("rest_advantage", direction="home"), _sig("def_matchup_weak")]
    result = signals_to_transition_modifiers(sigs, home_team="Lakers")
    assert "home_score_rate_scalar" in result
    assert "away_score_rate_scalar" in result
    assert result["home_score_rate_scalar"] == pytest.approx(1.04, abs=1e-9)
    assert result["away_score_rate_scalar"] == pytest.approx(1.05, abs=1e-9)


def test_all_mapped_types_are_in_signal_registry():
    """Validate that every key in the mapping is a real registry entry."""
    from omega.core.contracts.evidence import SIGNAL_REGISTRY

    unknown = MAPPED_SIGNAL_TYPES - frozenset(SIGNAL_REGISTRY)
    assert unknown == frozenset(), f"Keys missing from SIGNAL_REGISTRY: {unknown}"
