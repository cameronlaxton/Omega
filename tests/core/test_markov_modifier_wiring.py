"""Markov transition-modifier mechanisms (plan 5.3).

Before 2026-07-02 two of the documented Markov modifier pathways were dead
code: ``pace_scalar`` was produced by evidence_to_modifier but consumed by
nothing in MarkovSimulator, and the momentum-scalar branches were unreachable
because the strictly alternating possession loop overwrote the cross-team
momentum marker on every possession. These tests pin the wired mechanisms —
and that runs WITHOUT modifiers stay bit-identical to the unmodified engine.
"""

from __future__ import annotations

import random

import pytest

from omega.core.simulation.markov_engine import MarkovSimulator

_CTX = {"off_rating": 100.0, "def_rating": 100.0, "pace": 80.0}


def _sim(modifiers: dict[str, float] | None = None) -> MarkovSimulator:
    return MarkovSimulator(
        league="WNBA",
        players=[],
        home_context=dict(_CTX),
        away_context=dict(_CTX),
        transition_modifiers=modifiers,
    )


def _fallback_sim(modifiers: dict[str, float] | None = None) -> MarkovSimulator:
    return MarkovSimulator(
        league="TENNIS",
        players=[],
        home_context={"pace": 65.0},
        away_context={"pace": 65.0},
        transition_modifiers=modifiers,
    )


def _mean_scores(sim: MarkovSimulator, n: int, seed: int) -> tuple[float, float]:
    random.seed(seed)
    home, away = 0.0, 0.0
    for _ in range(n):
        state = sim.simulate_game()
        home += state.home_score
        away += state.away_score
    return home / n, away / n


class TestPaceScalarMechanism:
    def test_pace_scalar_changes_total_possessions(self):
        base = _sim()._base_n_possessions
        faster = _sim({"pace_scalar": 1.5})._base_n_possessions
        slower = _sim({"pace_scalar": 0.5})._base_n_possessions
        assert faster == round(base * 1.5)
        assert slower == round(base * 0.5)

    def test_pace_scalar_identity_when_absent_or_one(self):
        base = _sim()._base_n_possessions
        assert _sim({})._base_n_possessions == base
        assert _sim({"pace_scalar": 1.0})._base_n_possessions == base

    def test_pace_scalar_moves_expected_total(self):
        lo = _mean_scores(_sim({"pace_scalar": 0.8}), 200, seed=7)
        hi = _mean_scores(_sim({"pace_scalar": 1.2}), 200, seed=7)
        assert sum(hi) > sum(lo)

    def test_pace_scalar_moves_fallback_archetype_expected_total(self):
        lo = _mean_scores(_fallback_sim({"pace_scalar": 0.8}), 200, seed=17)
        hi = _mean_scores(_fallback_sim({"pace_scalar": 1.2}), 200, seed=17)
        assert sum(hi) > sum(lo)


class TestMomentumMechanism:
    def test_momentum_scalar_changes_distribution(self):
        # A large hot-hand boost must move the home mean vs the unmodified run.
        base = _mean_scores(_sim(), 300, seed=11)
        boosted = _mean_scores(_sim({"home_momentum_scalar": 2.0}), 300, seed=11)
        assert boosted[0] > base[0]

    def test_momentum_scalar_targets_only_its_side(self):
        base = _mean_scores(_sim(), 300, seed=13)
        away_boost = _mean_scores(_sim({"away_momentum_scalar": 2.0}), 300, seed=13)
        assert away_boost[1] > base[1]
        # Home side consumes the identical RNG stream until away's boost first
        # perturbs a draw, so home mean moves far less than away mean.
        assert abs(away_boost[0] - base[0]) < (away_boost[1] - base[1])

    def test_hot_state_applies_momentum_in_expected_ppp(self):
        sim = _sim({"home_momentum_scalar": 1.5, "away_momentum_scalar": 0.5})
        assert sim._expected_ppp("home", True) == pytest.approx(
            sim._expected_ppp("home", False) * 1.5
        )
        assert sim._expected_ppp("away", True) == pytest.approx(
            sim._expected_ppp("away", False) * 0.5
        )

    def test_no_modifier_runs_are_bit_identical(self):
        # Identity defaults: the wired mechanisms must not perturb the RNG
        # stream or the math when no modifiers are supplied.
        a = _mean_scores(_sim(), 100, seed=42)
        b = _mean_scores(_sim({}), 100, seed=42)
        c = _mean_scores(
            _sim({"home_momentum_scalar": 1.0, "away_momentum_scalar": 1.0, "pace_scalar": 1.0}),
            100,
            seed=42,
        )
        assert a == b == c
