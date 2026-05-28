"""Milestone 0 acceptance gate — backend registry.

Verifies that the GAME_BACKENDS / PROP_BACKENDS registries are wired correctly
at engine-module import time and that duplicate registration fails loudly.

References:
  omega/core/simulation/backends.py
  omega/core/simulation/engine.py  (import-time registration)
  docs/phase7/MULTI_SPORT_EXPANSION.md  (Milestone 0 acceptance gate)
"""

from __future__ import annotations

import pytest

# Importing the engine module triggers import-time backend registration.
import omega.core.simulation.engine  # noqa: F401
from omega.core.simulation.backends import (
    DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT,
    PropSimulationInput,
    register_game_backend,
    register_prop_backend,
    resolve_default_prop_backend,
    resolve_game_backend,
    resolve_prop_backend,
)


def test_existing_game_backends_resolve():
    assert resolve_game_backend("fast_score") is not None
    assert resolve_game_backend("markov_state") is not None


def test_unknown_game_backend_resolves_to_none():
    assert resolve_game_backend("nonexistent") is None


def test_duplicate_game_registration_raises():
    backend = resolve_game_backend("fast_score")
    with pytest.raises(ValueError, match="already registered"):
        register_game_backend("fast_score", backend)


def test_prop_distribution_router_registered():
    assert resolve_prop_backend("prop_distribution_router") is not None


def test_unknown_prop_backend_resolves_to_none():
    assert resolve_prop_backend("nonexistent") is None


def test_duplicate_prop_registration_raises():
    backend = resolve_prop_backend("prop_distribution_router")
    with pytest.raises(ValueError, match="already registered"):
        register_prop_backend("prop_distribution_router", backend)


def test_default_prop_backend_routing():
    # NFL yardage -> Negative Binomial; unknown stat -> distribution router.
    assert resolve_default_prop_backend("NFL", "rushing_yards") == "prop_neg_binom"
    assert resolve_default_prop_backend("NBA", "pts") == "prop_distribution_router"
    assert ("NFL", "passing_yards") in DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT


def test_prop_distribution_router_matches_direct_call():
    """The router must be bit-identical to calling run_player_simulation directly."""
    from omega.core.simulation.engine import run_player_simulation

    router = resolve_prop_backend("prop_distribution_router")
    request = PropSimulationInput(
        player_name="Test Player",
        league="NBA",
        stat_type="pts",
        line=25.5,
        projection_mean=27.0,
        n_iter=2000,
        seed=42,
        projection_std=6.0,
    )
    via_router = router.run(request)
    direct = run_player_simulation(
        {
            "league": "NBA",
            "stat_key": "pts",
            "mean": 27.0,
            "variance": 36.0,
            "market_line": 25.5,
        },
        n_iter=2000,
        seed=42,
    )
    assert via_router == direct
