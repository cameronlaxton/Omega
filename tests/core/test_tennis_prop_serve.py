"""Tests for the tennis_prop_serve prop backend (Phase 7 M3 PR-T5)."""

from __future__ import annotations

import pytest

from omega.core.contracts.schemas import PlayerPropRequest
from omega.core.contracts.service import analyze_player_prop
from omega.core.simulation.backends import (
    PropSimulationInput,
    resolve_default_prop_backend,
    resolve_prop_backend,
)
from omega.core.simulation.tennis_markov import expected_game_points
from omega.core.simulation.tennis_prop_serve import TennisServePropBackend


def _request(**overrides) -> PropSimulationInput:
    base = dict(
        player_name="Jannik Sinner",
        league="ATP",
        stat_type="player_aces",
        line=8.5,
        projection_mean=9.0,
        n_iter=4000,
        seed=17,
        prior_payload={"ace_rate": 0.12, "serve_win_pct": 0.67},
    )
    base.update(overrides)
    return PropSimulationInput(**base)


def test_backend_is_registered_and_routed():
    assert resolve_default_prop_backend("ATP", "player_aces") == "tennis_prop_serve"
    assert resolve_default_prop_backend("WTA", "player_aces") == "tennis_prop_serve"
    backend = resolve_prop_backend("tennis_prop_serve")
    assert backend is not None
    assert backend.backend_name == "tennis_prop_serve"


def test_expected_game_points_sane():
    # A service game is at least 4 points; typical pro games run ~6.
    assert expected_game_points(0.62) == pytest.approx(6.1, abs=0.4)
    assert expected_game_points(0.99) == pytest.approx(4.0, abs=0.1)


def test_ace_distribution_centers_on_rate_times_volume():
    backend = TennisServePropBackend()
    result = backend.run(_request())
    params = result["distribution_params"]
    expected_mean = params["ace_rate"] * params["service_points_mean"]
    assert result["mean"] == pytest.approx(expected_mean, rel=0.05)
    assert result["distribution_type"] == "binomial_serve_points"
    assert 0.0 < result["over_prob"] < 1.0
    assert len(result["samples"]) == 100


def test_same_seed_bit_identical_and_seeds_differ():
    backend = TennisServePropBackend()
    assert backend.run(_request()) == backend.run(_request())
    assert backend.run(_request())["mean"] != backend.run(_request(seed=18))["mean"]


def test_best_of_five_raises_ace_volume():
    backend = TennisServePropBackend()
    bo3 = backend.run(_request())
    bo5 = backend.run(
        _request(prior_payload={"ace_rate": 0.12, "serve_win_pct": 0.67, "match_format": "best_of_5"})
    )
    assert bo5["mean"] > bo3["mean"]


def test_rate_derived_from_projection_when_prior_absent():
    backend = TennisServePropBackend()
    result = backend.run(_request(prior_payload=None))
    # Derived rate must reproduce the projection mean on average.
    assert result["mean"] == pytest.approx(9.0, rel=0.05)


def test_service_routes_player_aces_without_fallback_note():
    req = PlayerPropRequest(
        player_name="Jannik Sinner",
        league="ATP",
        prop_type="player_aces",
        line=8.5,
        home_team="Jannik Sinner",
        away_team="Novak Djokovic",
        game_date="2026-06-29",
        odds_over=-110,
        odds_under=-110,
        n_iterations=2000,
        seed=4,
        player_context={
            "player_aces_mean": 9.0,
            "player_aces_std": 2.5,
            "ace_rate": 0.12,
            "serve_win_pct": 0.67,
        },
        game_context={"is_playoff": False, "rest_days": 2},
    )
    resp = analyze_player_prop(req)
    assert resp.status == "success"
    assert not any("unregistered" in n for n in (resp.notes or []))
    assert resp.distribution_type == "binomial_serve_points"
    assert resp.over_prob is not None and 0.0 < resp.over_prob < 1.0
