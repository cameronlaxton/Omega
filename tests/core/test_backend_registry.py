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
    GAME_BACKENDS,
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


# ---------------------------------------------------------------------------
# Gap 4 — evidence routing reads a backend capability, not the name
# ---------------------------------------------------------------------------


def test_game_backends_expose_valid_evidence_mode():
    assert GAME_BACKENDS, "expected import-time game backend registration"
    for name, backend in GAME_BACKENDS.items():
        assert getattr(backend, "evidence_mode", None) in {
            "plane_adjustment",
            "markov_transition",
        }, f"{name} has invalid evidence_mode"
    assert resolve_game_backend("fast_score").evidence_mode == "plane_adjustment"
    assert resolve_game_backend("markov_state").evidence_mode == "markov_transition"
    assert resolve_game_backend("markov_state_wnba").evidence_mode == "markov_transition"


# ---------------------------------------------------------------------------
# Step 3 — router forwards distribution override + dud_prob (bit-identical)
# ---------------------------------------------------------------------------


def test_router_forwards_distribution_override():
    """A caller distribution override must survive routing through prior_payload."""
    from omega.core.simulation.engine import run_player_simulation

    router = resolve_prop_backend("prop_distribution_router")
    via_router = router.run(
        PropSimulationInput(
            player_name="P",
            league="NBA",
            stat_type="blk",
            line=1.5,
            projection_mean=1.2,
            n_iter=4000,
            seed=7,
            projection_std=1.1,
            prior_payload={"distribution": "poisson"},
        )
    )
    direct = run_player_simulation(
        {
            "league": "NBA",
            "stat_key": "blk",
            "mean": 1.2,
            "variance": 1.1**2,
            "market_line": 1.5,
            "distribution": "poisson",
        },
        n_iter=4000,
        seed=7,
    )
    assert via_router == direct
    assert via_router["distribution_type"] == "poisson"


def test_router_forwards_dud_prob():
    """A dud probability must survive routing through prior_payload."""
    from omega.core.simulation.engine import run_player_simulation

    router = resolve_prop_backend("prop_distribution_router")
    via_router = router.run(
        PropSimulationInput(
            player_name="P",
            league="NBA",
            stat_type="pts",
            line=20.5,
            projection_mean=24.0,
            n_iter=4000,
            seed=11,
            projection_std=6.0,
            prior_payload={"dud_prob": 0.3},
        )
    )
    direct = run_player_simulation(
        {
            "league": "NBA",
            "stat_key": "pts",
            "mean": 24.0,
            "variance": 36.0,
            "market_line": 20.5,
            "dud_prob": 0.3,
        },
        n_iter=4000,
        seed=11,
    )
    assert via_router == direct


# ---------------------------------------------------------------------------
# Step 4 — analyze_player_prop dispatches through the registry
# ---------------------------------------------------------------------------


def test_analyze_prop_routes_through_registry_bit_identical():
    """The live prop path must match a direct router call with the same inputs.

    game_context is left unset so no context adjustment shifts the mean, making
    the service path's PropSimulationInput identical to the one built here.
    """
    from omega.core.contracts.service import PlayerPropRequest, analyze_player_prop

    req = PlayerPropRequest(
        player_name="LeBron James",
        league="NBA",
        prop_type="pts",
        line=25.5,
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        game_date="2026-05-17",
        n_iterations=500,
        seed=42,
        player_context={"pts_mean": 27.0, "pts_std": 5.5},
        game_context={"is_playoff": False, "rest_days": 2},
    )
    resp = analyze_player_prop(req)
    assert resp.status == "success"

    router = resolve_prop_backend("prop_distribution_router")
    sim = router.run(
        PropSimulationInput(
            player_name="LeBron James",
            league="NBA",
            stat_type="pts",
            line=25.5,
            projection_mean=27.0,
            n_iter=500,
            seed=42,
            projection_std=5.5,
            prior_payload={"distribution": None, "dud_prob": 0.0},
        )
    )
    assert resp.over_prob == sim["over_prob"]
    assert resp.under_prob == sim["under_prob"]


def test_analyze_prop_routes_to_neg_binom():
    """NFL yardage routes to prop_neg_binom (now registered)."""
    from omega.core.contracts.service import PlayerPropRequest, analyze_player_prop

    assert resolve_default_prop_backend("NFL", "rushing_yards") == "prop_neg_binom"
    assert resolve_prop_backend("prop_neg_binom") is not None  # now registered

    req = PlayerPropRequest(
        player_name="Saquon Barkley",
        league="NFL",
        prop_type="rushing_yards",
        line=82.5,
        home_team="Philadelphia Eagles",
        away_team="Dallas Cowboys",
        game_date="2026-05-17",
        n_iterations=500,
        seed=1,
        player_context={"rushing_yards_mean": 90.0, "rushing_yards_std": 30.0},
        game_context={"is_playoff": False, "rest_days": 7},
    )
    resp = analyze_player_prop(req)
    assert resp.status == "success"
    # Since prop_neg_binom is registered, the distribution type should reflect it
    assert resp.distribution_type == "negative_binomial"


# ---------------------------------------------------------------------------
# Registration-time Protocol validation (fail-loud at import, not at dispatch)
# ---------------------------------------------------------------------------


def test_register_game_backend_validates_contract():
    class _NoEvidenceMode:
        backend_name = "x"
        component_version = "x_v1"

        def run(self, request):  # pragma: no cover - never reached
            return {}

    with pytest.raises(TypeError, match="missing required attributes"):
        register_game_backend("x_missing_evidence_mode", _NoEvidenceMode())

    class _NonCallableRun:
        backend_name = "y"
        component_version = "y_v1"
        evidence_mode = "plane_adjustment"
        run = 123  # not callable

    with pytest.raises(TypeError, match="callable run"):
        register_game_backend("y_bad_run", _NonCallableRun())

    # A rejected backend must not pollute the registry.
    assert resolve_game_backend("x_missing_evidence_mode") is None
    assert resolve_game_backend("y_bad_run") is None


def test_register_prop_backend_validates_contract():
    class _NoComponentVersion:
        backend_name = "p"

        def run(self, request):  # pragma: no cover - never reached
            return {}

    with pytest.raises(TypeError, match="missing required attributes"):
        register_prop_backend("p_missing_cv", _NoComponentVersion())
    assert resolve_prop_backend("p_missing_cv") is None


# ---------------------------------------------------------------------------
# §5.5 seam — game-level priors reach the backend via prior_payload
# ---------------------------------------------------------------------------


def test_game_request_carries_prior_payload():
    from omega.core.contracts.schemas import GameAnalysisRequest

    req = GameAnalysisRequest(
        home_team="A", away_team="B", league="FIFA_WORLD_CUP_2026",
        prior_payload={"rho": -0.13},
        game_context={"is_playoff": False, "rest_days": 2},
    )
    assert req.prior_payload == {"rho": -0.13}


def test_game_prior_payload_flows_to_backend():
    """GameAnalysisRequest.prior_payload must reach the backend's run() request."""
    from omega.core.simulation.backends import GameSimulationInput
    from omega.core.simulation.engine import OmegaSimulationEngine

    captured: dict = {}

    class _CaptureBackend:
        backend_name = "capture"
        component_version = "capture_v1"
        evidence_mode = "plane_adjustment"

        def run(self, request: GameSimulationInput) -> dict:
            captured["prior_payload"] = request.prior_payload
            return {"success": False}  # contract-exempt (no distribution rows required)

    OmegaSimulationEngine().run_fast_game_simulation(
        home_team="A",
        away_team="B",
        league="NBA",
        n_iterations=100,
        home_context={},
        away_context={},
        prior_payload={"rho": -0.13, "pressure_coefficients": {"break_point": 0.02}},
        backend=_CaptureBackend(),
    )
    assert captured["prior_payload"] == {
        "rho": -0.13,
        "pressure_coefficients": {"break_point": 0.02},
    }
