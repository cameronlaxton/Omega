"""Unit tests for the soccer bivariate-Poisson + Dixon-Coles backend (M2)."""

from __future__ import annotations

import pytest

from omega.core.simulation import engine as engine_module  # noqa: F401  (registers backends)
from omega.core.simulation.backends import (
    GameSimulationInput,
    enforce_game_backend_contract,
    resolve_game_backend,
)
from omega.core.simulation.soccer_bivariate_poisson import SoccerPoissonBackend

_HOME_CTX = {"xg_for": 1.5, "xg_against": 1.1}
_AWAY_CTX = {"xg_for": 1.2, "xg_against": 1.3}


def _request(
    *,
    prior_payload: dict | None = None,
    n_iterations: int = 50_000,
    seed: int | None = 42,
    home_context: dict | None = _HOME_CTX,
    away_context: dict | None = _AWAY_CTX,
    league: str = "EPL",
) -> GameSimulationInput:
    return GameSimulationInput(
        home_team="Arsenal",
        away_team="Chelsea",
        league=league,
        n_iterations=n_iterations,
        home_context=home_context,
        away_context=away_context,
        seed=seed,
        prior_payload=prior_payload,
    )


def _score_pct(result: dict, scoreline: str) -> float:
    return float(result["correct_score_probs"].get(scoreline, 0.0))


def test_backend_is_registered():
    backend = resolve_game_backend("soccer_bivariate_poisson_dc")
    assert backend is not None
    assert backend.backend_name == "soccer_bivariate_poisson_dc"
    assert backend.evidence_mode == "plane_adjustment"


def test_missing_rho_fails_closed():
    backend = SoccerPoissonBackend()
    for payload in (None, {}, {"rho": None}, {"rho": "not-a-number"}):
        result = backend.run(_request(prior_payload=payload, n_iterations=100))
        assert result["success"] is False
        assert result.get("skipped") is True
        assert result["missing_requirements"] == ["rho_prior"]


def test_missing_context_fails_closed_before_rho():
    backend = SoccerPoissonBackend()
    result = backend.run(
        _request(prior_payload={"rho": -0.13}, home_context=None, n_iterations=100)
    )
    assert result["success"] is False
    assert any("home_context" in m for m in result["missing_requirements"])


def test_dixon_coles_shifts_low_score_cells():
    """Negative rho moves mass toward 0-0/1-1 and away from 1-0/0-1."""
    backend = SoccerPoissonBackend()
    with_dc = backend.run(_request(prior_payload={"rho": -0.13}))
    without_dc = backend.run(_request(prior_payload={"rho": 0.0}))
    assert with_dc["success"] and without_dc["success"]

    assert _score_pct(with_dc, "0-0") > _score_pct(without_dc, "0-0")
    assert _score_pct(with_dc, "1-1") > _score_pct(without_dc, "1-1")
    assert _score_pct(with_dc, "1-0") < _score_pct(without_dc, "1-0")
    assert _score_pct(with_dc, "0-1") < _score_pct(without_dc, "0-1")


def test_nonzero_draw_prob_and_three_way_fields():
    backend = SoccerPoissonBackend()
    result = backend.run(_request(prior_payload={"rho": -0.13}))
    assert result["draw_prob"] > 0.0
    for field in (
        "double_chance_home_draw_prob",
        "dnb_home_prob",
        "btts_yes_prob",
        "correct_score_probs",
    ):
        assert field in result


def test_same_seed_is_bit_identical():
    backend = SoccerPoissonBackend()
    payload = {"rho": -0.13, "rho_profile_id": "fifa_intl_v1"}
    first = backend.run(_request(prior_payload=payload))
    second = backend.run(_request(prior_payload=payload))
    assert first == second


def test_different_seeds_differ():
    backend = SoccerPoissonBackend()
    a = backend.run(_request(prior_payload={"rho": -0.13}, seed=1))
    b = backend.run(_request(prior_payload={"rho": -0.13}, seed=2))
    assert a["correct_score_probs"] != b["correct_score_probs"]


def test_contract_and_derivative_rows():
    backend = SoccerPoissonBackend()
    result = backend.run(_request(prior_payload={"rho": -0.13}, n_iterations=2_000))
    enforce_game_backend_contract(result)

    targets = {row["target"] for row in result["simulation_distributions"]}
    assert {
        "home_score",
        "away_score",
        "total",
        "spread",
        "total_goals",
        "home_clean_sheet",
        "away_clean_sheet",
        "both_teams_to_score",
    } <= targets
    for row in result["simulation_distributions"]:
        assert row["component_version"] == "soccer_bvp_dc_v1"
        assert row["seed"] == 42


def test_rho_provenance_echoed():
    backend = SoccerPoissonBackend()
    result = backend.run(
        _request(
            prior_payload={
                "rho": -0.11,
                "rho_profile_id": "fifa_intl_v1",
                "rho_as_of_date": "2026-06-10",
            },
            n_iterations=500,
        )
    )
    assert result["dc_rho"] == pytest.approx(-0.11)
    assert result["rho_profile_id"] == "fifa_intl_v1"
    assert result["rho_as_of_date"] == "2026-06-10"


def test_static_league_rho_is_never_read():
    """EPL config carries a legacy static rho; the backend must ignore it."""
    backend = SoccerPoissonBackend()
    result = backend.run(_request(prior_payload=None, n_iterations=100))
    assert result["success"] is False
    assert result["missing_requirements"] == ["rho_prior"]
