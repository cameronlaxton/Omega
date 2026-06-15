"""Tests for the NFL Negative-Binomial game backend (Phase 7 M4).

Covers the GameSimulationBackend contract, determinism, the discrete margin/total
pmf payload the teaser consumer reads, fail-closed on missing scoring inputs, and
the end-to-end analyze_game path (NFL now defaults to this backend) producing
Wong-teaser edges from normalized teaser markets.
"""

from __future__ import annotations

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest, MarketQuote, OddsInput
from omega.core.contracts.service import analyze_game
from omega.core.simulation.backends import (
    GameSimulationInput,
    enforce_game_backend_contract,
    resolve_game_backend,
)
from omega.core.simulation.nfl_neg_binom import NflSimulationBackend

_HOME = {"off_rating": 26.0, "def_rating": 20.0}
_AWAY = {"off_rating": 22.0, "def_rating": 23.0}


def _input(seed=7, n=4000, **kw):
    return GameSimulationInput(
        home_team="Chiefs",
        away_team="Bills",
        league="NFL",
        n_iterations=n,
        home_context=_HOME,
        away_context=_AWAY,
        seed=seed,
        **kw,
    )


def test_backend_registered_under_league_default():
    assert resolve_game_backend("nfl_neg_binom") is not None


def test_run_satisfies_contract_and_no_draw():
    result = NflSimulationBackend().run(_input())
    enforce_game_backend_contract(result)  # raises on contract violation
    assert result["success"] is True
    assert result["simulation_backend"] == "nfl_neg_binom"
    assert result["draw_prob"] == 0.0  # american_football has no draw
    assert result["home_win_prob"] + result["away_win_prob"] == pytest.approx(100.0, abs=0.2)


def test_emits_margin_and_total_pmfs():
    n = 3000
    result = NflSimulationBackend().run(_input(n=n))
    assert sum(result["margin_counts"].values()) == n
    assert sum(result["total_counts"].values()) == n
    targets = {row["target"] for row in result["simulation_distributions"]}
    assert "home_margin" in targets
    # The favored home side should win more than half the simulated games.
    assert result["home_win_prob"] > 50.0


def test_deterministic_for_same_seed():
    first = NflSimulationBackend().run(_input(seed=11))
    second = NflSimulationBackend().run(_input(seed=11))
    assert first["margin_counts"] == second["margin_counts"]
    for field in ("home_win_prob", "away_win_prob", "predicted_total", "predicted_spread"):
        assert first[field] == second[field]


def test_fails_closed_on_missing_inputs():
    result = NflSimulationBackend().run(
        GameSimulationInput(
            home_team="Chiefs",
            away_team="Bills",
            league="NFL",
            n_iterations=1000,
            home_context={"off_rating": 26.0},  # missing def_rating
            away_context=None,
            seed=7,
        )
    )
    assert result["success"] is False
    assert "home_context.def_rating" in result["missing_requirements"]
    assert "away_context.off_rating" in result["missing_requirements"]


def test_dispersion_override_changes_tail():
    """A smaller dispersion k fattens the upper tail (more blowout margins)."""
    tight = NflSimulationBackend().run(_input(prior_payload={"team_score_nb_k": 30.0}))
    fat = NflSimulationBackend().run(_input(prior_payload={"team_score_nb_k": 2.0}))
    max_margin_tight = max(abs(int(k)) for k in tight["margin_counts"])
    max_margin_fat = max(abs(int(k)) for k in fat["margin_counts"])
    assert max_margin_fat >= max_margin_tight


def _nfl_request() -> GameAnalysisRequest:
    return GameAnalysisRequest(
        home_team="Chiefs",
        away_team="Bills",
        league="NFL",  # default simulation_backend resolves to nfl_neg_binom
        n_iterations=4000,
        seed=7,
        home_context=_HOME,
        away_context=_AWAY,
        game_context={"is_playoff": False, "rest_days": 7},
        odds=OddsInput(
            markets=[
                MarketQuote(market_type="teaser", selection="Home", price=-120, line=-2.5),
                MarketQuote(market_type="teaser", selection="Away", price=-120, line=8.5),
                MarketQuote(market_type="teaser", selection="Over", price=-120, line=39.5),
                MarketQuote(market_type="teaser", selection="Under", price=-120, line=51.5),
            ]
        ),
    )


def test_analyze_game_routes_to_nfl_backend_and_builds_teaser_edges():
    resp = analyze_game(_nfl_request())
    assert resp.status == "success"
    assert resp.simulation.simulation_backend == "nfl_neg_binom"
    markets = {(e.market, e.side) for e in resp.edges}
    assert ("teaser", "home") in markets
    assert ("teaser", "away") in markets
    assert ("teaser", "over") in markets
    assert ("teaser", "under") in markets
    home_leg = next(e for e in resp.edges if e.market == "teaser" and e.side == "home")
    assert home_leg.line == pytest.approx(-2.5)


def test_analyze_game_teaser_edges_deterministic():
    first = analyze_game(_nfl_request())
    second = analyze_game(_nfl_request())
    assert len(first.edges) == len(second.edges)
    for a, b in zip(first.edges, second.edges):
        assert (a.market, a.side, a.edge_pct, a.ev_pct) == (b.market, b.side, b.edge_pct, b.ev_pct)
