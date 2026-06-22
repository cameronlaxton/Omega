"""Tests for the tennis closed-form Markov backend (Phase 7 M3 PR-T3).

Covers: exact game-chain vs Newton closed form, tiebreak symmetry, pressure
deltas at the right nodes, best-of-3 vs best-of-5, the M3 acceptance gate
(closed form vs 100k game-level MC within 0.5%), pressure ablation on
derivative markets, backend contract/determinism, fail-closed missing
context (no 0.64/0.62 defaults), and league-default dispatch for ATP/WTA/
GRAND_SLAM.
"""

from __future__ import annotations

import numpy as np
import pytest

from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze_game
from omega.core.simulation import tennis_markov as tm
from omega.core.simulation.backends import (
    GameSimulationInput,
    enforce_game_backend_contract,
    resolve_game_backend,
)

_HOME_CTX = {"serve_win_pct": 0.67, "return_win_pct": 0.40}
_AWAY_CTX = {"serve_win_pct": 0.63, "return_win_pct": 0.36}


# ---------------------------------------------------------------------------
# Chain math
# ---------------------------------------------------------------------------


def test_hold_closed_form_known_value():
    assert tm.p_hold_closed_form(0.62) == pytest.approx(0.7759, abs=1e-3)


@pytest.mark.parametrize("p", [0.45, 0.55, 0.62, 0.70, 0.80])
def test_game_chain_equals_newton_formula_when_flat(p):
    assert tm.p_hold_chain(p) == pytest.approx(tm.p_hold_closed_form(p), abs=1e-9)


def test_break_point_delta_lowers_hold_and_game_point_delta_raises_it():
    base = tm.p_hold_chain(0.62)
    assert tm.p_hold_chain(0.62, bp_delta=-0.05) < base
    assert tm.p_hold_chain(0.62, gp_delta=0.03) > base


def test_tiebreak_symmetry_and_edge():
    assert tm.p_tiebreak(0.62, 0.62) == pytest.approx(0.5, abs=1e-9)
    assert tm.p_tiebreak(0.66, 0.60) > 0.5


def test_set_tiebreak_uses_current_server(monkeypatch):
    def fake_tiebreak(pa, pb, tb_delta_a=0.0, tb_delta_b=0.0, *, first_server_a=True):
        return 0.9 if first_server_a else 0.1

    monkeypatch.setattr(tm, "p_tiebreak", fake_tiebreak)
    a_serves_tiebreak = tm.p_set(0.5, 0.5, None, None, first_server_a=True)
    b_serves_tiebreak = tm.p_set(0.5, 0.5, None, None, first_server_a=False)
    assert a_serves_tiebreak > b_serves_tiebreak


def test_set_probability_monotone_in_serve_strength():
    weak = tm.p_set(0.60, 0.62, None, None)
    strong = tm.p_set(0.66, 0.62, None, None)
    assert strong > weak


def test_best_of_five_amplifies_the_favorite():
    bo3 = tm.match_set_score_distribution(0.66, 0.60, None, None, 3)
    bo5 = tm.match_set_score_distribution(0.66, 0.60, None, None, 5)
    p3 = sum(p for (a, b), p in bo3.items() if a > b)
    p5 = sum(p for (a, b), p in bo5.items() if a > b)
    assert p5 > p3 > 0.5


def test_set_score_distribution_sums_to_one():
    dist = tm.match_set_score_distribution(0.65, 0.61, None, None, 5)
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(dist) == {(3, 0), (3, 1), (3, 2), (0, 3), (1, 3), (2, 3)}


# ---------------------------------------------------------------------------
# M3 acceptance: closed form vs 100k Monte Carlo within 0.5%
# ---------------------------------------------------------------------------


def test_closed_form_matches_100k_monte_carlo_within_half_percent():
    pa, pb = 0.655, 0.615
    coeffs_a = {"break_point_against": -0.03, "tiebreak": -0.01, "serving_for_set": -0.015}
    coeffs_b = {"break_point_against": -0.02, "set_point_serving": -0.01}

    terminals = tm.match_set_score_distribution(pa, pb, coeffs_a, coeffs_b, 3)
    closed_form = sum(p for (a, b), p in terminals.items() if a > b)

    rng = np.random.default_rng(20260629)
    n = 100_000
    wins = sum(
        1 for _ in range(n) if tm._simulate_match(rng, pa, pb, coeffs_a, coeffs_b, 2)[0] == 2
    )
    assert abs(closed_form - wins / n) < 0.005


# ---------------------------------------------------------------------------
# Pressure ablation (design verification test 11)
# ---------------------------------------------------------------------------


def test_pressure_ablation_shifts_derivative_markets():
    pa, pb = 0.65, 0.62
    pressured = {"break_point_against": -0.04, "serving_for_set": -0.02}

    flat_set1 = tm._p_set_avg(pa, pb, None, None)
    pressured_set1 = tm._p_set_avg(pa, pb, pressured, None)
    # A's pressure penalties must lower A's set-winner probability by >= 10bps.
    assert flat_set1 - pressured_set1 > 0.001

    flat_match = sum(
        p for (a, b), p in tm.match_set_score_distribution(pa, pb, None, None, 3).items() if a > b
    )
    pressured_match = sum(
        p
        for (a, b), p in tm.match_set_score_distribution(pa, pb, pressured, None, 3).items()
        if a > b
    )
    assert flat_match - pressured_match > 0.001


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


def _request(**overrides) -> GameSimulationInput:
    base = dict(
        home_team="Jannik Sinner",
        away_team="Novak Djokovic",
        league="ATP",
        n_iterations=2000,
        home_context=_HOME_CTX,
        away_context=_AWAY_CTX,
        seed=11,
    )
    base.update(overrides)
    return GameSimulationInput(**base)


def test_backend_registered_and_contract():
    backend = resolve_game_backend("tennis_markov_iid")
    assert backend is not None
    result = backend.run(_request())
    enforce_game_backend_contract(result)
    assert result["draw_prob"] == 0.0
    assert result["match_format"] == "best_of_3"

    rows = {row["target"]: row for row in result["simulation_distributions"]}
    for target in ("match_winner", "set_winner_set_1", "player_a_wins_a_set"):
        assert rows[target]["distribution_type"] == "markov_closed_form"
    for target in ("total_games_match", "total_games_set_1"):
        assert rows[target]["distribution_type"] == "empirical"
    assert result["home_win_prob"] / 100.0 == pytest.approx(
        rows["match_winner"]["sample_mean"], abs=5e-4
    )


def test_backend_same_seed_bit_identical():
    backend = tm.TennisMarkovBackend()
    assert backend.run(_request()) == backend.run(_request())


def test_missing_serve_stats_fail_closed_without_defaults():
    backend = tm.TennisMarkovBackend()
    result = backend.run(_request(home_context={"serve_win_pct": 0.67}))
    assert result["success"] is False
    assert "home_context.return_win_pct" in result["missing_requirements"]

    result = backend.run(_request(home_context=None, away_context=None))
    assert result["success"] is False
    assert len(result["missing_requirements"]) == 4


def test_match_format_override_and_grand_slam_default():
    backend = tm.TennisMarkovBackend()
    bo3 = backend.run(_request())
    bo5_override = backend.run(_request(prior_payload={"match_format": "best_of_5"}))
    slam = backend.run(_request(league="GRAND_SLAM"))
    assert bo5_override["match_format"] == "best_of_5"
    assert slam["match_format"] == "best_of_5"
    # The favorite gains in best-of-5.
    assert bo5_override["home_win_prob"] > bo3["home_win_prob"]

    bad = backend.run(_request(prior_payload={"match_format": "best_of_7"}))
    assert bad["success"] is False
    assert bad["missing_requirements"] == ["match_format"]


def test_games_total_market_uses_games_not_sets():
    backend = tm.TennisMarkovBackend()
    result = backend.run(_request(over_under=22.5))
    assert result["predicted_total"] > 15  # games, not sets
    assert 0.0 < result["over_prob"] < 100.0
    assert result["over_prob"] + result["under_prob"] == pytest.approx(100.0, abs=1.0)


def test_pressure_source_echoed():
    backend = tm.TennisMarkovBackend()
    result = backend.run(
        _request(
            prior_payload={
                "pressure_coefficients": {"home": {"break_point_against": -0.02}},
                "pressure_coefficient_source": "group_fallback",
            }
        )
    )
    assert result["pressure_coefficient_source"] == "group_fallback"


# ---------------------------------------------------------------------------
# Service dispatch
# ---------------------------------------------------------------------------


def _service_request(league="ATP", home_context=_HOME_CTX, away_context=_AWAY_CTX, **kw):
    return GameAnalysisRequest(
        home_team="Jannik Sinner",
        away_team="Novak Djokovic",
        league=league,
        n_iterations=1000,
        seed=5,
        home_context=home_context,
        away_context=away_context,
        game_context={"is_playoff": False, "rest_days": 2},
        **kw,
    )


@pytest.mark.parametrize("league", ["ATP", "WTA", "GRAND_SLAM"])
def test_league_default_routes_to_tennis_backend(league):
    resp = analyze_game(_service_request(league=league))
    assert resp.status == "success"
    assert resp.simulation.simulation_backend == "tennis_markov_iid"
    assert resp.simulation.draw_prob == 0.0


def test_service_skips_without_serve_stats():
    resp = analyze_game(_service_request(home_context={"elo_rating": 2100}))
    assert resp.status == "skipped"
    assert any("serve_win_pct" in m for m in resp.missing_requirements)


def test_moneyline_edges_priced():
    resp = analyze_game(_service_request(odds={"moneyline_home": -150, "moneyline_away": +180}))
    assert resp.status == "success"
    assert {e.side for e in resp.edges} >= {"home", "away"}
