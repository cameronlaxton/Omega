"""Exact-evaluation parity tests.

Pin the exact analytic market probabilities to the Monte-Carlo path they replace:
at high ``n`` the MC estimate must converge to the exact value (within a few
standard errors), and the exact result must satisfy the same backend contract.
This is the go/no-go gate for trusting exact eval as the backtest/calibration
substrate — see the exact-eval plan.
"""

from __future__ import annotations

import math

import pytest

from omega.core.simulation import engine as engine_module  # noqa: F401  (registers backends)
from omega.core.simulation.backends import (
    GameSimulationInput,
    PropSimulationInput,
    enforce_game_backend_contract,
)
from omega.core.simulation.engine import FastScoreSimulationBackend
from omega.core.simulation.prop_neg_binom import NegBinomPropBackend
from omega.core.simulation.soccer_bivariate_poisson import SoccerPoissonBackend

_N_MC = 200_000


def _se_tol_pp(p: float, n: int, k: float = 5.0, floor: float = 0.4) -> float:
    """Tolerance in percentage points: ``k`` MC standard errors, with a floor."""
    se = math.sqrt(max(p * (1.0 - p), 1e-9) / n)
    return max(floor, k * se * 100.0)


def _assert_prob_parity(result_exact: dict, result_mc: dict, field: str, n: int) -> None:
    exact_pp = result_exact[field]
    mc_pp = result_mc[field]
    tol = _se_tol_pp(exact_pp / 100.0, n)
    assert abs(exact_pp - mc_pp) <= tol, (
        f"{field}: exact={exact_pp} mc={mc_pp} diff={abs(exact_pp - mc_pp):.3f} tol={tol:.3f}"
    )


def _game_request(
    league: str, home_ctx: dict, away_ctx: dict, *, exact: bool, n: int
) -> GameSimulationInput:
    return GameSimulationInput(
        home_team="H",
        away_team="A",
        league=league,
        n_iterations=n,
        home_context=home_ctx,
        away_context=away_ctx,
        seed=12345,
        spread_home=-0.5,
        over_under=2.5,
        exact=exact,
    )


# --- Poisson game archetypes -------------------------------------------------

_SOCCER_HOME = {"off_rating": 1.6, "def_rating": 1.1, "xg_for": 1.6, "xg_against": 1.1}
_SOCCER_AWAY = {"off_rating": 1.2, "def_rating": 1.3, "xg_for": 1.2, "xg_against": 1.3}


def test_soccer_exact_matches_high_n_mc():
    backend = FastScoreSimulationBackend()
    exact = backend.run(_game_request("EPL", _SOCCER_HOME, _SOCCER_AWAY, exact=True, n=1000))
    mc = backend.run(_game_request("EPL", _SOCCER_HOME, _SOCCER_AWAY, exact=False, n=_N_MC))

    assert exact["success"] and mc["success"]
    assert exact["component_version"].endswith("_exact_v1")

    for field in (
        "home_win_prob",
        "away_win_prob",
        "draw_prob",
        "over_prob",
        "under_prob",
        "home_cover_prob",
        "away_cover_prob",
        "btts_yes_prob",
        "btts_no_prob",
        "dnb_home_prob",
        "dnb_away_prob",
        "double_chance_home_draw_prob",
    ):
        _assert_prob_parity(exact, mc, field, _N_MC)

    # Predicted means converge too.
    assert abs(exact["predicted_home_score"] - mc["predicted_home_score"]) <= 0.1
    assert abs(exact["predicted_away_score"] - mc["predicted_away_score"]) <= 0.1


def test_soccer_correct_score_parity():
    backend = FastScoreSimulationBackend()
    exact = backend.run(_game_request("EPL", _SOCCER_HOME, _SOCCER_AWAY, exact=True, n=1000))
    mc = backend.run(_game_request("EPL", _SOCCER_HOME, _SOCCER_AWAY, exact=False, n=_N_MC))
    for scoreline in ("0-0", "1-0", "1-1", "2-1"):
        ex = exact["correct_score_probs"].get(scoreline, 0.0)
        mm = mc["correct_score_probs"].get(scoreline, 0.0)
        assert abs(ex - mm) <= _se_tol_pp(ex / 100.0, _N_MC), f"{scoreline}: {ex} vs {mm}"


def test_baseball_exact_matches_high_n_mc():
    backend = FastScoreSimulationBackend()
    home = {"off_rating": 4.8, "def_rating": 4.1}
    away = {"off_rating": 4.2, "def_rating": 4.5}
    exact = backend.run(_game_request("MLB", home, away, exact=True, n=1000))
    mc = backend.run(_game_request("MLB", home, away, exact=False, n=_N_MC))
    assert exact["success"] and mc["success"]
    for field in ("home_win_prob", "away_win_prob", "over_prob", "under_prob"):
        _assert_prob_parity(exact, mc, field, _N_MC)


def _game_request_lines(
    league: str, home_ctx: dict, away_ctx: dict, *, spread: float, total: float, exact: bool, n: int
) -> GameSimulationInput:
    return GameSimulationInput(
        home_team="H",
        away_team="A",
        league=league,
        n_iterations=n,
        home_context=home_ctx,
        away_context=away_ctx,
        seed=12345,
        spread_home=spread,
        over_under=total,
        exact=exact,
    )


def test_basketball_exact_matches_high_n_mc():
    """Normal archetype: clipped-normal exact vs high-n MC (clip is negligible)."""
    backend = FastScoreSimulationBackend()
    home = {"off_rating": 118.0, "def_rating": 110.0, "pace": 100.0}
    away = {"off_rating": 113.0, "def_rating": 112.0, "pace": 99.0}
    exact = backend.run(
        _game_request_lines("NBA", home, away, spread=-3.5, total=224.5, exact=True, n=1000)
    )
    mc = backend.run(
        _game_request_lines("NBA", home, away, spread=-3.5, total=224.5, exact=False, n=_N_MC)
    )
    assert exact["success"] and mc["success"]
    assert exact["component_version"].endswith("_exact_v1")
    for field in (
        "home_win_prob",
        "away_win_prob",
        "over_prob",
        "under_prob",
        "home_cover_prob",
        "away_cover_prob",
    ):
        _assert_prob_parity(exact, mc, field, _N_MC)
    assert abs(exact["predicted_home_score"] - mc["predicted_home_score"]) <= 0.2
    assert abs(exact["predicted_total"] - mc["predicted_total"]) <= 0.2


def test_golf_exact_matches_high_n_mc():
    """Golf is an *uncensored* normal with inverted (lower-strokes-win) semantics;
    the exact path evaluates the summed-round normal in the same negated frame."""
    backend = FastScoreSimulationBackend()
    home = {"strokes_gained_total": 1.5}
    away = {"strokes_gained_total": 0.3}
    exact = backend.run(GameSimulationInput("A", "B", "PGA", 1000, home, away, seed=3, exact=True))
    mc = backend.run(GameSimulationInput("A", "B", "PGA", _N_MC, home, away, seed=3, exact=False))
    assert exact["success"] and mc["success"]
    # The stronger golfer (higher SG) should be favored.
    assert exact["home_win_prob"] > 50.0
    for field in ("home_win_prob", "away_win_prob"):
        _assert_prob_parity(exact, mc, field, _N_MC)


def test_football_exact_matches_high_n_mc():
    """Football has a non-trivial score=0 clip; the censored-normal exact path
    must still match MC (a pure-Gaussian approx would drift here)."""
    backend = FastScoreSimulationBackend()
    home = {"off_rating": 24.0, "def_rating": 21.0}
    away = {"off_rating": 22.0, "def_rating": 23.0}
    exact = backend.run(
        _game_request_lines("NFL", home, away, spread=-2.5, total=45.5, exact=True, n=1000)
    )
    mc = backend.run(
        _game_request_lines("NFL", home, away, spread=-2.5, total=45.5, exact=False, n=_N_MC)
    )
    assert exact["success"] and mc["success"]
    for field in (
        "home_win_prob",
        "away_win_prob",
        "over_prob",
        "under_prob",
        "home_cover_prob",
        "away_cover_prob",
    ):
        _assert_prob_parity(exact, mc, field, _N_MC)
    assert abs(exact["predicted_home_score"] - mc["predicted_home_score"]) <= 0.2


def test_hockey_exact_matches_high_n_mc():
    backend = FastScoreSimulationBackend()
    home = {"off_rating": 3.2, "def_rating": 2.8}
    away = {"off_rating": 2.9, "def_rating": 3.1}
    exact = backend.run(_game_request("NHL", home, away, exact=True, n=1000))
    mc = backend.run(_game_request("NHL", home, away, exact=False, n=_N_MC))
    assert exact["success"] and mc["success"]
    for field in ("home_win_prob", "away_win_prob", "over_prob", "under_prob"):
        _assert_prob_parity(exact, mc, field, _N_MC)


def test_exact_result_satisfies_backend_contract():
    backend = FastScoreSimulationBackend()
    exact = backend.run(_game_request("EPL", _SOCCER_HOME, _SOCCER_AWAY, exact=True, n=1000))
    # Must not raise.
    enforce_game_backend_contract(exact)
    targets = {row["target"] for row in exact["simulation_distributions"]}
    assert {"home_score", "away_score", "total", "spread"} <= targets
    for row in exact["simulation_distributions"]:
        assert row["distribution_type"] == "analytic_pmf"


def test_exact_flag_is_noop_for_path_dependent_archetypes():
    """Tennis is a path-dependent set-by-set chain (no closed form on this path),
    so the exact flag is ignored — same seed → identical MC result either way."""
    backend = FastScoreSimulationBackend()
    home = {"serve_win_pct": 0.66, "return_win_pct": 0.38}
    away = {"serve_win_pct": 0.62, "return_win_pct": 0.34}
    mc = backend.run(GameSimulationInput("A", "B", "ATP", 2000, home, away, seed=5, exact=False))
    flagged = backend.run(
        GameSimulationInput("A", "B", "ATP", 2000, home, away, seed=5, exact=True)
    )
    assert mc["home_win_prob"] == flagged["home_win_prob"]
    assert mc["component_version"] == flagged["component_version"]
    assert not mc["component_version"].endswith("_exact_v1")


def test_exact_is_seed_independent():
    """Exact eval has no sampling, so market probs do not depend on the seed."""
    backend = FastScoreSimulationBackend()
    a = backend.run(
        GameSimulationInput("H", "A", "EPL", 1000, _SOCCER_HOME, _SOCCER_AWAY, seed=1, exact=True)
    )
    b = backend.run(
        GameSimulationInput("H", "A", "EPL", 1000, _SOCCER_HOME, _SOCCER_AWAY, seed=999, exact=True)
    )
    assert a["home_win_prob"] == b["home_win_prob"]
    assert a["correct_score_probs"] == b["correct_score_probs"]


# --- Production Dixon-Coles soccer backend -----------------------------------


def _soccer_dc_request(*, exact: bool, n: int) -> GameSimulationInput:
    return GameSimulationInput(
        home_team="Arsenal",
        away_team="Chelsea",
        league="EPL",
        n_iterations=n,
        home_context={"xg_for": 1.6, "xg_against": 1.1},
        away_context={"xg_for": 1.2, "xg_against": 1.3},
        seed=42,
        spread_home=-0.5,
        over_under=2.5,
        prior_payload={"rho": -0.13, "rho_profile_id": "epl_v1"},
        exact=exact,
    )


def test_soccer_dc_backend_exact_matches_high_n_mc():
    backend = SoccerPoissonBackend()
    exact = backend.run(_soccer_dc_request(exact=True, n=1000))
    mc = backend.run(_soccer_dc_request(exact=False, n=_N_MC))

    assert exact["success"] and mc["success"]
    assert exact["component_version"] == "soccer_bvp_dc_exact_v1"
    assert exact["dc_rho"] == mc["dc_rho"]
    assert exact["rho_profile_id"] == "epl_v1"
    enforce_game_backend_contract(exact)

    for field in (
        "home_win_prob",
        "away_win_prob",
        "draw_prob",
        "over_prob",
        "under_prob",
        "btts_yes_prob",
        "dnb_home_prob",
        "double_chance_home_draw_prob",
    ):
        _assert_prob_parity(exact, mc, field, _N_MC)

    # Exact derivative rows present and contract-valid.
    targets = {row["target"] for row in exact["simulation_distributions"]}
    assert {
        "total_goals",
        "home_clean_sheet",
        "away_clean_sheet",
        "both_teams_to_score",
        "first_half_total",
    } <= targets

    # Soccer-derivatives pmfs normalize and match MC in expectation.
    assert abs(sum(exact["margin_counts"].values()) - 1.0) < 1e-6
    assert abs(sum(exact["total_counts"].values()) - 1.0) < 1e-6
    assert abs(sum(exact["fh_total_counts"].values()) - 1.0) < 1e-6

    def _expectation(counts: dict) -> float:
        total = sum(counts.values())
        return sum(float(k) * v for k, v in counts.items()) / total

    assert abs(_expectation(exact["total_counts"]) - _expectation(mc["total_counts"])) < 0.05
    assert abs(_expectation(exact["margin_counts"]) - _expectation(mc["margin_counts"])) < 0.05


def test_soccer_dc_exact_asian_handicap_matches_mc():
    """The soccer-derivatives edge path consumes margin_counts; exact and MC must
    yield the same Asian-handicap evaluation."""
    from omega.core.edge.soccer_derivatives import evaluate_asian_handicap

    backend = SoccerPoissonBackend()
    exact = backend.run(_soccer_dc_request(exact=True, n=1000))
    mc = backend.run(_soccer_dc_request(exact=False, n=_N_MC))

    ex = evaluate_asian_handicap(exact["margin_counts"], -0.5, "home")
    mm = evaluate_asian_handicap(mc["margin_counts"], -0.5, "home")
    assert abs(ex.ev_per_unit(-110) - mm.ev_per_unit(-110)) < 0.01


# --- Negative-binomial props -------------------------------------------------


@pytest.mark.parametrize("line", [60.5, 75.0, 90.5])
def test_neg_binom_prop_exact_matches_high_n_mc(line: float):
    backend = NegBinomPropBackend()
    payload = {"nb_dispersion_k": 8.0}
    common = dict(
        player_name="P",
        league="NFL",
        stat_type="rush_yds",
        line=line,
        projection_mean=75.0,
        prior_payload=payload,
        seed=7,
    )
    exact = backend.run(PropSimulationInput(n_iter=1000, exact=True, **common))
    mc = backend.run(PropSimulationInput(n_iter=_N_MC, exact=False, **common))

    for field in ("over_prob", "under_prob", "push_prob"):
        tol = _se_tol_pp(exact[field], _N_MC, floor=0.01) / 100.0
        assert abs(exact[field] - mc[field]) <= tol, f"{field}: {exact[field]} vs {mc[field]}"

    assert exact["distribution_type"] == "negative_binomial_exact"
    # Mean of the exact distribution equals the requested projection mean.
    assert exact["mean"] == pytest.approx(75.0, abs=0.5)
