"""PR-S8: big-five club leagues promoted to the bivariate-Poisson backend.

Promoted: EPL (+PREMIER_LEAGUE alias), LA_LIGA (+LALIGA), SERIE_A, LIGUE_1 —
each with a per-competition rho profile fit from the StatsBomb Big-5 2015/16
full-league release. Not promoted (no credible fit dataset): CHAMPIONS_LEAGUE
(finals only) and BUNDESLIGA (single-team seasons).
"""

from __future__ import annotations

import pytest

from omega.core.config.leagues import get_league_config
from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze_game

_PROMOTED = {
    "EPL": "epl_v1",
    "PREMIER_LEAGUE": "epl_v1",
    "LA_LIGA": "laliga_v1",
    "LALIGA": "laliga_v1",
    "SERIE_A": "seriea_v1",
    "LIGUE_1": "ligue1_v1",
}
_NOT_PROMOTED = ("CHAMPIONS_LEAGUE", "BUNDESLIGA")


@pytest.mark.parametrize("league,profile", sorted(_PROMOTED.items()))
def test_promoted_league_config(league, profile):
    config = get_league_config(league)
    assert config["default_game_backend"] == "soccer_bivariate_poisson_dc"
    assert config["rho_fit_profile"] == profile


@pytest.mark.parametrize("league", _NOT_PROMOTED)
def test_unpromoted_league_stays_on_fast_score(league):
    config = get_league_config(league)
    assert "default_game_backend" not in config
    assert "rho_fit_profile" not in config


def _request(league, prior=None):
    return GameAnalysisRequest(
        home_team="Home FC",
        away_team="Away FC",
        league=league,
        n_iterations=500,
        seed=3,
        home_context={"xg_for": 1.5, "xg_against": 1.1},
        away_context={"xg_for": 1.2, "xg_against": 1.3},
        game_context={"is_playoff": False, "rest_days": 4},
        prior_payload=prior,
    )


@pytest.mark.parametrize("league", sorted(set(_PROMOTED)))
def test_promoted_league_fails_closed_without_rho(league):
    resp = analyze_game(_request(league))
    assert resp.status == "skipped"
    assert resp.missing_requirements == ["rho_prior"]


@pytest.mark.parametrize("league", sorted(set(_PROMOTED)))
def test_promoted_league_runs_dc_backend_with_rho(league):
    resp = analyze_game(_request(league, prior={"rho": -0.02}))
    assert resp.status == "success"
    assert resp.simulation.simulation_backend == "soccer_bivariate_poisson_dc"
    assert resp.simulation.draw_prob > 0.0


def test_unpromoted_league_still_succeeds_on_fast_score():
    # fast_score validates the archetype's literal off/def keys (the DC
    # backend additionally accepts pure-xG contexts).
    req = GameAnalysisRequest(
        home_team="Home FC",
        away_team="Away FC",
        league="BUNDESLIGA",
        n_iterations=500,
        seed=3,
        home_context={"off_rating": 1.6, "def_rating": 1.2},
        away_context={"off_rating": 1.4, "def_rating": 1.5},
        game_context={"is_playoff": False, "rest_days": 4},
    )
    resp = analyze_game(req)
    assert resp.status == "success"
    assert resp.simulation.simulation_backend == "fast_score"
    assert resp.simulation.draw_prob > 0.0


def test_explicit_fast_score_request_on_promoted_league_uses_league_default():
    """simulation_backend='fast_score' is the schema default, so promoted
    leagues reroute it to the DC backend — the documented league-default
    semantics (an explicitly different backend is always honored)."""
    req = _request("EPL", prior={"rho": -0.02})
    assert req.simulation_backend == "fast_score"
    resp = analyze_game(req)
    assert resp.simulation.simulation_backend == "soccer_bivariate_poisson_dc"


def test_explicit_markov_request_is_honored():
    req = GameAnalysisRequest(
        home_team="Home FC",
        away_team="Away FC",
        league="EPL",
        n_iterations=500,
        seed=3,
        simulation_backend="soccer_bivariate_poisson_dc",
        home_context={"xg_for": 1.5, "xg_against": 1.1},
        away_context={"xg_for": 1.2, "xg_against": 1.3},
        game_context={"is_playoff": False, "rest_days": 4},
        prior_payload={"rho": -0.02},
    )
    resp = analyze_game(req)
    assert resp.status == "success"
    assert resp.simulation.simulation_backend == "soccer_bivariate_poisson_dc"
