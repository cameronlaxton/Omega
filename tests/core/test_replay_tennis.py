"""M3 acceptance: 20 mixed-surface tennis matches replay deterministically.

Mirrors test_replay_wnba.py / test_replay_soccer_world_cup.py: rerunning
analyze_game with identical inputs + seed must be bit-identical, with
draw_prob == 0.0 and the tennis Markov backend on every match.
"""

from __future__ import annotations

import hashlib

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze_game

# (league, player_a, player_b, surface, spw_a, rpw_a, spw_b, rpw_b)
_FIXTURES = [
    ("ATP", "Jannik Sinner", "Carlos Alcaraz", "hard", 0.674, 0.412, 0.668, 0.418),
    ("ATP", "Novak Djokovic", "Alexander Zverev", "hard", 0.660, 0.405, 0.665, 0.380),
    ("ATP", "Daniil Medvedev", "Andrey Rublev", "hard", 0.655, 0.395, 0.674, 0.366),
    ("ATP", "Taylor Fritz", "Frances Tiafoe", "hard", 0.671, 0.360, 0.640, 0.370),
    ("ATP", "Carlos Alcaraz", "Casper Ruud", "clay", 0.644, 0.452, 0.638, 0.420),
    ("ATP", "Stefanos Tsitsipas", "Holger Rune", "clay", 0.650, 0.400, 0.645, 0.405),
    ("ATP", "Jack Draper", "Tommy Paul", "clay", 0.642, 0.390, 0.635, 0.398),
    ("ATP", "Jannik Sinner", "Novak Djokovic", "grass", 0.690, 0.385, 0.672, 0.390),
    ("ATP", "Alex de Minaur", "Hubert Hurkacz", "grass", 0.645, 0.400, 0.700, 0.340),
    ("ATP", "Grigor Dimitrov", "Felix Auger-Aliassime", "grass", 0.668, 0.375, 0.662, 0.360),
    ("WTA", "Iga Swiatek", "Aryna Sabalenka", "hard", 0.620, 0.460, 0.625, 0.450),
    ("WTA", "Coco Gauff", "Jessica Pegula", "hard", 0.595, 0.455, 0.600, 0.445),
    ("WTA", "Elena Rybakina", "Qinwen Zheng", "hard", 0.640, 0.420, 0.615, 0.430),
    ("WTA", "Iga Swiatek", "Jasmine Paolini", "clay", 0.645, 0.480, 0.590, 0.440),
    ("WTA", "Madison Keys", "Mirra Andreeva", "grass", 0.635, 0.415, 0.600, 0.445),
    ("GRAND_SLAM", "Jannik Sinner", "Carlos Alcaraz", "grass", 0.690, 0.400, 0.680, 0.410),
    ("GRAND_SLAM", "Novak Djokovic", "Jack Draper", "grass", 0.672, 0.395, 0.665, 0.380),
    ("GRAND_SLAM", "Carlos Alcaraz", "Alexander Zverev", "clay", 0.655, 0.445, 0.660, 0.395),
    ("GRAND_SLAM", "Aryna Sabalenka", "Coco Gauff", "hard", 0.628, 0.448, 0.598, 0.452),
    ("GRAND_SLAM", "Daniil Medvedev", "Taylor Fritz", "hard", 0.658, 0.398, 0.672, 0.362),
]

_SIM_FIELDS = (
    "home_win_prob",
    "away_win_prob",
    "draw_prob",
    "predicted_home_score",
    "predicted_away_score",
    "predicted_spread",
    "predicted_total",
)


def _request(league, a, b, surface, spw_a, rpw_a, spw_b, rpw_b, *, seed):
    return GameAnalysisRequest(
        home_team=a,
        away_team=b,
        league=league,
        n_iterations=1500,
        seed=seed,
        home_context={"serve_win_pct": spw_a, "return_win_pct": rpw_a},
        away_context={"serve_win_pct": spw_b, "return_win_pct": rpw_b},
        game_context={"surface": surface, "is_playoff": False, "rest_days": 2},
        prior_payload={
            "pressure_coefficients": {
                "home": {"break_point_against": -0.015, "tiebreak": -0.005},
                "away": {"break_point_against": -0.012},
            },
            "surface": surface,
        },
    )


@pytest.mark.parametrize(
    "fixture", _FIXTURES,
    ids=[f"{f[0]}_{f[1]}_v_{f[2]}_{f[3]}".replace(" ", "_") for f in _FIXTURES],
)
def test_tennis_replay_is_bit_identical(fixture):
    league, a, b, surface, spw_a, rpw_a, spw_b, rpw_b = fixture
    seed = (
        int.from_bytes(hashlib.sha256(repr((a, b, surface)).encode()).digest()[:4], "big")
        % 100_000
    )
    req = _request(league, a, b, surface, spw_a, rpw_a, spw_b, rpw_b, seed=seed)

    first = analyze_game(req)
    second = analyze_game(req)

    assert first.status == "success"
    assert first.simulation is not None and second.simulation is not None
    assert first.simulation.simulation_backend == "tennis_markov_iid"
    assert first.simulation.component_version == "tennis_markov_iid_v1"
    assert first.simulation.draw_prob == 0.0

    for field in _SIM_FIELDS:
        assert getattr(first.simulation, field) == getattr(second.simulation, field), field


def test_grand_slam_fixtures_run_best_of_five():
    league, a, b, surface, spw_a, rpw_a, spw_b, rpw_b = _FIXTURES[15]
    slam = analyze_game(_request(league, a, b, surface, spw_a, rpw_a, spw_b, rpw_b, seed=77))
    tour = analyze_game(_request("ATP", a, b, surface, spw_a, rpw_a, spw_b, rpw_b, seed=77))
    assert slam.status == tour.status == "success"
    # Best-of-5 produces materially more expected games than best-of-3.
    assert slam.simulation.predicted_total > tour.simulation.predicted_total + 5
