from omega.core.simulation.backends import GameSimulationInput, PropSimulationInput
from omega.core.simulation.dispersion import DispersionPolicy
from omega.core.simulation.prop_neg_binom import NegBinomPropBackend
from omega.core.simulation.soccer_bivariate_poisson import SoccerPoissonBackend
from omega.core.simulation.tennis_markov import TennisMarkovBackend


def test_soccer_replay_parity(monkeypatch):
    import omega.core.config.leagues
    monkeypatch.setattr(
        omega.core.config.leagues,
        "get_league_config",
        lambda league_code: {"avg_total": 2.5, "home_advantage": 0.2},
    )

    backend = SoccerPoissonBackend()
    input1 = GameSimulationInput(
        seed=101,
        n_iterations=5000,
        home_team="A",
        away_team="B",
        league="EPL",
        home_context={"xg_for": 1.5, "xg_against": 1.0},
        away_context={"xg_for": 1.0, "xg_against": 1.5},
        prior_payload={"rho": 0.05},
        exact=False,
        dispersion=DispersionPolicy(variance_multiplier=1.2)
    )

    res1 = backend.run(input1)
    res2 = backend.run(input1)

    assert res1["home_win_prob"] == res2["home_win_prob"]
    assert res1["away_win_prob"] == res2["away_win_prob"]


def test_tennis_replay_parity(monkeypatch):
    import omega.core.config.leagues
    monkeypatch.setattr(
        omega.core.config.leagues,
        "get_league_config",
        lambda league_code: {"match_format": "best_of_3"},
    )

    backend = TennisMarkovBackend()
    input1 = GameSimulationInput(
        seed=202,
        n_iterations=2000,
        home_team="P1",
        away_team="P2",
        league="ATP",
        home_context={"serve_win_pct": 0.65, "return_win_pct": 0.35},
        away_context={"serve_win_pct": 0.65, "return_win_pct": 0.35},
        prior_payload={
            "pressure_coefficients": {
                "home": {"break_point": 0.05},
                "away": {"break_point": 0.05},
            }
        },
        exact=False,
        dispersion=DispersionPolicy(variance_multiplier=1.5)
    )

    res1 = backend.run(input1)
    res2 = backend.run(input1)

    assert res1["home_win_prob"] == res2["home_win_prob"]


def test_prop_replay_parity():
    backend = NegBinomPropBackend()

    input1 = PropSimulationInput(
        player_name="P1",
        league="NBA",
        stat_type="pts",
        seed=303,
        n_iter=5000,
        line=15.5,
        projection_mean=16.0,
        prior_payload={"nb_dispersion_k": 3.0},
        exact=False,
        dispersion=DispersionPolicy(variance_multiplier=2.0)
    )

    res1 = backend.run(input1)
    res2 = backend.run(input1)

    assert res1["over_prob"] == res2["over_prob"]
