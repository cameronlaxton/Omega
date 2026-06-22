from omega.core.simulation.backends import GameSimulationInput, PropSimulationInput
from omega.core.simulation.dispersion import DispersionPolicy
from omega.core.simulation.prop_neg_binom import NegBinomPropBackend
from omega.core.simulation.soccer_bivariate_poisson import SoccerPoissonBackend


def test_neg_binom_dispersion_parity():
    backend = NegBinomPropBackend()

    input_mc = PropSimulationInput(
        player_name="P",
        league="NBA",
        stat_type="pts",
        seed=42,
        n_iter=50000,
        line=10.5,
        projection_mean=11.0,
        prior_payload={"nb_dispersion_k": 5.0},
        exact=False,
        dispersion=DispersionPolicy(variance_multiplier=1.5),
    )
    mc_res = backend.run(input_mc)

    import dataclasses

    input_ex = dataclasses.replace(input_mc, exact=True)
    ex_res = backend.run(input_ex)

    assert abs(mc_res["over_prob"] - ex_res["over_prob"]) < 0.015
    assert abs(mc_res["under_prob"] - ex_res["under_prob"]) < 0.015
    assert abs(mc_res["push_prob"] - ex_res["push_prob"]) < 0.015
    assert abs(mc_res["mean"] - ex_res["mean"]) < 0.1

    # Assert policy applied
    assert "nb_dispersion_k" in input_mc.dispersion.applied_to


def test_soccer_bivariate_dispersion_parity(monkeypatch):
    import omega.core.config.leagues

    monkeypatch.setattr(
        omega.core.config.leagues,
        "get_league_config",
        lambda league_code: {"avg_total": 2.5, "home_advantage": 0.2},
    )

    backend = SoccerPoissonBackend()
    input_mc = GameSimulationInput(
        seed=42,
        n_iterations=50000,
        home_team="A",
        away_team="B",
        league="EPL",
        home_context={"xg_for": 1.5, "xg_against": 1.0},
        away_context={"xg_for": 1.0, "xg_against": 1.5},
        prior_payload={"rho": 0.05},
        exact=False,
        dispersion=DispersionPolicy(variance_multiplier=1.2),
    )

    mc_res = backend.run(input_mc)

    import dataclasses

    input_ex = dataclasses.replace(input_mc, exact=True)
    ex_res = backend.run(input_ex)

    assert abs(mc_res["home_win_prob"] - ex_res["home_win_prob"]) < 1.5
    assert abs(mc_res["away_win_prob"] - ex_res["away_win_prob"]) < 1.5
    assert abs(mc_res["draw_prob"] - ex_res["draw_prob"]) < 1.5

    assert "home_lambda" in input_mc.dispersion.applied_to
    assert "away_lambda" in input_mc.dispersion.applied_to
