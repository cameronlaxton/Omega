"""Structural sharpness knobs — the raw-ECE levers ``omega-fit-backend-structure``
tunes at the source (Part 1 of the Structural Calibration Loop).

Each knob is **mean-preserving** and **identity-default**, so:

* a request without the knob (or with it at 1.0) is BIT-IDENTICAL to the pre-knob
  engine — zero blast radius until a profile is promoted; and
* turning it away from 1.0 softens (or sharpens) probability *extremity* WITHOUT
  moving the score means / totals, so the correction is the in-model analogue of
  shrinkage and stays coherent across every derived market.

Knobs, by archetype:
* ``margin_sd_scale`` — Normal score-SD multiplier (NBA / american-football).
* ``lambda_gap_scale`` — mean-total-preserving Poisson lambda-gap compression
  (soccer DC, MLB, NHL).
* ``nb_k_scale`` — negative-binomial dispersion-``k`` multiplier (NFL scores, props).

The knobs are applied at the shared *parameter* point (lambdas / sigmas / k), so the
Monte-Carlo sampler and the exact evaluator consume the identical adjusted params —
MC and exact cannot drift (asserted below).
"""

from __future__ import annotations

from omega.core.simulation.backends import GameSimulationInput, PropSimulationInput
from omega.core.simulation.engine import OmegaSimulationEngine
from omega.core.simulation.nfl_neg_binom import NflSimulationBackend
from omega.core.simulation.prop_neg_binom import NegBinomPropBackend
from omega.core.simulation.soccer_bivariate_poisson import SoccerPoissonBackend

_ENG = OmegaSimulationEngine()
_SOCCER = SoccerPoissonBackend()
_NFL = NflSimulationBackend()
_PROP = NegBinomPropBackend()

_GAME_KEYS = (
    "home_win_prob",
    "away_win_prob",
    "draw_prob",
    "predicted_home_score",
    "predicted_away_score",
    "predicted_total",
    "predicted_spread",
)


# ---------------------------------------------------------------------------
# soccer Dixon-Coles backend — lambda_gap_scale (mean-TOTAL-preserving)
# ---------------------------------------------------------------------------


def _soccer(prior: dict, *, exact: bool = True) -> dict:
    req = GameSimulationInput(
        home_team="H",
        away_team="A",
        league="EPL",
        n_iterations=40000,
        home_context={"xg_for": 2.0, "xg_against": 0.9},
        away_context={"xg_for": 0.9, "xg_against": 1.6},
        seed=123,
        prior_payload=prior,
        exact=exact,
    )
    res = _SOCCER.run(req)
    assert res.get("success"), res.get("skip_reason")
    return res


def test_soccer_gap_scale_default_is_bit_identical():
    base = _soccer({"rho": -0.10})
    explicit = _soccer({"rho": -0.10, "lambda_gap_scale": 1.0})
    for k in _GAME_KEYS:
        assert base[k] == explicit[k], k


def test_soccer_gap_scale_preserves_total_and_softens_moneyline():
    base = _soccer({"rho": -0.10})
    comp = _soccer({"rho": -0.10, "lambda_gap_scale": 0.6})
    # E[total] preserved at the lambda level -> predicted total unchanged (to rounding).
    assert abs(comp["predicted_total"] - base["predicted_total"]) <= 0.1
    # Compression pulls the favorite toward 50 and lifts the draw.
    assert comp["home_win_prob"] < base["home_win_prob"]
    assert comp["draw_prob"] > base["draw_prob"]
    assert comp["away_win_prob"] > base["away_win_prob"]


def test_soccer_gap_scale_honored_on_both_paths():
    """The knob rides the shared lambdas, so exact and MC agree within MC s.e."""
    ex = _soccer({"rho": -0.10, "lambda_gap_scale": 0.7}, exact=True)
    mc = _soccer({"rho": -0.10, "lambda_gap_scale": 0.7}, exact=False)
    assert abs(ex["home_win_prob"] - mc["home_win_prob"]) < 2.0
    assert abs(ex["predicted_total"] - mc["predicted_total"]) < 0.1


# ---------------------------------------------------------------------------
# Normal archetype (basketball) — margin_sd_scale (mean-INDEPENDENT)
# ---------------------------------------------------------------------------


def _nba(prior: dict | None, *, exact: bool = True) -> dict:
    res = _ENG.run_fast_game_simulation(
        "H",
        "A",
        league="NBA",
        n_iterations=200,
        home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        away_context={"off_rating": 110.0, "def_rating": 112.0, "pace": 100.0},
        seed=7,
        exact=exact,
        prior_payload=prior,
    )
    assert res.get("success"), res.get("skip_reason")
    return res


def test_margin_sd_scale_absent_equals_explicit_one():
    assert _nba(None) == _nba({"margin_sd_scale": 1.0})


def test_margin_sd_scale_preserves_means_and_softens():
    base = _nba(None)
    wide = _nba({"margin_sd_scale": 1.5})
    # Mean-independent: predicted scores unchanged (censoring negligible at NBA scale).
    assert wide["predicted_home_score"] == base["predicted_home_score"]
    assert wide["predicted_away_score"] == base["predicted_away_score"]
    # Wider score SDs pull the favorite's win probability toward 50.
    assert wide["home_win_prob"] < base["home_win_prob"]


# ---------------------------------------------------------------------------
# Poisson archetype (baseball, fast_score) — lambda_gap_scale via config bridge
# ---------------------------------------------------------------------------


def _mlb(prior: dict | None) -> dict:
    res = _ENG.run_fast_game_simulation(
        "H",
        "A",
        league="MLB",
        n_iterations=200,
        home_context={"off_rating": 5.2, "def_rating": 3.8},
        away_context={"off_rating": 3.9, "def_rating": 4.6},
        seed=7,
        exact=True,
        prior_payload=prior,
    )
    assert res.get("success"), res.get("skip_reason")
    return res


def test_baseball_gap_scale_absent_equals_explicit_one():
    assert _mlb(None) == _mlb({"lambda_gap_scale": 1.0})


def test_baseball_gap_scale_preserves_total_and_softens():
    base = _mlb(None)
    comp = _mlb({"lambda_gap_scale": 0.6})
    assert abs(comp["predicted_total"] - base["predicted_total"]) <= 0.1
    assert comp["home_win_prob"] < base["home_win_prob"]


# ---------------------------------------------------------------------------
# Negative-binomial backends — nb_k_scale
# ---------------------------------------------------------------------------


def _prop(prior: dict) -> dict:
    req = PropSimulationInput(
        player_name="P",
        league="NFL",
        stat_type="passing_yards",
        line=250.0,
        projection_mean=260.0,
        n_iter=200,
        seed=7,
        prior_payload={"nb_dispersion_k": 8.0, **prior},
        exact=True,
    )
    return _PROP.run(req)


def test_prop_nb_k_scale_absent_equals_explicit_one():
    assert _prop({}) == _prop({"nb_k_scale": 1.0})


def test_prop_nb_k_scale_sharpens_distribution():
    base = _prop({})
    sharp = _prop({"nb_k_scale": 3.0})  # larger k -> smaller variance
    assert sharp["std"] < base["std"]


def _nfl(prior: dict | None) -> dict:
    req = GameSimulationInput(
        home_team="H",
        away_team="A",
        league="NFL",
        n_iterations=200,
        home_context={"off_rating": 26.0, "def_rating": 20.0},
        away_context={"off_rating": 19.0, "def_rating": 24.0},
        seed=7,
        prior_payload=prior,
    )
    res = _NFL.run(req)
    assert res.get("success"), res.get("skip_reason")
    return res


def test_nfl_nb_k_scale_absent_equals_explicit_one():
    assert _nfl(None) == _nfl({"nb_k_scale": 1.0})


def test_nfl_nb_k_scale_scales_recorded_dispersion():
    base = _nfl(None)
    sharp = _nfl({"nb_k_scale": 2.0})
    # The recorded team-score k (provenance) is the scaled value used for sampling.
    assert sharp["team_score_nb_k"] == base["team_score_nb_k"] * 2.0
