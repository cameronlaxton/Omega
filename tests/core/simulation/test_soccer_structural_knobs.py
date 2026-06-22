"""P8.2 — soccer backend reads structural knobs from prior_payload.

The pilot for backend parameter-profile governance. The soccer bivariate-Poisson
backend now reads ``home_advantage``, ``lambda_scale``, and ``first_half_share``
from ``prior_payload``, each defaulting to the historical constant — so output is
BIT-IDENTICAL when no governed profile is injected (the live World Cup state),
and a promoted profile can override them. It also echoes ``parameter_profile_ref``
for trace provenance (P8.0.3).
"""

from __future__ import annotations

from omega.core.config.leagues import get_league_config
from omega.core.simulation.backends import GameSimulationInput
from omega.core.simulation.soccer_bivariate_poisson import (
    _FIRST_HALF_GOAL_SHARE,
    SoccerPoissonBackend,
)

_BACKEND = SoccerPoissonBackend()


def _run(prior: dict, *, exact: bool = True) -> dict:
    req = GameSimulationInput(
        home_team="Arsenal",
        away_team="Chelsea",
        league="EPL",
        n_iterations=20000,
        home_context={"xg_for": 1.7, "xg_against": 1.0},
        away_context={"xg_for": 1.3, "xg_against": 1.2},
        seed=123,
        prior_payload=prior,
        exact=exact,
    )
    res = _BACKEND.run(req)
    assert res.get("success"), res.get("skip_reason")
    return res


def _fh_expected_total(res: dict) -> float:
    counts = res["fh_total_counts"]
    total = sum(counts.values())
    return sum(int(k) * v for k, v in counts.items()) / total


def test_default_knobs_are_bit_identical():
    """Supplying the knobs at their historical defaults reproduces the no-knob
    result exactly — proof the new reads don't change live pricing."""
    hca = get_league_config("EPL").get("home_advantage", 0.0)
    base = _run({"rho": -0.05})
    explicit_default = _run(
        {
            "rho": -0.05,
            "home_advantage": hca,
            "first_half_share": _FIRST_HALF_GOAL_SHARE,
            "lambda_scale": 1.0,
        }
    )
    for key in (
        "home_win_prob",
        "away_win_prob",
        "draw_prob",
        "predicted_total",
        "predicted_spread",
    ):
        assert base[key] == explicit_default[key], key


def test_lambda_scale_increases_total():
    base = _run({"rho": -0.05})
    scaled = _run({"rho": -0.05, "lambda_scale": 1.4})
    assert scaled["predicted_total"] > base["predicted_total"]


def test_home_advantage_shifts_home_win_prob():
    neutral = _run({"rho": -0.05, "home_advantage": 0.0})
    boosted = _run({"rho": -0.05, "home_advantage": 1.0})
    assert boosted["home_win_prob"] > neutral["home_win_prob"]


def test_first_half_share_shifts_first_half_total():
    low = _run({"rho": -0.05, "first_half_share": 0.30})
    high = _run({"rho": -0.05, "first_half_share": 0.60})
    assert _fh_expected_total(high) > _fh_expected_total(low)


def test_backend_echoes_parameter_profile_ref_exact_and_mc():
    ref = {
        "param_profile_id": "soccer_bivariate_poisson_dc__EPL__v2__abc123",
        "backend_name": "soccer_bivariate_poisson_dc",
    }
    for exact in (True, False):
        res = _run({"rho": -0.05, "parameter_profile_ref": ref}, exact=exact)
        assert res["parameter_profile_ref"] == ref


def test_missing_knobs_leave_provenance_absent():
    """No parameter profile injected -> no parameter_profile_ref on the result
    (the legacy rho provenance still flows, unaffected)."""
    res = _run({"rho": -0.05, "rho_profile_id": "epl_v1", "rho_as_of_date": "2026-06-01"})
    assert "parameter_profile_ref" not in res
    assert res["rho_profile_id"] == "epl_v1"
