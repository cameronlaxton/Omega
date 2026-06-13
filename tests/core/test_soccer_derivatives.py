"""Tests for Asian-handicap / first-half-total derivative evaluation (PR-S7).

Hand-computable pmf fixtures verify the quarter-ball split, push and
half-stake semantics, and the EV bridge into the binary edge framework, plus
the end-to-end service path on the soccer bivariate-Poisson backend.
"""

from __future__ import annotations

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest, MarketQuote, SoccerDerivativeMarket
from omega.core.contracts.service import analyze_game
from omega.core.edge.soccer_derivatives import (
    evaluate_asian_handicap,
    evaluate_threshold_bet,
    evaluate_total,
)
from omega.core.simulation.backends import GameSimulationInput
from omega.core.simulation.soccer_bivariate_poisson import SoccerPoissonBackend

_MARGINS = {"-1": 25, "0": 25, "1": 30, "2": 20}  # n=100
_TOTALS = {"1": 30, "2": 40, "3": 30}


def test_half_line_no_push():
    ev = evaluate_asian_handicap(_MARGINS, -0.5, "home")
    assert ev.p_win == pytest.approx(0.5)
    assert ev.p_loss == pytest.approx(0.5)
    assert ev.p_push == ev.p_half_win == ev.p_half_loss == 0.0


def test_level_line_pushes_on_draw():
    ev = evaluate_asian_handicap(_MARGINS, 0.0, "home")
    assert ev.p_win == pytest.approx(0.5)
    assert ev.p_push == pytest.approx(0.25)
    assert ev.p_loss == pytest.approx(0.25)


def test_quarter_line_half_loss_bucket():
    """Home -0.25 splits into 0 and -0.5; a draw loses half the stake."""
    ev = evaluate_asian_handicap(_MARGINS, -0.25, "home")
    assert ev.p_win == pytest.approx(0.5)
    assert ev.p_half_loss == pytest.approx(0.25)
    assert ev.p_loss == pytest.approx(0.25)
    assert ev.ev_per_unit(+100) == pytest.approx(0.125)


def test_away_side_mirrors_home_line():
    ev = evaluate_asian_handicap(_MARGINS, -0.5, "away")  # away +0.5
    assert ev.p_win == pytest.approx(0.5)  # margins -1 and 0
    assert ev.p_loss == pytest.approx(0.5)


def test_quarter_total_over_under_buckets():
    over = evaluate_total(_TOTALS, 2.25, "over")
    assert over.p_win == pytest.approx(0.3)
    assert over.p_half_loss == pytest.approx(0.4)
    assert over.p_loss == pytest.approx(0.3)
    assert over.ev_per_unit(+100) == pytest.approx(-0.2)
    assert over.equivalent_win_prob(+100) == pytest.approx(0.4)

    under = evaluate_total(_TOTALS, 2.25, "under")
    assert under.p_win == pytest.approx(0.3)
    assert under.p_half_win == pytest.approx(0.4)
    assert under.ev_per_unit(+100) == pytest.approx(0.2)
    assert under.equivalent_win_prob(+100) == pytest.approx(0.6)

    # At even odds the equivalent probabilities are complementary.
    assert over.equivalent_win_prob(+100) + under.equivalent_win_prob(+100) == pytest.approx(1.0)


def test_negative_american_odds_ev():
    ev = evaluate_asian_handicap(_MARGINS, -0.5, "home")  # p=0.5 win/lose
    assert ev.ev_per_unit(-110) == pytest.approx(0.5 * (100 / 110) - 0.5)


def test_empty_counts_fail_loud():
    with pytest.raises(ValueError, match="empty value counts"):
        evaluate_threshold_bet({}, 1.5, direction="over")


def test_invalid_inputs():
    with pytest.raises(ValueError):
        evaluate_asian_handicap(_MARGINS, -0.5, "draw")
    with pytest.raises(ValueError):
        evaluate_total(_TOTALS, 2.5, "middle")


def test_derivative_market_enum_values():
    assert {m.value for m in SoccerDerivativeMarket} == {
        "asian_handicap",
        "total_goals_over_under",
        "both_teams_to_score",
        "correct_score",
        "first_half_total",
    }


# ---------------------------------------------------------------------------
# Backend pmf emission
# ---------------------------------------------------------------------------


def test_backend_emits_pmfs_and_first_half_row():
    backend = SoccerPoissonBackend()
    n = 3000
    result = backend.run(
        GameSimulationInput(
            home_team="Arsenal",
            away_team="Chelsea",
            league="EPL",
            n_iterations=n,
            home_context={"xg_for": 1.5, "xg_against": 1.1},
            away_context={"xg_for": 1.2, "xg_against": 1.3},
            seed=42,
            prior_payload={"rho": -0.13},
        )
    )
    assert result["success"]
    for key in ("margin_counts", "total_counts", "fh_total_counts"):
        assert sum(result[key].values()) == n

    targets = {row["target"] for row in result["simulation_distributions"]}
    assert "first_half_total" in targets

    # Thinning can never produce more 1H goals than match goals.
    max_total = max(int(k) for k in result["total_counts"])
    max_fh = max(int(k) for k in result["fh_total_counts"])
    assert max_fh <= max_total


# ---------------------------------------------------------------------------
# Service integration
# ---------------------------------------------------------------------------


def _request_with_derivatives() -> GameAnalysisRequest:
    return GameAnalysisRequest(
        home_team="Arsenal",
        away_team="Chelsea",
        league="EPL",
        n_iterations=4000,
        seed=7,
        simulation_backend="soccer_bivariate_poisson_dc",
        home_context={"xg_for": 1.5, "xg_against": 1.1},
        away_context={"xg_for": 1.2, "xg_against": 1.3},
        game_context={"is_playoff": False, "rest_days": 3},
        prior_payload={"rho": -0.13},
        odds={
            "moneyline_home": -115,
            "moneyline_away": +300,
            "moneyline_draw": +260,
            "asian_handicap_home": -0.25,
            "ah_home_price": -105,
            "ah_away_price": -115,
            "first_half_total": 1.25,
            "fh_over_price": +105,
            "fh_under_price": -125,
        },
    )


def test_service_builds_ah_and_first_half_edges():
    resp = analyze_game(_request_with_derivatives())
    assert resp.status == "success"
    markets = {(e.market, e.side) for e in resp.edges}
    assert ("asian_handicap", "home") in markets
    assert ("asian_handicap", "away") in markets
    assert ("first_half_total", "over") in markets
    assert ("first_half_total", "under") in markets

    ah_home = next(e for e in resp.edges if e.market == "asian_handicap" and e.side == "home")
    assert ah_home.line == pytest.approx(-0.25)
    ah_away = next(e for e in resp.edges if e.market == "asian_handicap" and e.side == "away")
    assert ah_away.line == pytest.approx(0.25)


def test_service_derivative_edges_read_normalized_markets():
    req = GameAnalysisRequest(
        home_team="Arsenal",
        away_team="Chelsea",
        league="EPL",
        n_iterations=4000,
        seed=7,
        simulation_backend="soccer_bivariate_poisson_dc",
        home_context={"xg_for": 1.5, "xg_against": 1.1},
        away_context={"xg_for": 1.2, "xg_against": 1.3},
        game_context={"is_playoff": False, "rest_days": 3},
        prior_payload={"rho": -0.13},
        odds={
            "moneyline_home": -115,
            "moneyline_away": +300,
            "moneyline_draw": +260,
            "markets": [
                MarketQuote(market_type="asian_handicap", selection="Home", price=-105, line=-0.25),
                MarketQuote(market_type="asian_handicap", selection="Away", price=-115, line=0.25),
                MarketQuote(market_type="first_half_total", selection="Over", price=+105, line=1.25),
                MarketQuote(market_type="first_half_total", selection="Under", price=-125, line=1.25),
            ],
        },
    )

    resp = analyze_game(req)
    markets = {(e.market, e.side) for e in resp.edges}
    assert ("asian_handicap", "home") in markets
    assert ("first_half_total", "over") in markets


def test_service_derivative_edges_are_deterministic():
    first = analyze_game(_request_with_derivatives())
    second = analyze_game(_request_with_derivatives())
    assert len(first.edges) == len(second.edges)
    for a, b in zip(first.edges, second.edges):
        assert (a.market, a.side, a.edge_pct, a.ev_pct) == (b.market, b.side, b.edge_pct, b.ev_pct)
