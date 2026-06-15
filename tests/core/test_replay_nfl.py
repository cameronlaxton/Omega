"""M4 acceptance: NFL Negative-Binomial replay determinism + teaser edges.

Mirrors tests/core/test_replay_soccer_world_cup.py (gate 1): several NFL fixtures
replay bit-identically through the canonical analyze_game path on the league's
default backend (nfl_neg_binom), and the Wong-teaser edge rows are reproducible.
"""

from __future__ import annotations

import hashlib

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest, MarketQuote, OddsInput

from omega.core.contracts.service import analyze_game

# (home, away, home_off, home_def, away_off, away_def)
_FIXTURES = [
    ("Chiefs", "Bills", 27.0, 19.0, 25.0, 21.0),
    ("Eagles", "Cowboys", 26.0, 20.0, 24.0, 22.0),
    ("49ers", "Rams", 28.0, 18.0, 23.0, 23.0),
    ("Ravens", "Bengals", 25.0, 21.0, 26.0, 20.0),
    ("Dolphins", "Jets", 24.0, 22.0, 19.0, 24.0),
]


def _teaser_odds() -> OddsInput:
    return OddsInput(
        markets=[
            MarketQuote(market_type="teaser", selection="Home", price=-120, line=-2.5),
            MarketQuote(market_type="teaser", selection="Away", price=-120, line=8.5),
            MarketQuote(market_type="teaser", selection="Over", price=-120, line=39.5),
            MarketQuote(market_type="teaser", selection="Under", price=-120, line=51.5),
        ]
    )


def _request(home, away, hoff, hdef, aoff, adef, *, seed) -> GameAnalysisRequest:
    return GameAnalysisRequest(
        home_team=home,
        away_team=away,
        league="NFL",
        n_iterations=4000,
        seed=seed,
        home_context={"off_rating": hoff, "def_rating": hdef},
        away_context={"off_rating": aoff, "def_rating": adef},
        game_context={"is_playoff": False, "rest_days": 7},
        odds=_teaser_odds(),
    )


@pytest.mark.parametrize(
    "home,away,hoff,hdef,aoff,adef",
    _FIXTURES,
    ids=[f"{f[0]}_v_{f[1]}".replace(" ", "_") for f in _FIXTURES],
)
def test_replay_is_bit_identical(home, away, hoff, hdef, aoff, adef):
    seed = int.from_bytes(hashlib.sha256(f"{home}|{away}".encode()).digest()[:4], "big") % 100_000
    first = analyze_game(_request(home, away, hoff, hdef, aoff, adef, seed=seed))
    second = analyze_game(_request(home, away, hoff, hdef, aoff, adef, seed=seed))

    assert first.status == "success"
    assert first.simulation.simulation_backend == "nfl_neg_binom"
    assert first.simulation.component_version == "nfl_nb_v1"
    assert first.simulation.draw_prob in (0.0, None)

    for field in (
        "home_win_prob",
        "away_win_prob",
        "predicted_home_score",
        "predicted_away_score",
        "predicted_spread",
        "predicted_total",
    ):
        assert getattr(first.simulation, field) == getattr(second.simulation, field)

    # Teaser edges priced and reproducible.
    teaser_sides = {e.side for e in first.edges if e.market == "teaser"}
    assert {"home", "away", "over", "under"} <= teaser_sides
    assert len(first.edges) == len(second.edges)
    for a, b in zip(first.edges, second.edges):
        assert (a.market, a.side, a.edge_pct, a.ev_pct, a.line) == (
            b.market,
            b.side,
            b.edge_pct,
            b.ev_pct,
            b.line,
        )
