"""Regression tests locking the sim invariants restored by commit 95c8d34.

These tests defend the BUG-SIM-2 / BUG-SPREAD-1 / BUG-TOTALS-1 fixes against
silent re-regression. They are invariant-shaped (not value-equality), so they
survive normal evolution of the score model.

References:
  docs/session_bugs_20260526.md
  omega/core/simulation/engine.py::_build_team_score_result
  omega/core/contracts/service.py::analyze_game
"""

from __future__ import annotations

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest, OddsInput
from omega.core.contracts.service import analyze_game
from omega.core.simulation.backends import GameSimulationInput
from omega.core.simulation.engine import (
    FastScoreSimulationBackend,
    MarkovGameSimulationBackend,
)

_GAME_CONTEXT = {"is_playoff": False, "rest_days": 2}
_N_ITER = 2000
_SEED = 12345


# ---------------------------------------------------------------------------
# BUG-SIM-2: no-draw archetypes must not leak draw probability
# ---------------------------------------------------------------------------


_NO_DRAW_CASES = [
    (
        "MLB",
        {"off_rating": 4.5, "def_rating": 4.0, "pace": 92.0},
        {"off_rating": 4.2, "def_rating": 4.3, "pace": 92.0},
    ),
    (
        "NBA",
        {"off_rating": 118.0, "def_rating": 110.0, "pace": 100.0},
        {"off_rating": 112.0, "def_rating": 116.0, "pace": 98.0},
    ),
    (
        "NFL",
        {"off_rating": 24.0, "def_rating": 21.0},
        {"off_rating": 22.0, "def_rating": 23.0},
    ),
]


@pytest.mark.parametrize(("league", "home_ctx", "away_ctx"), _NO_DRAW_CASES)
def test_fast_score_no_draw_leak_for_no_draw_archetypes(league, home_ctx, away_ctx):
    """Archetypes with supports_draw=False (MLB, NBA, NFL) must produce
    draw_prob=0 and ML probs that sum to 100 — see BUG-SIM-2 remediation."""
    backend = FastScoreSimulationBackend()
    result = backend.run(
        GameSimulationInput(
            home_team="Home",
            away_team="Away",
            league=league,
            n_iterations=_N_ITER,
            home_context=home_ctx,
            away_context=away_ctx,
            seed=_SEED,
        )
    )
    assert result["success"] is True, f"backend skipped: {result.get('skip_reason')}"
    assert result["draw_prob"] == 0.0, f"{league} draw_prob leaked: {result['draw_prob']}"
    assert result["home_win_prob"] + result["away_win_prob"] == pytest.approx(100.0, abs=0.2)


@pytest.mark.parametrize(("league", "home_ctx", "away_ctx"), _NO_DRAW_CASES)
def test_markov_no_draw_leak_for_no_draw_archetypes(league, home_ctx, away_ctx):
    """Same invariant via the Markov backend (which also routes through
    _build_team_score_result). Markov skips leagues it doesn't model — that's
    handled below."""
    backend = MarkovGameSimulationBackend()
    result = backend.run(
        GameSimulationInput(
            home_team="Home",
            away_team="Away",
            league=league,
            n_iterations=_N_ITER,
            home_context=home_ctx,
            away_context=away_ctx,
            seed=_SEED,
        )
    )
    if not result.get("success"):
        # Markov may legitimately skip leagues it doesn't model yet — that's
        # not what this test guards. Skip in that case rather than fail.
        pytest.skip(f"Markov skipped {league}: {result.get('skip_reason')}")
    assert result["draw_prob"] == 0.0
    assert result["home_win_prob"] + result["away_win_prob"] == pytest.approx(100.0, abs=0.2)


# ---------------------------------------------------------------------------
# BUG-SPREAD-1: spread edge true_prob must come from cover_prob, not ML win prob
# ---------------------------------------------------------------------------


def test_spread_edge_uses_cover_prob_not_moneyline_prob():
    """For a wide spread, the spread edge true_prob must differ materially from
    the moneyline edge true_prob. If they collide, the engine has fallen back
    to using ML win prob as a spread proxy (BUG-SPREAD-1 regression)."""
    request = GameAnalysisRequest(
        home_team="Boston Celtics",
        away_team="Indiana Pacers",
        league="NBA",
        odds=OddsInput(
            moneyline_home=-300,
            moneyline_away=240,
            spread_home=-7.5,
            spread_home_price=-110,
            spread_away_price=-110,
            over_under=220.5,
            total_over_price=-110,
            total_under_price=-110,
        ),
        n_iterations=_N_ITER,
        seed=_SEED,
        home_context={"off_rating": 122.0, "def_rating": 110.0, "pace": 100.0},
        away_context={"off_rating": 110.0, "def_rating": 115.0, "pace": 98.0},
        game_context=_GAME_CONTEXT,
    )
    response = analyze_game(request)
    assert response.status == "success", response.skip_reason

    home_ml = next(e for e in response.edges if e.market == "moneyline" and e.side == "home")
    home_spread = next(e for e in response.edges if e.market == "spread" and e.side == "home")
    assert home_spread.true_prob is not None
    assert home_ml.true_prob is not None
    assert abs(home_spread.true_prob - home_ml.true_prob) >= 0.05, (
        f"spread true_prob ({home_spread.true_prob}) suspiciously equals "
        f"ML true_prob ({home_ml.true_prob}) — possible BUG-SPREAD-1 regression"
    )
    # spread_coverage_prob attribute should now be populated (not None)
    assert home_spread.spread_coverage_prob is not None
    assert home_spread.spread_coverage_prob == pytest.approx(home_spread.true_prob, abs=1e-6)


# ---------------------------------------------------------------------------
# BUG-TOTALS-1: total edges must be emitted when over_under is supplied
# ---------------------------------------------------------------------------


def test_total_edges_emitted_when_over_under_provided():
    """fast_score must emit market='total' rows for both Over and Under when
    a total line + both prices are supplied."""
    request = GameAnalysisRequest(
        home_team="Cleveland Guardians",
        away_team="Washington Nationals",
        league="MLB",
        odds=OddsInput(
            moneyline_home=-135,
            moneyline_away=110,
            spread_home=-1.5,
            spread_home_price=155,
            spread_away_price=-180,
            over_under=7.5,
            total_over_price=-110,
            total_under_price=-110,
        ),
        n_iterations=_N_ITER,
        seed=_SEED,
        home_context={"off_rating": 4.6, "def_rating": 4.0, "pace": 92.0},
        away_context={"off_rating": 4.1, "def_rating": 4.4, "pace": 92.0},
        game_context={"is_playoff": False, "rest_days": 1, "park_factor": 1.0},
    )
    response = analyze_game(request)
    assert response.status == "success", response.skip_reason

    total_edges = [e for e in response.edges if e.market == "total"]
    sides = {e.side for e in total_edges}
    assert sides == {"over", "under"}, (
        f"expected both Over and Under total edges, got sides={sides}"
    )
    for e in total_edges:
        assert e.true_prob is not None
        assert 0.0 < e.true_prob < 1.0
        assert e.line == pytest.approx(7.5)
