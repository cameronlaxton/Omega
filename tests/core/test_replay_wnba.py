"""Milestone 1 — WNBA backend replay determinism + selection.

No production WNBA traces exist yet, so this uses synthetic-but-realistic WNBA
inputs: re-running analyze_game() with the same seed must be bit-identical, and
the WNBA league must dispatch to the markov_state_wnba backend via its league
default.

References:
  omega/core/simulation/markov_wnba.py
  omega/core/sport_baselines.py
  omega/core/config/leagues.py (WNBA default_game_backend)
  docs/phase7/MULTI_SPORT_EXPANSION.md (Milestone 1 acceptance)
"""

from __future__ import annotations

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze_game
from omega.core.simulation.backends import (
    GameSimulationInput,
    resolve_game_backend,
)

# Five WNBA matchups with realistic off/def ratings + pace context.
_WNBA_GAMES = [
    ("Las Vegas Aces", "New York Liberty", 108.0, 102.0, 99.0, 101.0, 82.0),
    ("Connecticut Sun", "Minnesota Lynx", 101.0, 98.0, 97.0, 100.0, 78.0),
    ("Seattle Storm", "Phoenix Mercury", 103.0, 104.0, 100.0, 99.0, 80.0),
    ("Indiana Fever", "Chicago Sky", 99.0, 96.0, 101.0, 103.0, 81.0),
    ("Atlanta Dream", "Washington Mystics", 100.0, 100.0, 98.0, 99.0, 79.0),
]


def _request(home, away, h_off, h_def, a_off, a_def, pace, seed=2026):
    return GameAnalysisRequest(
        home_team=home,
        away_team=away,
        league="WNBA",
        n_iterations=4000,
        seed=seed,
        home_context={"off_rating": h_off, "def_rating": h_def, "pace": pace},
        away_context={"off_rating": a_off, "def_rating": a_def, "pace": pace},
        game_context={"is_playoff": False, "rest_days": 2},
    )


def test_wnba_dispatches_to_markov_wnba_backend():
    req = _request(*_WNBA_GAMES[0])
    resp = analyze_game(req)
    assert resp.status == "success", resp.skip_reason
    assert resp.simulation is not None
    assert resp.simulation.simulation_backend == "markov_state_wnba"
    assert resp.simulation.component_version == "markov_wnba_v1"


@pytest.mark.parametrize("game", _WNBA_GAMES)
def test_wnba_replay_is_deterministic(game):
    first = analyze_game(_request(*game))
    second = analyze_game(_request(*game))
    assert first.status == "success" == second.status
    s1, s2 = first.simulation, second.simulation
    assert s1 is not None
    assert s2 is not None
    assert s1.home_win_prob == s2.home_win_prob
    assert s1.away_win_prob == s2.away_win_prob
    assert s1.predicted_total == s2.predicted_total
    assert s1.predicted_spread == s2.predicted_spread
    # WNBA is basketball: draws are not supported and must resolve to 0.
    assert s1.draw_prob == 0.0
    assert s1.home_win_prob + s1.away_win_prob == pytest.approx(100.0, abs=0.2)


def test_provided_context_reports_provided_not_baseline():
    """Full context must not be flagged as a league-default baseline run."""
    backend = resolve_game_backend("markov_state_wnba")
    assert backend is not None
    req = GameSimulationInput(
        home_team="Las Vegas Aces",
        away_team="New York Liberty",
        league="WNBA",
        n_iterations=2000,
        seed=5,
        home_context={"off_rating": 105.0, "def_rating": 100.0, "pace": 82.0},
        away_context={"off_rating": 101.0, "def_rating": 99.0, "pace": 82.0},
    )
    result = backend.run(req)
    assert result["context_source"] == "provided"
    assert result["baseline_used"] is False


def test_missing_context_falls_back_to_wnba_baselines():
    """No context -> WNBA baselines fill in and the run is flagged league_default."""
    backend = resolve_game_backend("markov_state_wnba")
    assert backend is not None
    req = GameSimulationInput(
        home_team="Seattle Storm",
        away_team="Phoenix Mercury",
        league="WNBA",
        n_iterations=2000,
        seed=5,
    )
    result = backend.run(req)
    assert result["success"] is True
    assert result["context_source"] == "league_default"
    assert result["baseline_used"] is True
    # Baselines anchor total near the WNBA league average (~160).
    assert 140.0 < result["predicted_total"] < 180.0
