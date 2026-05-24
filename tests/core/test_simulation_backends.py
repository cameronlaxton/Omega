from __future__ import annotations

import pytest

from omega.core.contracts.service import analyze
from omega.core.simulation.backends import GameSimulationInput, enforce_game_backend_contract
from omega.core.simulation.engine import MarkovGameSimulationBackend, OmegaSimulationEngine
from omega.core.simulation.markov_engine import MarkovSimulator
from omega.trace.persistable import PersistableTrace
from omega.trace.store import TraceStore


class _BadBackend:
    backend_name = "bad"
    component_version = "bad_v1"

    def run(self, request):  # noqa: ANN001
        return {
            "success": True,
            "home_team": request.home_team,
            "away_team": request.away_team,
            "league": request.league,
            "iterations": request.n_iterations,
            "home_win_prob": 50.0,
            "away_win_prob": 50.0,
            "draw_prob": 0.0,
            "predicted_home_score": 100.0,
            "predicted_away_score": 100.0,
            "predicted_spread": 0.0,
            "predicted_total": 200.0,
            "context_source": "provided",
            "baseline_used": False,
            "simulation_distributions": [],
        }


def _game_request() -> dict:
    return {
        "home_team": "Boston Celtics",
        "away_team": "Indiana Pacers",
        "league": "NBA",
        "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
        "game_context": {"is_playoff": False, "rest_days": 2},
        "n_iterations": 100,
        "seed": 123,
    }


def _fast_game_kwargs() -> dict:
    request = dict(_game_request())
    request.pop("game_context")
    return request


def test_fast_backend_result_satisfies_contract():
    result = OmegaSimulationEngine().run_fast_game_simulation(**_fast_game_kwargs())

    assert result["success"] is True
    assert result["simulation_backend"] == "fast_score"
    assert result["component_version"] == "fast_score_v1"
    enforce_game_backend_contract(result)
    assert len(result["simulation_distributions"]) >= 4
    assert all(row["component_version"] == "fast_score_v1" for row in result["simulation_distributions"])


def test_successful_backend_without_distribution_rows_is_rejected():
    engine = OmegaSimulationEngine(game_backend=_BadBackend())

    with pytest.raises(ValueError, match="distribution rows"):
        engine.run_fast_game_simulation(**_fast_game_kwargs())


def test_analyze_game_distribution_rows_persist_to_v10_table(tmp_path):
    trace = analyze(_game_request(), session_id="sess-test-backend", bankroll=1000.0)
    store = TraceStore(db_path=tmp_path / "omega.db")
    store.persist(PersistableTrace.from_analyze_output(trace).to_store_record())

    rows = store.get_simulation_distributions(trace["trace_id"])

    assert len(rows) >= 4
    assert {row["target"] for row in rows} >= {"home_score", "away_score", "total", "spread"}
    assert all(row["component_version"] == "fast_score_v1" for row in rows)
    store.close()


def test_analyze_game_markov_rows_persist_to_v10_table(tmp_path):
    request = _game_request()
    request["simulation_backend"] = "markov_state"
    trace = analyze(request, session_id="sess-test-markov", bankroll=1000.0)
    store = TraceStore(db_path=tmp_path / "omega.db")
    store.persist(PersistableTrace.from_analyze_output(trace).to_store_record())

    rows = store.get_simulation_distributions(trace["trace_id"])

    assert len(rows) >= 4
    assert all(row["distribution_type"] == "empirical_markov" for row in rows)
    assert all(row["component_version"] == "markov_state_v1" for row in rows)
    assert all(row["distribution_params"]["base_possessions"] > 0 for row in rows)
    assert all("transition_matrix_ids" in row["distribution_params"] for row in rows)
    store.close()


def test_markov_backend_satisfies_contract_with_v10_empirical_rows():
    engine = OmegaSimulationEngine(game_backend=MarkovGameSimulationBackend())

    result = engine.run_fast_game_simulation(**_fast_game_kwargs())

    enforce_game_backend_contract(result)
    assert result["simulation_backend"] == "markov_state"
    assert result["component_version"] == "markov_state_v1"
    assert result["context_source"] == "provided"
    assert result["simulation_distributions"]
    row = result["simulation_distributions"][0]
    assert row["distribution_type"] == "empirical_markov"
    assert row["sample_mean"] is not None
    assert row["sample_std"] is not None
    assert row["p10"] is not None
    assert row["p50"] is not None
    assert row["p90"] is not None
    assert row["distribution_params"]["base_possessions"] > 0
    assert "transition_matrix_ids" in row["distribution_params"]


def test_markov_backend_accepts_scalar_transition_modifiers():
    request = _fast_game_kwargs()
    request["n_iterations"] = 500
    request["seed"] = 777
    base = OmegaSimulationEngine(game_backend=MarkovGameSimulationBackend()).run_fast_game_simulation(
        **request
    )
    adjusted = MarkovGameSimulationBackend().run(
        request=GameSimulationInput(
            home_team=request["home_team"],
            away_team=request["away_team"],
            league=request["league"],
            n_iterations=request["n_iterations"],
            home_context=request["home_context"],
            away_context=request["away_context"],
            seed=request["seed"],
            transition_modifiers={"home_score_rate_scalar": 0.5},
        )
    )

    enforce_game_backend_contract(adjusted)
    assert adjusted["predicted_home_score"] < base["predicted_home_score"]
    assert (
        adjusted["simulation_distributions"][0]["distribution_params"]["transition_modifiers"][
            "home_score_rate_scalar"
        ]
        == 0.5
    )


def test_markov_simulator_exposes_transition_matrix_ids_and_terminal_state():
    simulator = MarkovSimulator(
        league="NBA",
        players=[{"name": "Test Player", "team_side": "home", "pts_mean": 20.0}],
        home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        away_context={"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
        transition_modifiers={"home_score_rate_scalar": 0.9},
    )

    state = simulator.simulate_game(20)

    assert simulator._base_n_possessions > 0
    assert "home" in simulator.transition_matrix_ids
    assert state.home_score >= 0
    assert state.away_score >= 0
    assert state.get_player_stat("Test Player", "pts") >= 0
