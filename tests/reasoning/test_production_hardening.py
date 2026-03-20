"""
Phase 5 tests: ExecutionTrace, seed reproducibility, strict validation, downgrade tracking.
"""

import pytest
from unittest.mock import MagicMock, patch

from omega.core.models import (
    AnswerPlan,
    Entity,
    EntityRole,
    ExecutionMode,
    ExecutionTrace,
    GatheredFact,
    GatherSlot,
    InputImportance,
    OutputPackage,
    ProviderResult,
    QueryUnderstanding,
    Subject,
    UserGoal,
)
from omega.core.simulation.validation import validate_sim_context
from omega.reasoning.evaluator import apply_quality_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_understanding(**overrides) -> QueryUnderstanding:
    defaults = dict(
        subjects=[Subject.GAME],
        league="NBA",
        entities=[
            Entity(name="Lakers", role=EntityRole.HOME, entity_type="team"),
            Entity(name="Celtics", role=EntityRole.AWAY, entity_type="team"),
        ],
        goal=UserGoal.ANALYZE,
        wants_betting_advice=True,
        raw_prompt="Lakers vs Celtics NBA",
    )
    defaults.update(overrides)
    return QueryUnderstanding(**defaults)


def _make_slot(key="home_team.team_stat", importance=InputImportance.CRITICAL, **kw):
    defaults = dict(key=key, data_type="team_stat", entity="Lakers", league="NBA", importance=importance)
    defaults.update(kw)
    return GatherSlot(**defaults)


def _make_fact(slot=None, filled=True, quality=0.8, data=None):
    slot = slot or _make_slot()
    result = None
    if filled:
        result = ProviderResult(
            data=data or {"off_rating": 112.0, "def_rating": 108.0},
            source="espn",
            method="structured_api",
            confidence=0.9,
        )
    return GatheredFact(slot=slot, result=result, filled=filled, quality_score=quality)


def _make_plan(**overrides):
    defaults = dict(
        execution_modes=[ExecutionMode.NATIVE_SIM],
        output_packages=[OutputPackage.BET_CARD, OutputPackage.GAME_BREAKDOWN],
        simulation_required=True,
        betting_recommendations_included=True,
        quality_thresholds={"bet_card": 0.7, "game_breakdown": 0.5},
    )
    defaults.update(overrides)
    return AnswerPlan(**defaults)


# ---------------------------------------------------------------------------
# ExecutionTrace tests
# ---------------------------------------------------------------------------

class TestExecutionTrace:
    """Test that ExecutionTrace model is correctly structured."""

    def test_trace_has_identity_fields(self):
        trace = ExecutionTrace(prompt="test query")
        assert trace.trace_id is not None
        assert trace.run_id is not None
        assert len(trace.run_id) == 12
        assert trace.timestamp is not None
        assert trace.model_version == "omega-v0"

    def test_trace_serializes_cleanly(self):
        trace = ExecutionTrace(prompt="test", league="NBA", matchup="Celtics @ Lakers")
        d = trace.model_dump()
        assert d["prompt"] == "test"
        assert d["league"] == "NBA"
        assert d["matchup"] == "Celtics @ Lakers"
        assert isinstance(d["run_id"], str)

    def test_trace_backtest_fields_default_empty(self):
        trace = ExecutionTrace(prompt="test")
        assert trace.predictions is None
        assert trace.recommendations == []
        assert trace.odds_snapshot is None


# ---------------------------------------------------------------------------
# Seed reproducibility tests
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    """Test that seeded simulations produce deterministic results."""

    def test_seed_produces_identical_results(self):
        from omega.core.simulation.engine import OmegaSimulationEngine
        engine = OmegaSimulationEngine()
        ctx = {"off_rating": 112.0, "def_rating": 108.0, "pace": 100.0}

        r1 = engine.run_fast_game_simulation(
            "Lakers", "Celtics", "NBA", n_iterations=500,
            home_context=ctx, away_context=ctx, seed=42,
        )
        r2 = engine.run_fast_game_simulation(
            "Lakers", "Celtics", "NBA", n_iterations=500,
            home_context=ctx, away_context=ctx, seed=42,
        )
        assert r1["home_win_prob"] == r2["home_win_prob"]
        assert r1["predicted_total"] == r2["predicted_total"]

    def test_different_seeds_differ(self):
        from omega.core.simulation.engine import OmegaSimulationEngine
        engine = OmegaSimulationEngine()
        ctx = {"off_rating": 112.0, "def_rating": 108.0, "pace": 100.0}

        r1 = engine.run_fast_game_simulation(
            "Lakers", "Celtics", "NBA", n_iterations=1000,
            home_context=ctx, away_context=ctx, seed=42,
        )
        r2 = engine.run_fast_game_simulation(
            "Lakers", "Celtics", "NBA", n_iterations=1000,
            home_context=ctx, away_context=ctx, seed=999,
        )
        # With enough iterations, different seeds should produce different results
        # (not guaranteed but overwhelmingly likely)
        results_differ = (
            r1["home_win_prob"] != r2["home_win_prob"]
            or r1["predicted_total"] != r2["predicted_total"]
        )
        assert results_differ, "Different seeds produced identical results (extremely unlikely)"


# ---------------------------------------------------------------------------
# Strict validation tests
# ---------------------------------------------------------------------------

class TestStrictValidation:
    """Test strict=True validation mode."""

    def test_strict_rejects_invalid_values(self):
        ctx = {"off_rating": "garbage", "def_rating": 108.0, "pace": 100.0}
        with pytest.raises(ValueError, match="non-numeric"):
            validate_sim_context(ctx, "NBA", "home", strict=True)

    def test_strict_rejects_out_of_bounds(self):
        ctx = {"off_rating": 999.0, "def_rating": 108.0, "pace": 100.0}
        with pytest.raises(ValueError, match="outside bounds"):
            validate_sim_context(ctx, "NBA", "home", strict=True)

    def test_strict_collects_all_violations(self):
        ctx = {"off_rating": "bad", "def_rating": float("nan"), "pace": 999.0}
        with pytest.raises(ValueError) as exc_info:
            validate_sim_context(ctx, "NBA", "home", strict=True)
        msg = str(exc_info.value)
        assert "3 violation(s)" in msg

    def test_strict_minimum_data_threshold(self):
        ctx = {"off_rating": 112.0}  # Only 1 valid key
        with pytest.raises(ValueError, match="insufficient valid data"):
            validate_sim_context(ctx, "NBA", "home", strict=True)

    def test_permissive_drops_silently(self):
        ctx = {"off_rating": "garbage", "def_rating": 108.0, "pace": 100.0}
        result = validate_sim_context(ctx, "NBA", "home", strict=False)
        assert "off_rating" not in result
        assert result["def_rating"] == 108.0

    def test_strict_no_context_raises(self):
        with pytest.raises(ValueError, match="no context data"):
            validate_sim_context(None, "NBA", "home", strict=True)

    def test_strict_passes_with_good_data(self):
        ctx = {"off_rating": 112.0, "def_rating": 108.0, "pace": 100.0}
        result = validate_sim_context(ctx, "NBA", "home", strict=True)
        assert result["off_rating"] == 112.0
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Downgrade tracking tests
# ---------------------------------------------------------------------------

class TestDowngradeTracking:
    """Test that quality gate populates downgrades on AnswerPlan."""

    def test_downgrades_empty_when_quality_sufficient(self):
        plan = _make_plan()
        facts = [
            _make_fact(_make_slot("home_team.team_stat", InputImportance.CRITICAL), quality=0.9),
            _make_fact(_make_slot("away_team.team_stat", InputImportance.CRITICAL), quality=0.9),
            _make_fact(_make_slot("odds", InputImportance.IMPORTANT, data_type="odds"), quality=0.8),
        ]
        revised = apply_quality_gate(plan, facts)
        assert revised.downgrades == []

    def test_downgrades_bet_card_on_missing_critical(self):
        plan = _make_plan()
        facts = [
            _make_fact(_make_slot("home_team.team_stat", InputImportance.CRITICAL), filled=False, quality=0.0),
        ]
        revised = apply_quality_gate(plan, facts)
        assert "dropped_bet_card" in revised.downgrades
        assert OutputPackage.BET_CARD not in revised.output_packages

    def test_downgrades_on_empty_facts(self):
        plan = _make_plan()
        revised = apply_quality_gate(plan, [])
        assert "dropped_bet_card" in revised.downgrades

    def test_ultra_low_data_downgrade(self):
        plan = _make_plan()
        facts = [
            _make_fact(_make_slot("x", InputImportance.OPTIONAL), filled=True, quality=0.1),
        ]
        revised = apply_quality_gate(plan, facts)
        assert "ultra_low_data" in revised.downgrades
        assert OutputPackage.LIMITED_CONTEXT_ANSWER in revised.output_packages

    def test_native_sim_to_research_downgrade(self):
        plan = _make_plan()
        # No critical inputs filled, low fill rate
        facts = [
            _make_fact(_make_slot("a", InputImportance.CRITICAL), filled=False, quality=0.0),
            _make_fact(_make_slot("b", InputImportance.CRITICAL), filled=False, quality=0.0),
            _make_fact(_make_slot("c", InputImportance.IMPORTANT), filled=False, quality=0.0),
        ]
        revised = apply_quality_gate(plan, facts)
        assert "native_sim_to_research" in revised.downgrades
        assert ExecutionMode.NATIVE_SIM not in revised.execution_modes


# ---------------------------------------------------------------------------
# Orchestrator trace integration tests
# ---------------------------------------------------------------------------

class TestOrchestratorTrace:
    """Test that the orchestrator populates trace in handle_query response."""

    @patch("omega.reasoning.orchestrator.gather_facts")
    @patch("omega.reasoning.orchestrator.understand")
    def test_trace_in_response(self, mock_understand, mock_gather):
        """Trace dict is present in handle_query response."""
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        mock_understand.return_value = _make_understanding()
        mock_gather.return_value = [
            _make_fact(_make_slot("home_team.team_stat", InputImportance.CRITICAL)),
            _make_fact(_make_slot("away_team.team_stat", InputImportance.CRITICAL)),
        ]

        config = OrchestratorConfig(llm_api_key="")
        orch = Orchestrator(config)
        response = orch.handle_query("Lakers vs Celtics NBA")

        assert "trace" in response
        trace = response["trace"]
        assert trace["prompt"] == "Lakers vs Celtics NBA"
        assert trace["run_id"] is not None
        assert trace["league"] == "NBA"

    @patch("omega.reasoning.orchestrator.gather_facts")
    @patch("omega.reasoning.orchestrator.understand")
    def test_trace_has_stage_timings(self, mock_understand, mock_gather):
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        mock_understand.return_value = _make_understanding()
        mock_gather.return_value = [_make_fact()]

        config = OrchestratorConfig(llm_api_key="")
        orch = Orchestrator(config)
        response = orch.handle_query("Lakers vs Celtics NBA")

        trace = response["trace"]
        assert "understanding" in trace["stage_timings"]
        assert "strategy" in trace["stage_timings"]
        assert trace["total_duration_ms"] > 0

    @patch("omega.reasoning.orchestrator.gather_facts")
    @patch("omega.reasoning.orchestrator.understand")
    def test_trace_captures_error(self, mock_understand, mock_gather):
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        mock_understand.side_effect = RuntimeError("LLM failed")

        config = OrchestratorConfig(llm_api_key="")
        orch = Orchestrator(config)
        response = orch.handle_query("test query")

        assert "trace" in response
        assert response["trace"]["error"] == "LLM failed"
        assert response["trace"]["total_duration_ms"] > 0

    @patch("omega.reasoning.orchestrator.gather_facts")
    @patch("omega.reasoning.orchestrator.understand")
    def test_trace_downgrades_captured(self, mock_understand, mock_gather):
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        mock_understand.return_value = _make_understanding()
        # No filled facts → quality gate will downgrade
        mock_gather.return_value = []

        config = OrchestratorConfig(llm_api_key="")
        orch = Orchestrator(config)
        response = orch.handle_query("Lakers vs Celtics NBA")

        trace = response["trace"]
        assert len(trace["downgrades"]) > 0

    @patch("omega.reasoning.orchestrator.gather_facts")
    @patch("omega.reasoning.orchestrator.understand")
    def test_trace_facts_summary(self, mock_understand, mock_gather):
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        mock_understand.return_value = _make_understanding()
        mock_gather.return_value = [
            _make_fact(_make_slot("home_team.team_stat", InputImportance.CRITICAL), filled=True),
            _make_fact(_make_slot("away_team.team_stat", InputImportance.CRITICAL), filled=False),
        ]

        config = OrchestratorConfig(llm_api_key="")
        orch = Orchestrator(config)
        response = orch.handle_query("Lakers vs Celtics NBA")

        summary = response["trace"]["facts_summary"]
        assert summary["total_slots"] >= 2  # planner may add more slots
        assert summary["filled"] == 1  # only 1 of our 2 mocked facts is filled
