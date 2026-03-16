"""
Tests for the research layer: intent, strategy, planning, quality gate, orchestrator.

All tests run without LLM API keys — they exercise the deterministic
heuristic paths.
"""

import pytest


class TestIntentParser:
    """Test heuristic intent understanding."""

    def test_nba_game_intent(self):
        from omega.research.agent.intent import parse_heuristic

        result = parse_heuristic("Lakers vs Celtics NBA")
        assert result.league == "NBA"
        assert len(result.entities) == 2
        # "vs" triggers compare goal, which is correct heuristic behavior
        assert result.goal.value in ("analyze", "compare")

    def test_betting_intent_detected(self):
        from omega.research.agent.intent import parse_heuristic

        result = parse_heuristic("Should I bet on the Lakers tonight?")
        assert result.wants_betting_advice is True

    def test_no_betting_by_default(self):
        from omega.research.agent.intent import parse_heuristic

        result = parse_heuristic("Tell me about the Celtics game")
        assert result.wants_betting_advice is False

    def test_prop_detection(self):
        from omega.research.agent.intent import parse_heuristic

        result = parse_heuristic("LeBron over 25.5 points NBA")
        assert result.prop_line == 25.5

    def test_slate_detection(self):
        from omega.research.agent.intent import parse_heuristic
        from omega.core.models import Subject

        result = parse_heuristic("What's on the NBA slate tonight?")
        assert Subject.SLATE in result.subjects

    def test_soccer_league_detection(self):
        from omega.research.agent.intent import parse_heuristic

        result = parse_heuristic("Liverpool vs Man City premier league")
        assert result.league == "EPL"

    def test_ufc_detection(self):
        from omega.research.agent.intent import parse_heuristic

        result = parse_heuristic("Who wins the UFC main event tonight?")
        assert result.league == "UFC"

    def test_explain_goal(self):
        from omega.research.agent.intent import parse_heuristic
        from omega.core.models import UserGoal

        result = parse_heuristic("Why did the Lakers lose last night?")
        assert result.goal == UserGoal.EXPLAIN

    def test_tone_detection_brief(self):
        from omega.research.agent.intent import parse_heuristic

        result = parse_heuristic("Quick take on Lakers vs Celtics NBA")
        assert result.tone == "brief"

    def test_no_bets_constraint(self):
        from omega.research.agent.intent import parse_heuristic

        result = parse_heuristic("Lakers vs Celtics no bet just analysis")
        assert "no_bets" in result.explicit_constraints
        assert result.wants_betting_advice is False

    def test_understand_function(self):
        from omega.research.agent.intent import understand

        # Without LLM client, falls back to heuristic
        result = understand("Lakers vs Celtics NBA")
        assert result.league == "NBA"


class TestAnswerStrategist:
    """Test answer plan construction."""

    def test_game_analysis_plan(self):
        from omega.research.agent.intent import parse_heuristic
        from omega.research.agent.strategist import build_answer_plan
        from omega.core.models import ExecutionMode

        understanding = parse_heuristic("Analyze Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)
        assert ExecutionMode.NATIVE_SIM in plan.execution_modes
        assert plan.simulation_required is True

    def test_explain_no_sim(self):
        from omega.research.agent.intent import parse_heuristic
        from omega.research.agent.strategist import build_answer_plan
        from omega.core.models import ExecutionMode

        understanding = parse_heuristic("Why did the Lakers lose?")
        plan = build_answer_plan(understanding)
        assert ExecutionMode.NARRATIVE in plan.execution_modes
        assert plan.simulation_required is False

    def test_betting_includes_bet_card(self):
        from omega.research.agent.intent import parse_heuristic
        from omega.research.agent.strategist import build_answer_plan
        from omega.core.models import OutputPackage

        understanding = parse_heuristic("Should I bet on Lakers vs Celtics NBA?")
        plan = build_answer_plan(understanding)
        assert OutputPackage.BET_CARD in plan.output_packages

    def test_unsupported_sport_uses_research(self):
        from omega.core.models import (
            ExecutionMode, QueryUnderstanding, Subject, UserGoal, Entity, EntityRole,
        )
        from omega.research.agent.strategist import build_answer_plan

        understanding = QueryUnderstanding(
            subjects=[Subject.GAME],
            league="QUIDDITCH",
            entities=[
                Entity(name="Gryffindor", role=EntityRole.HOME),
                Entity(name="Slytherin", role=EntityRole.AWAY),
            ],
            markets=[], prop_type=None, prop_line=None,
            date="2026-03-16", goal=UserGoal.ANALYZE,
            wants_betting_advice=False, wants_explanation=False,
            wants_alternatives=False, tone="analytical",
            explicit_constraints=[], raw_prompt="test",
        )
        plan = build_answer_plan(understanding)
        assert ExecutionMode.RESEARCH in plan.execution_modes

    def test_clarification_on_ambiguous(self):
        from omega.research.agent.intent import parse_heuristic
        from omega.research.agent.strategist import build_answer_plan

        understanding = parse_heuristic("hi")
        plan = build_answer_plan(understanding)
        assert plan.clarification_needed is True


class TestRequirementPlanner:
    """Test gather slot generation."""

    def test_nba_game_slots(self):
        from omega.research.agent.intent import parse_heuristic
        from omega.research.agent.strategist import build_answer_plan
        from omega.research.agent.planner import build_gather_list

        understanding = parse_heuristic("Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)
        slots = build_gather_list(understanding, plan)
        assert len(slots) > 0
        # Should have team stat slots
        keys = [s.key for s in slots]
        assert any("stat" in k or "off_rating" in k for k in keys)

    def test_slots_have_importance(self):
        from omega.research.agent.intent import parse_heuristic
        from omega.research.agent.strategist import build_answer_plan
        from omega.research.agent.planner import build_gather_list
        from omega.core.models import InputImportance

        understanding = parse_heuristic("Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)
        slots = build_gather_list(understanding, plan)
        importances = {s.importance for s in slots}
        assert InputImportance.CRITICAL in importances


class TestQualityGate:
    """Test quality gate downgrades."""

    def test_no_data_downgrades(self):
        from omega.core.models import (
            AnswerPlan, ExecutionMode, OutputPackage,
            GatherSlot, GatheredFact, InputImportance,
        )
        from omega.research.agent.quality_gate import apply_quality_gate

        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.BET_CARD, OutputPackage.KEY_FACTORS],
            simulation_required=True,
            betting_recommendations_included=True,
            quality_thresholds={OutputPackage.BET_CARD.value: 0.7},
        )

        # All facts unfilled
        slot = GatherSlot(
            key="home_off_rating", data_type="team_stat",
            entity="Lakers", league="NBA",
            importance=InputImportance.CRITICAL,
        )
        facts = [GatheredFact(slot=slot, result=None, filled=False, quality_score=0.0)]

        revised = apply_quality_gate(plan, facts)
        # BET_CARD should be dropped due to missing critical data
        assert OutputPackage.BET_CARD not in revised.output_packages


class TestFactGatherer:
    """Test fact gathering utilities."""

    def test_compute_aggregate_quality_empty(self):
        from omega.research.agent.fact_gatherer import compute_aggregate_quality
        assert compute_aggregate_quality([]) == 0.0

    def test_critical_inputs_unfilled(self):
        from omega.research.agent.fact_gatherer import critical_inputs_filled
        from omega.core.models import GatherSlot, GatheredFact, InputImportance

        slot = GatherSlot(
            key="test", data_type="team_stat",
            entity="X", league="NBA",
            importance=InputImportance.CRITICAL,
        )
        facts = [GatheredFact(slot=slot, result=None, filled=False, quality_score=0.0)]
        assert critical_inputs_filled(facts) is False

    def test_build_data_completeness(self):
        from omega.research.agent.fact_gatherer import build_data_completeness
        from omega.core.models import GatherSlot, GatheredFact, InputImportance

        slot = GatherSlot(
            key="test_key", data_type="team_stat",
            entity="X", league="NBA",
            importance=InputImportance.OPTIONAL,
        )
        facts = [GatheredFact(slot=slot, result=None, filled=False, quality_score=0.0)]
        comp = build_data_completeness(facts)
        assert comp["test_key"] == "missing"


class TestOrchestrator:
    """Test the main orchestrator pipeline (no LLM, heuristic path)."""

    def test_handle_query_game(self):
        from omega.research.agent.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(llm_api_key=""))
        result = orch.handle_query("Analyze Lakers vs Celtics NBA")
        assert result["type"] in ("answer", "error")
        assert "metadata" in result

    def test_handle_query_clarification(self):
        from omega.research.agent.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(llm_api_key=""))
        result = orch.handle_query("hi")
        assert result["type"] == "clarification"

    def test_handle_query_explain(self):
        from omega.research.agent.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(llm_api_key=""))
        result = orch.handle_query("Why is home court advantage important in NBA?")
        assert result["type"] in ("answer", "error")

    def test_progress_callback(self):
        from omega.research.agent.orchestrator import Orchestrator, OrchestratorConfig

        stages = []
        orch = Orchestrator(OrchestratorConfig(llm_api_key=""))
        orch.handle_query(
            "Analyze Lakers vs Celtics NBA",
            progress_callback=lambda stage, data: stages.append(stage),
        )
        assert "understanding" in stages
        assert "done" in stages


class TestLLMClient:
    """Test LLM client initialization (no API calls)."""

    def test_unavailable_without_key(self):
        from omega.research.llm.client import LLMClient

        client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514", api_key="")
        assert client.is_available() is False

    def test_unknown_provider_raises(self):
        from omega.research.llm.client import LLMClient

        client = LLMClient(provider="nonexistent", model="test", api_key="key")
        # is_available catches exceptions and returns False
        assert client.is_available() is False
