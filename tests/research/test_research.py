"""
Tests for the research layer: intent, strategy, planning, quality gate, orchestrator.

All tests run without LLM API keys — they exercise the deterministic
heuristic paths.
"""

import pytest


class TestIntentParser:
    """Test heuristic intent understanding."""

    def test_nba_game_intent(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Lakers vs Celtics NBA")
        assert result.league == "NBA"
        assert len(result.entities) == 2
        # "vs" triggers compare goal, which is correct heuristic behavior
        assert result.goal.value in ("analyze", "compare")

    def test_betting_intent_detected(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Should I bet on the Lakers tonight?")
        assert result.wants_betting_advice is True

    def test_no_betting_by_default(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Tell me about the Celtics game")
        assert result.wants_betting_advice is False

    def test_prop_detection(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("LeBron over 25.5 points NBA")
        assert result.prop_line == 25.5

    def test_slate_detection(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.core.models import Subject

        result = parse_heuristic("What's on the NBA slate tonight?")
        assert Subject.SLATE in result.subjects

    def test_soccer_league_detection(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Liverpool vs Man City premier league")
        assert result.league == "EPL"

    def test_ufc_detection(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Who wins the UFC main event tonight?")
        assert result.league == "UFC"

    def test_explain_goal(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.core.models import UserGoal

        result = parse_heuristic("Why did the Lakers lose last night?")
        assert result.goal == UserGoal.EXPLAIN

    def test_tone_detection_brief(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Quick take on Lakers vs Celtics NBA")
        assert result.tone == "brief"

    def test_no_bets_constraint(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Lakers vs Celtics no bet just analysis")
        assert "no_bets" in result.explicit_constraints
        assert result.wants_betting_advice is False

    def test_understand_function(self):
        from omega.reasoning.intent import understand

        # Without LLM client, falls back to heuristic
        result = understand("Lakers vs Celtics NBA")
        assert result.league == "NBA"


class TestAnswerStrategist:
    """Test answer plan construction."""

    def test_game_analysis_plan(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.core.models import ExecutionMode

        understanding = parse_heuristic("Analyze Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)
        assert ExecutionMode.NATIVE_SIM in plan.execution_modes
        assert plan.simulation_required is True

    def test_explain_no_sim(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.core.models import ExecutionMode

        understanding = parse_heuristic("Why did the Lakers lose?")
        plan = build_answer_plan(understanding)
        assert ExecutionMode.NARRATIVE in plan.execution_modes
        assert plan.simulation_required is False

    def test_betting_includes_bet_card(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.core.models import OutputPackage

        understanding = parse_heuristic("Should I bet on Lakers vs Celtics NBA?")
        plan = build_answer_plan(understanding)
        assert OutputPackage.BET_CARD in plan.output_packages

    def test_unsupported_sport_uses_research(self):
        from omega.core.models import (
            ExecutionMode, QueryUnderstanding, Subject, UserGoal, Entity, EntityRole,
        )
        from omega.reasoning.router import build_answer_plan

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
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan

        understanding = parse_heuristic("hi")
        plan = build_answer_plan(understanding)
        assert plan.clarification_needed is True


class TestRequirementPlanner:
    """Test gather slot generation."""

    def test_nba_game_slots(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.reasoning.planner import build_gather_list

        understanding = parse_heuristic("Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)
        slots = build_gather_list(understanding, plan)
        assert len(slots) > 0
        # Should have team stat slots
        keys = [s.key for s in slots]
        assert any("stat" in k or "off_rating" in k for k in keys)

    def test_slots_have_importance(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.reasoning.planner import build_gather_list
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
        from omega.reasoning.evaluator import apply_quality_gate

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
        from omega.reasoning.gatherer import compute_aggregate_quality
        assert compute_aggregate_quality([]) == 0.0

    def test_critical_inputs_unfilled(self):
        from omega.reasoning.gatherer import critical_inputs_filled
        from omega.core.models import GatherSlot, GatheredFact, InputImportance

        slot = GatherSlot(
            key="test", data_type="team_stat",
            entity="X", league="NBA",
            importance=InputImportance.CRITICAL,
        )
        facts = [GatheredFact(slot=slot, result=None, filled=False, quality_score=0.0)]
        assert critical_inputs_filled(facts) is False

    def test_build_data_completeness(self):
        from omega.reasoning.gatherer import build_data_completeness
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
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(llm_api_key=""))
        result = orch.handle_query("Analyze Lakers vs Celtics NBA")
        assert result["type"] in ("answer", "error")
        assert "metadata" in result

    def test_handle_query_clarification(self):
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(llm_api_key=""))
        result = orch.handle_query("hi")
        assert result["type"] == "clarification"

    def test_handle_query_explain(self):
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(llm_api_key=""))
        result = orch.handle_query("Why is home court advantage important in NBA?")
        assert result["type"] in ("answer", "error")

    def test_progress_callback(self):
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

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
        from omega.reasoning.llm.client import LLMClient

        client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514", api_key="")
        assert client.is_available() is False

    def test_unknown_provider_raises(self):
        from omega.reasoning.llm.client import LLMClient

        client = LLMClient(provider="nonexistent", model="test", api_key="key")
        # is_available catches exceptions and returns False
        assert client.is_available() is False


# ===================================================================
# Phase 3 tests — comprehensive coverage
# ===================================================================


class TestIntentEdgeCases:
    """Edge cases for heuristic intent parsing."""

    def test_at_pattern_extraction(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Lakers at Celtics NBA")
        assert result.league == "NBA"
        assert len(result.entities) == 2

    def test_college_football_over_nfl(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Texas vs Oklahoma college football")
        assert result.league == "NCAAF"

    def test_college_basketball_over_nba(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Duke vs UNC college basketball")
        assert result.league == "NCAAB"

    def test_plain_football_defaults_to_nfl(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Who wins the football game tonight?")
        assert result.league == "NFL"

    def test_case_insensitive_league(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Lakers vs Celtics nba")
        assert result.league == "NBA"

    def test_three_teams_extracts_first_two(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Lakers vs Celtics vs Warriors NBA")
        # regex captures first "X vs Y" pair
        assert len(result.entities) == 2

    def test_no_teams_no_league(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.core.models import Subject

        result = parse_heuristic("What is a moneyline?")
        assert result.league is None
        assert Subject.GENERAL_SPORTS in result.subjects

    def test_champions_league_detection(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Real Madrid vs PSG champions league")
        assert result.league == "CHAMPIONS_LEAGUE"


class TestComposerPackages:
    """Verify all 11 OutputPackage types produce sections in compose_response."""

    def _make_fixtures(self, packages, **kwargs):
        """Build minimal fixtures for compose_response."""
        from omega.core.models import (
            AnswerPlan, ExecutionMode, ExecutionResult, OutputPackage,
            QueryUnderstanding, Subject, UserGoal, GatherSlot, GatheredFact,
            InputImportance, ProviderResult,
        )
        from datetime import datetime, timezone

        understanding = QueryUnderstanding(
            subjects=[Subject.GAME], league="NBA",
            goal=UserGoal.ANALYZE, raw_prompt="test prompt",
        )
        plan = AnswerPlan(
            execution_modes=[ExecutionMode.RESEARCH],
            output_packages=packages,
        )
        slot = GatherSlot(
            key="test.stat", data_type="team_stat",
            entity="Team", league="NBA", importance=InputImportance.IMPORTANT,
        )
        facts = [GatheredFact(
            slot=slot,
            result=ProviderResult(
                data={"off_rating": 110.0}, source="test",
                fetched_at=datetime.now(timezone.utc), confidence=0.9,
            ),
            filled=True, quality_score=0.9,
        )]
        execution = ExecutionResult(
            mode=ExecutionMode.RESEARCH,
            simulation=kwargs.get("simulation"),
            edges=kwargs.get("edges", []),
            best_bet=kwargs.get("best_bet"),
            data_quality_score=0.8,
        )
        return understanding, plan, facts, execution

    def test_key_factors(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures([OutputPackage.KEY_FACTORS])
        result = compose_response(u, p, f, e)
        assert "key_factors" in result["sections"]

    def test_research_report(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures([OutputPackage.RESEARCH_REPORT])
        result = compose_response(u, p, f, e)
        assert "research" in result["sections"]

    def test_plain_explanation(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures([OutputPackage.PLAIN_EXPLANATION])
        result = compose_response(u, p, f, e)
        assert "test prompt" in result["text"]

    def test_compact_summary(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures([OutputPackage.COMPACT_SUMMARY])
        result = compose_response(u, p, f, e)
        assert "compact_summary" in result["sections"]

    def test_limited_context_answer(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures([OutputPackage.LIMITED_CONTEXT_ANSWER])
        result = compose_response(u, p, f, e)
        assert "limited_context" in result["sections"]
        assert "caveat" in result["sections"]["limited_context"]

    def test_bankroll_guidance(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures([OutputPackage.BANKROLL_GUIDANCE])
        result = compose_response(u, p, f, e)
        assert "bankroll_guidance" in result["sections"]

    def test_news_digest(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures([OutputPackage.NEWS_DIGEST])
        result = compose_response(u, p, f, e)
        assert "news_digest" in result["sections"]
        assert "items" in result["sections"]["news_digest"]

    def test_scenario_analysis(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures([OutputPackage.SCENARIO_ANALYSIS])
        result = compose_response(u, p, f, e)
        assert "scenario_analysis" in result["sections"]

    def test_alternative_bets(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures(
            [OutputPackage.ALTERNATIVE_BETS],
            edges=[{"side": "home", "edge_pct": 3.5}, {"side": "away", "edge_pct": 1.2}],
        )
        result = compose_response(u, p, f, e)
        assert "alternative_bets" in result["sections"]
        assert len(result["sections"]["alternative_bets"]["edges"]) == 1

    def test_bet_card_with_best_bet(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures(
            [OutputPackage.BET_CARD],
            best_bet={"side": "home", "edge_pct": 5.0, "kelly_fraction": 0.03},
            edges=[{"side": "home", "edge_pct": 5.0}],
        )
        result = compose_response(u, p, f, e)
        assert "bet_card" in result["sections"]

    def test_game_breakdown_with_sim(self):
        from omega.synthesis.composer import compose_response
        from omega.core.models import OutputPackage

        u, p, f, e = self._make_fixtures(
            [OutputPackage.GAME_BREAKDOWN],
            simulation={"home_team": "Lakers", "away_team": "Celtics",
                        "home_win_prob": 55, "away_win_prob": 45},
        )
        result = compose_response(u, p, f, e)
        assert "game_breakdown" in result["sections"]


class TestQualityGateEdgeCases:
    """Edge cases for quality gate."""

    def test_empty_facts_drops_bet_card(self):
        from omega.core.models import (
            AnswerPlan, ExecutionMode, OutputPackage,
        )
        from omega.reasoning.evaluator import apply_quality_gate

        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.BET_CARD, OutputPackage.KEY_FACTORS],
            simulation_required=True,
            betting_recommendations_included=True,
            quality_thresholds={OutputPackage.BET_CARD.value: 0.7},
        )
        revised = apply_quality_gate(plan, [])
        assert OutputPackage.BET_CARD not in revised.output_packages

    def test_all_unfilled_drops_bet_card(self):
        from omega.core.models import (
            AnswerPlan, ExecutionMode, OutputPackage,
            GatherSlot, GatheredFact, InputImportance,
        )
        from omega.reasoning.evaluator import apply_quality_gate

        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.BET_CARD],
            simulation_required=True,
            betting_recommendations_included=True,
            quality_thresholds={OutputPackage.BET_CARD.value: 0.7},
        )
        slots = [
            GatherSlot(key=f"s{i}", data_type="team_stat", entity="X",
                       league="NBA", importance=InputImportance.CRITICAL)
            for i in range(3)
        ]
        facts = [GatheredFact(slot=s, result=None, filled=False, quality_score=0.0)
                 for s in slots]
        revised = apply_quality_gate(plan, facts)
        assert OutputPackage.BET_CARD not in revised.output_packages

    def test_ultra_low_data_limited_context(self):
        from omega.core.models import (
            AnswerPlan, ExecutionMode, OutputPackage,
            GatherSlot, GatheredFact, InputImportance,
        )
        from omega.reasoning.evaluator import apply_quality_gate

        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.KEY_FACTORS],
            simulation_required=True,
            quality_thresholds={},
        )
        slot = GatherSlot(key="s1", data_type="team_stat", entity="X",
                          league="NBA", importance=InputImportance.OPTIONAL)
        facts = [GatheredFact(slot=slot, result=None, filled=False, quality_score=0.0)]
        revised = apply_quality_gate(plan, facts)
        assert OutputPackage.LIMITED_CONTEXT_ANSWER in revised.output_packages


class TestGathererHelpers:
    """Test gatherer helper function edge cases."""

    def test_critical_inputs_filled_no_critical_slots(self):
        from omega.reasoning.gatherer import critical_inputs_filled
        from omega.core.models import GatherSlot, GatheredFact, InputImportance

        slot = GatherSlot(key="opt", data_type="team_stat", entity="X",
                          league="NBA", importance=InputImportance.OPTIONAL)
        facts = [GatheredFact(slot=slot, result=None, filled=True, quality_score=0.5)]
        # No CRITICAL slots -> should return False (not vacuous True)
        assert critical_inputs_filled(facts) is False

    def test_critical_inputs_filled_empty_list(self):
        from omega.reasoning.gatherer import critical_inputs_filled
        assert critical_inputs_filled([]) is False

    def test_important_inputs_filled_no_important_slots(self):
        from omega.reasoning.gatherer import important_inputs_filled
        from omega.core.models import GatherSlot, GatheredFact, InputImportance

        slot = GatherSlot(key="opt", data_type="team_stat", entity="X",
                          league="NBA", importance=InputImportance.OPTIONAL)
        facts = [GatheredFact(slot=slot, result=None, filled=True, quality_score=0.5)]
        assert important_inputs_filled(facts) is False

    def test_important_inputs_filled_empty_list(self):
        from omega.reasoning.gatherer import important_inputs_filled
        assert important_inputs_filled([]) is False


class TestPlannerSlotConsolidation:
    """Verify planner generates one slot per (entity, data_type) for team stats."""

    def test_nba_two_teams_generates_two_team_stat_slots(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.reasoning.planner import build_gather_list

        understanding = parse_heuristic("Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)
        slots = build_gather_list(understanding, plan)

        team_stat_slots = [s for s in slots if s.data_type == "team_stat"]
        # Should have exactly 2 team stat slots (one per team), not N per key
        assert len(team_stat_slots) == 2

    def test_team_stat_slot_is_critical(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.reasoning.planner import build_gather_list
        from omega.core.models import InputImportance

        understanding = parse_heuristic("Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)
        slots = build_gather_list(understanding, plan)

        team_stat_slots = [s for s in slots if s.data_type == "team_stat"]
        # NBA archetype has critical_team_keys, so importance should be CRITICAL
        for slot in team_stat_slots:
            assert slot.importance == InputImportance.CRITICAL

    def test_single_team_generates_one_team_stat_slot(self):
        from omega.core.models import (
            QueryUnderstanding, Subject, UserGoal, Entity, EntityRole,
            ExecutionMode, OutputPackage, AnswerPlan,
        )
        from omega.reasoning.planner import build_gather_list

        understanding = QueryUnderstanding(
            subjects=[Subject.GAME], league="NBA",
            entities=[Entity(name="Lakers", role=EntityRole.HOME)],
            goal=UserGoal.ANALYZE, raw_prompt="test",
        )
        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.GAME_BREAKDOWN],
            simulation_required=True,
        )
        slots = build_gather_list(understanding, plan)
        team_stat_slots = [s for s in slots if s.data_type == "team_stat"]
        assert len(team_stat_slots) == 1
