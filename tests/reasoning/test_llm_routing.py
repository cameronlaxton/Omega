"""
Phase 3B tests: LLM-enhanced routing, adaptive slots, player entity extraction.

All LLM tests mock the client — no real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from omega.core.models import (
    AnswerPlan,
    Entity,
    EntityRole,
    ExecutionMode,
    GatherSlot,
    InputImportance,
    OutputPackage,
    QueryUnderstanding,
    Subject,
    UserGoal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_understanding(**overrides) -> QueryUnderstanding:
    """Build a QueryUnderstanding with sensible defaults, overridden as needed."""
    defaults = dict(
        subjects=[Subject.GAME],
        league="NBA",
        entities=[
            Entity(name="Lakers", role=EntityRole.HOME, entity_type="team"),
            Entity(name="Celtics", role=EntityRole.AWAY, entity_type="team"),
        ],
        goal=UserGoal.ANALYZE,
        wants_betting_advice=False,
        wants_explanation=False,
        wants_alternatives=False,
        tone="analytical",
        explicit_constraints=[],
        raw_prompt="Lakers vs Celtics NBA",
    )
    defaults.update(overrides)
    return QueryUnderstanding(**defaults)


def _mock_llm_client(tool_result=None):
    """Create a mock LLMClient that returns the given tool result."""
    client = MagicMock()
    client.is_available.return_value = True
    client.call_with_tools.return_value = tool_result
    return client


# ===========================================================================
# Router tests
# ===========================================================================

class TestRouterComparisonGuard:
    """Comparison without betting intent routes to RESEARCH, not NATIVE_SIM."""

    def test_comparison_no_betting_routes_to_research(self):
        from omega.reasoning.router import build_answer_plan

        understanding = _make_understanding(
            goal=UserGoal.COMPARE,
            wants_betting_advice=False,
        )
        plan = build_answer_plan(understanding)

        assert ExecutionMode.NATIVE_SIM not in plan.execution_modes
        assert ExecutionMode.RESEARCH in plan.execution_modes
        assert plan.simulation_required is False

    def test_comparison_with_betting_keeps_sim(self):
        from omega.reasoning.router import build_answer_plan

        understanding = _make_understanding(
            goal=UserGoal.COMPARE,
            wants_betting_advice=True,
        )
        plan = build_answer_plan(understanding)

        assert ExecutionMode.NATIVE_SIM in plan.execution_modes
        assert plan.simulation_required is True

    def test_comparison_no_betting_has_key_factors(self):
        from omega.reasoning.router import build_answer_plan

        understanding = _make_understanding(
            goal=UserGoal.COMPARE,
            wants_betting_advice=False,
        )
        plan = build_answer_plan(understanding)

        assert OutputPackage.GAME_BREAKDOWN in plan.output_packages
        assert OutputPackage.KEY_FACTORS in plan.output_packages
        assert OutputPackage.BET_CARD not in plan.output_packages


class TestRouterAmbiguityDetection:
    """Ambiguity detection fires only for the 2 specific patterns."""

    def test_no_entities_with_sim_is_ambiguous(self):
        from omega.reasoning.router import _detect_routing_ambiguity

        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.GAME_BREAKDOWN],
            simulation_required=True,
        )
        understanding = _make_understanding(entities=[])

        result = _detect_routing_ambiguity(understanding, plan)
        assert result == "simulation_selected_but_no_entities"

    def test_summarize_sim_no_betting_is_ambiguous(self):
        from omega.reasoning.router import _detect_routing_ambiguity

        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.COMPACT_SUMMARY],
            simulation_required=True,
        )
        understanding = _make_understanding(
            goal=UserGoal.SUMMARIZE,
            wants_betting_advice=False,
        )

        result = _detect_routing_ambiguity(understanding, plan)
        assert result == "summarize_sim_no_betting"

    def test_normal_analyze_is_not_ambiguous(self):
        from omega.reasoning.router import _detect_routing_ambiguity

        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.GAME_BREAKDOWN],
            simulation_required=True,
        )
        understanding = _make_understanding(goal=UserGoal.ANALYZE)

        result = _detect_routing_ambiguity(understanding, plan)
        assert result is None

    def test_research_mode_is_not_ambiguous(self):
        from omega.reasoning.router import _detect_routing_ambiguity

        plan = AnswerPlan(
            execution_modes=[ExecutionMode.RESEARCH],
            output_packages=[OutputPackage.RESEARCH_REPORT],
        )
        understanding = _make_understanding(entities=[])

        result = _detect_routing_ambiguity(understanding, plan)
        assert result is None


class TestRouterLLMArbitration:
    """LLM arbitration overrides mode when called, preserves plan on failure."""

    def test_llm_overrides_mode(self):
        from omega.reasoning.router import build_answer_plan

        llm = _mock_llm_client(tool_result={
            "execution_mode": "research",
            "include_betting": False,
            "reasoning": "This is a recap query",
        })
        understanding = _make_understanding(
            goal=UserGoal.SUMMARIZE,
            wants_betting_advice=False,
        )
        plan = build_answer_plan(understanding, llm_client=llm)

        assert ExecutionMode.RESEARCH in plan.execution_modes
        assert ExecutionMode.NATIVE_SIM not in plan.execution_modes
        assert OutputPackage.RESEARCH_REPORT in plan.output_packages

    def test_llm_failure_keeps_deterministic(self):
        from omega.reasoning.router import build_answer_plan

        llm = _mock_llm_client(tool_result=None)  # LLM returns nothing
        understanding = _make_understanding(
            goal=UserGoal.SUMMARIZE,
            wants_betting_advice=False,
        )
        plan = build_answer_plan(understanding, llm_client=llm)

        # Deterministic plan stands — SUMMARIZE + sim for NBA
        assert ExecutionMode.NATIVE_SIM in plan.execution_modes

    def test_llm_exception_keeps_deterministic(self):
        from omega.reasoning.router import build_answer_plan

        llm = _mock_llm_client()
        llm.call_with_tools.side_effect = RuntimeError("API error")
        understanding = _make_understanding(
            goal=UserGoal.SUMMARIZE,
            wants_betting_advice=False,
        )
        plan = build_answer_plan(understanding, llm_client=llm)

        assert ExecutionMode.NATIVE_SIM in plan.execution_modes

    def test_no_llm_client_skips_arbitration(self):
        from omega.reasoning.router import build_answer_plan

        understanding = _make_understanding(
            goal=UserGoal.SUMMARIZE,
            wants_betting_advice=False,
        )
        plan = build_answer_plan(understanding, llm_client=None)

        # No LLM → deterministic plan
        assert ExecutionMode.NATIVE_SIM in plan.execution_modes


# ===========================================================================
# Intent tests — player entity extraction
# ===========================================================================

class TestIntentPlayerExtraction:
    """Heuristic parser detects player names when no vs/at pattern found."""

    def test_player_name_detected(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("How is LeBron James performing in the NBA?")
        player_entities = [e for e in result.entities if e.entity_type == "player"]
        assert len(player_entities) >= 1
        assert player_entities[0].name == "LeBron James"

    def test_player_gets_subject_role(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("How is Luka Doncic doing in the NBA?")
        player_entities = [e for e in result.entities if e.entity_type == "player"]
        assert len(player_entities) >= 1
        assert player_entities[0].role == EntityRole.SUBJECT

    def test_no_false_player_for_team_name(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("How are the Lakers doing in the NBA?")
        player_entities = [e for e in result.entities if e.entity_type == "player"]
        # "Lakers" is a single word, shouldn't trigger 2+ capitalized word pattern
        assert len(player_entities) == 0

    def test_player_with_vs_pattern_uses_team_extraction(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Lakers vs Celtics NBA")
        # vs pattern found → no player extraction attempted
        team_entities = [e for e in result.entities if e.entity_type == "team"]
        assert len(team_entities) == 2

    def test_player_entity_triggers_game_subject(self):
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("How is Stephen Curry doing in the NBA lately?")
        assert Subject.GAME in result.subjects
        assert Subject.GENERAL_SPORTS not in result.subjects


# ===========================================================================
# Planner tests — adaptive slots
# ===========================================================================

class TestPlannerFocusDetection:
    """Query focus detection identifies player, aspect, and temporal signals."""

    def test_player_focus_detected(self):
        from omega.reasoning.planner import _detect_query_focus

        understanding = _make_understanding(
            entities=[Entity(name="LeBron James", role=EntityRole.SUBJECT, entity_type="player")],
            raw_prompt="How is LeBron James performing lately?",
        )
        focus = _detect_query_focus(understanding)
        assert focus["focus_type"] == "player"

    def test_defense_aspect_detected(self):
        from omega.reasoning.planner import _detect_query_focus

        understanding = _make_understanding(
            raw_prompt="How is the Celtics defense this season?",
        )
        focus = _detect_query_focus(understanding)
        assert focus["focus_aspect"] == "defense"

    def test_recent_temporal_detected(self):
        from omega.reasoning.planner import _detect_query_focus

        understanding = _make_understanding(
            raw_prompt="How have the Lakers been playing lately?",
        )
        focus = _detect_query_focus(understanding)
        assert focus["temporal"] == "recent"

    def test_no_focus_for_generic_query(self):
        from omega.reasoning.planner import _detect_query_focus

        understanding = _make_understanding(
            raw_prompt="Lakers vs Celtics NBA",
        )
        focus = _detect_query_focus(understanding)
        assert focus["focus_aspect"] is None
        assert focus["temporal"] is None


class TestPlannerAdaptiveSlots:
    """Adaptive slots add player data when player focus detected."""

    def test_player_entity_gets_player_slots(self):
        from omega.reasoning.planner import build_gather_list

        understanding = _make_understanding(
            entities=[Entity(name="LeBron James", role=EntityRole.SUBJECT, entity_type="player")],
            raw_prompt="How is LeBron James performing in the NBA?",
        )
        plan = AnswerPlan(
            execution_modes=[ExecutionMode.RESEARCH],
            output_packages=[OutputPackage.RESEARCH_REPORT],
        )
        slots = build_gather_list(understanding, plan)

        data_types = {s.data_type for s in slots}
        assert "player_stat" in data_types
        assert "player_game_log" in data_types

    def test_defense_focus_sets_hint_on_team_stat(self):
        from omega.reasoning.planner import build_gather_list

        understanding = _make_understanding(
            raw_prompt="How is the Celtics defense in the NBA?",
        )
        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.GAME_BREAKDOWN],
            simulation_required=True,
        )
        slots = build_gather_list(understanding, plan)

        team_stat_slots = [s for s in slots if s.data_type == "team_stat"]
        assert len(team_stat_slots) > 0
        assert all(s.focus_hint == "defense" for s in team_stat_slots)

    def test_no_duplicate_player_slots(self):
        from omega.reasoning.planner import build_gather_list

        understanding = _make_understanding(
            entities=[Entity(name="LeBron James", role=EntityRole.SUBJECT, entity_type="player")],
            subjects=[Subject.PLAYER_PROP],
            prop_type="pts",
            raw_prompt="LeBron James over 25.5 points NBA",
        )
        plan = AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.BET_CARD],
            simulation_required=True,
            betting_recommendations_included=True,
        )
        slots = build_gather_list(understanding, plan)

        # Deduplication should prevent duplicate keys
        keys = [s.key for s in slots]
        assert len(keys) == len(set(keys))


# ===========================================================================
# Integration tests — full pipeline
# ===========================================================================

class TestRoutingIntegration:
    """End-to-end: intent → router → planner produces correct results."""

    def test_comparison_query_no_bet_card(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.reasoning.planner import build_gather_list

        understanding = parse_heuristic("Compare Lakers versus Celtics defense NBA")
        plan = build_answer_plan(understanding)
        slots = build_gather_list(understanding, plan)

        # "compare" and "versus" trigger COMPARE goal
        assert understanding.goal == UserGoal.COMPARE
        assert OutputPackage.BET_CARD not in plan.output_packages
        assert ExecutionMode.RESEARCH in plan.execution_modes
        # Should still gather data for analysis
        assert len(slots) > 0

    def test_player_query_gets_player_data(self):
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan
        from omega.reasoning.planner import build_gather_list

        understanding = parse_heuristic("How is LeBron James performing in the NBA lately?")
        plan = build_answer_plan(understanding)
        slots = build_gather_list(understanding, plan)

        # Should have player slots
        data_types = {s.data_type for s in slots}
        assert "player_stat" in data_types or "player_game_log" in data_types

    def test_college_football_still_works(self):
        """Regression: ensure college football intent detection still works."""
        from omega.reasoning.intent import parse_heuristic

        result = parse_heuristic("Alabama vs Georgia college football")
        assert result.league == "NCAAF"

    def test_standard_game_analysis_unchanged(self):
        """Regression: standard game analysis routing is unaffected."""
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan

        understanding = parse_heuristic("Analyze Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)

        # "Analyze" signal → ANALYZE goal → NATIVE_SIM
        assert understanding.goal == UserGoal.ANALYZE
        assert ExecutionMode.NATIVE_SIM in plan.execution_modes
        assert plan.simulation_required is True

    def test_vs_matchup_without_analyze_defaults_to_sim(self):
        """'vs' is a matchup indicator, not a comparison signal."""
        from omega.reasoning.intent import parse_heuristic
        from omega.reasoning.router import build_answer_plan

        understanding = parse_heuristic("Lakers vs Celtics NBA")
        plan = build_answer_plan(understanding)

        # "vs" no longer triggers COMPARE; defaults to ANALYZE → NATIVE_SIM
        assert understanding.goal == UserGoal.ANALYZE
        assert ExecutionMode.NATIVE_SIM in plan.execution_modes
