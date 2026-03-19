"""
Answer Strategist — decides what kind of answer to produce.

Given a QueryUnderstanding, this module determines:
  - Which execution modes to use (native sim, research, bankroll calc, etc.)
  - Which output packages to include in the response
  - Whether simulation and/or betting recommendations are appropriate
  - Whether clarification is needed
  - Data quality thresholds per output package

The strategist runs BEFORE any data gathering or computation. It may be
revised later by the quality gate if data quality doesn't meet thresholds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from omega.core.models import (
    AnswerPlan,
    ExecutionMode,
    OutputPackage,
    QueryUnderstanding,
    Subject,
    UserGoal,
)
from omega.core.simulation.archetypes import get_archetype, LEAGUE_TO_ARCHETYPE

if TYPE_CHECKING:
    from omega.reasoning.llm.client import LLMClient

logger = logging.getLogger("omega.agent.strategist")


def _has_native_archetype(league: Optional[str]) -> bool:
    """Check if a league has a native simulation archetype."""
    if league is None:
        return False
    return league.upper() in LEAGUE_TO_ARCHETYPE


def _needs_clarification(understanding: QueryUnderstanding) -> tuple[bool, Optional[str]]:
    """Determine if the prompt is too ambiguous to produce a useful answer.

    Only ask when ambiguity would materially change the answer.
    """
    # No entities and no league — truly ambiguous
    if not understanding.entities and not understanding.league and understanding.goal not in (
        UserGoal.LEARN, UserGoal.DISCUSS
    ):
        # Check if there's enough context in the raw prompt
        prompt_lower = understanding.raw_prompt.lower()
        # Single-team mentions without league can often be resolved
        if len(prompt_lower.split()) <= 2:
            return True, "Which sport or matchup are you asking about?"

    # Ambiguous multi-team-city names would be caught by entity resolution later,
    # not at the strategy level.

    return False, None


# ---------------------------------------------------------------------------
# LLM arbitration for narrowly-scoped ambiguous cases
# ---------------------------------------------------------------------------

ROUTE_ARBITRATION_TOOL: Dict[str, Any] = {
    "name": "select_route",
    "description": "Select the correct execution mode for an ambiguous sports query",
    "input_schema": {
        "type": "object",
        "properties": {
            "execution_mode": {
                "type": "string",
                "enum": ["native_sim", "research", "narrative", "mixed"],
            },
            "include_betting": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["execution_mode", "include_betting"],
    },
}


def _detect_routing_ambiguity(
    understanding: QueryUnderstanding, plan: AnswerPlan,
) -> Optional[str]:
    """Check for two specific misroute patterns. Returns reason or None.

    Only fires for narrowly-defined cases where deterministic routing
    is likely wrong. This is NOT a general-purpose confidence check.
    """
    has_sim = ExecutionMode.NATIVE_SIM in plan.execution_modes

    # Pattern 1: NATIVE_SIM selected but zero entities — can't build team contexts
    if has_sim and not understanding.entities:
        return "simulation_selected_but_no_entities"

    # Pattern 2: SUMMARIZE + NATIVE_SIM + no betting — likely a recap, not a future game
    if (has_sim
            and understanding.goal == UserGoal.SUMMARIZE
            and not understanding.wants_betting_advice):
        return "summarize_sim_no_betting"

    return None


def _llm_arbitrate_mode(
    understanding: QueryUnderstanding,
    deterministic_plan: AnswerPlan,
    ambiguity_reason: str,
    llm_client: "LLMClient",
) -> Optional[AnswerPlan]:
    """Ask the LLM to pick the right mode for an ambiguous case.

    Returns a revised AnswerPlan if the LLM overrides, or None to keep
    the deterministic plan.
    """
    system = (
        "You are a routing arbiter for a sports analytics system. "
        "A query was classified but the routing is ambiguous. "
        "Pick the correct execution mode.\n\n"
        "Modes:\n"
        "- native_sim: Full Monte Carlo simulation + betting edges (use for future games where user wants quantitative analysis)\n"
        "- research: Fact gathering + narrative (use for recaps, comparisons without betting, general info)\n"
        "- narrative: Pure explanation (use for educational/conceptual questions)\n"
        "- mixed: Light simulation for context + narrative (use when sim adds value but no formal edges needed)\n"
    )
    user_msg = (
        f"Query: {understanding.raw_prompt}\n"
        f"Detected goal: {understanding.goal.value}\n"
        f"Betting intent: {understanding.wants_betting_advice}\n"
        f"Entities: {[e.name for e in understanding.entities]}\n"
        f"Current routing: {[m.value for m in deterministic_plan.execution_modes]}\n"
        f"Ambiguity: {ambiguity_reason}\n"
    )

    result = llm_client.call_with_tools(
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        tools=[ROUTE_ARBITRATION_TOOL],
    )
    if result is None:
        return None

    try:
        mode = ExecutionMode(result["execution_mode"])
        include_betting = result.get("include_betting", False)
    except (KeyError, ValueError):
        return None

    # Only override if the LLM picked a different mode
    if mode in deterministic_plan.execution_modes:
        return None

    # Rebuild plan with the LLM-selected mode
    revised = deterministic_plan.model_copy(deep=True)
    revised.execution_modes = [mode]
    revised.simulation_required = mode in (ExecutionMode.NATIVE_SIM, ExecutionMode.MIXED)
    if not include_betting:
        revised.output_packages = [
            p for p in revised.output_packages
            if p not in (OutputPackage.BET_CARD, OutputPackage.ALTERNATIVE_BETS)
        ]
        revised.betting_recommendations_included = False
    if mode == ExecutionMode.RESEARCH and OutputPackage.RESEARCH_REPORT not in revised.output_packages:
        revised.output_packages.append(OutputPackage.RESEARCH_REPORT)
        revised.quality_thresholds[OutputPackage.RESEARCH_REPORT.value] = 0.3

    logger.info("LLM arbitration overrode routing: %s → %s (reason: %s)",
                [m.value for m in deterministic_plan.execution_modes],
                mode.value, ambiguity_reason)
    return revised


def build_answer_plan(
    understanding: QueryUnderstanding,
    llm_client: Optional["LLMClient"] = None,
) -> AnswerPlan:
    """Build an AnswerPlan from a QueryUnderstanding.

    This is the core decision function. It determines what execution modes
    and output packages the response should contain.
    """
    # Check for clarification needs first
    needs_clarify, question = _needs_clarification(understanding)
    if needs_clarify:
        return AnswerPlan(
            execution_modes=[ExecutionMode.NARRATIVE],
            output_packages=[OutputPackage.PLAIN_EXPLANATION],
            clarification_needed=True,
            clarification_question=question,
        )

    has_archetype = _has_native_archetype(understanding.league)
    modes = []
    packages = []
    sim_required = False
    betting_included = False
    thresholds = {}

    # -------------------------------------------------------------------
    # Select execution modes based on goal + subject
    # -------------------------------------------------------------------

    # Goals that never need simulation
    if understanding.goal in (UserGoal.EXPLAIN, UserGoal.LEARN):
        modes.append(ExecutionMode.NARRATIVE)
        packages.append(OutputPackage.PLAIN_EXPLANATION)
        if understanding.goal == UserGoal.EXPLAIN:
            packages.append(OutputPackage.KEY_FACTORS)
        return AnswerPlan(
            execution_modes=modes,
            output_packages=packages,
            quality_thresholds=thresholds,
        )

    # Pure bankroll questions
    if Subject.BANKROLL in understanding.subjects and Subject.GAME not in understanding.subjects:
        modes.append(ExecutionMode.BANKROLL_CALC)
        packages.append(OutputPackage.BANKROLL_GUIDANCE)
        # May add sim context if a slate is mentioned
        if Subject.SLATE in understanding.subjects and has_archetype:
            modes.append(ExecutionMode.NATIVE_SIM)
            sim_required = True
        return AnswerPlan(
            execution_modes=modes,
            output_packages=packages,
            simulation_required=sim_required,
            quality_thresholds=thresholds,
        )

    # Discussion/general — route to research
    if understanding.goal == UserGoal.DISCUSS:
        modes.append(ExecutionMode.RESEARCH)
        packages.append(OutputPackage.RESEARCH_REPORT)
        packages.append(OutputPackage.NEWS_DIGEST)
        thresholds[OutputPackage.RESEARCH_REPORT.value] = 0.3
        return AnswerPlan(
            execution_modes=modes,
            output_packages=packages,
            quality_thresholds=thresholds,
        )

    # -------------------------------------------------------------------
    # Game / Prop / Slate / Comparison — may need simulation
    # -------------------------------------------------------------------

    game_subjects = {Subject.GAME, Subject.PLAYER_PROP, Subject.SLATE, Subject.COMPARISON}
    has_game_subject = bool(game_subjects & set(understanding.subjects))

    # Comparison without betting intent → research, not simulation
    is_non_betting_comparison = (
        understanding.goal == UserGoal.COMPARE
        and not understanding.wants_betting_advice
    )

    if has_game_subject and has_archetype and not is_non_betting_comparison:
        # Native simulation path available
        modes.append(ExecutionMode.NATIVE_SIM)
        sim_required = True
    elif is_non_betting_comparison and has_game_subject:
        # Comparison without betting → research mode
        modes.append(ExecutionMode.RESEARCH)
        sim_required = False
    elif has_game_subject and not has_archetype:
        # Unsupported sport — research mode
        modes.append(ExecutionMode.RESEARCH)
        sim_required = False
    else:
        modes.append(ExecutionMode.RESEARCH)

    # -------------------------------------------------------------------
    # Select output packages based on goal
    # -------------------------------------------------------------------

    goal = understanding.goal

    if goal == UserGoal.DECIDE:
        if sim_required and understanding.wants_betting_advice:
            packages.append(OutputPackage.BET_CARD)
            betting_included = True
            thresholds[OutputPackage.BET_CARD.value] = 0.7
        packages.append(OutputPackage.KEY_FACTORS)
        if understanding.wants_alternatives:
            packages.append(OutputPackage.ALTERNATIVE_BETS)

    elif goal == UserGoal.ANALYZE:
        packages.append(OutputPackage.GAME_BREAKDOWN)
        packages.append(OutputPackage.KEY_FACTORS)
        thresholds[OutputPackage.GAME_BREAKDOWN.value] = 0.5
        # Add bet_card conditionally — only if user wants it AND sim runs
        if understanding.wants_betting_advice and sim_required:
            packages.append(OutputPackage.BET_CARD)
            betting_included = True
            thresholds[OutputPackage.BET_CARD.value] = 0.7

    elif goal == UserGoal.COMPARE:
        packages.append(OutputPackage.GAME_BREAKDOWN)
        packages.append(OutputPackage.KEY_FACTORS)
        thresholds[OutputPackage.GAME_BREAKDOWN.value] = 0.5

    elif goal == UserGoal.SUMMARIZE:
        packages.append(OutputPackage.COMPACT_SUMMARY)
        # Add bet card if edges found and user wants betting
        if understanding.wants_betting_advice and sim_required:
            packages.append(OutputPackage.BET_CARD)
            betting_included = True
            thresholds[OutputPackage.BET_CARD.value] = 0.7

    elif goal == UserGoal.MONITOR:
        packages.append(OutputPackage.COMPACT_SUMMARY)
        if sim_required:
            packages.append(OutputPackage.BET_CARD)
            betting_included = True
            thresholds[OutputPackage.BET_CARD.value] = 0.7
        if understanding.wants_alternatives:
            packages.append(OutputPackage.ALTERNATIVE_BETS)

    else:
        # Fallback: research report
        if OutputPackage.RESEARCH_REPORT not in packages:
            packages.append(OutputPackage.RESEARCH_REPORT)
            thresholds[OutputPackage.RESEARCH_REPORT.value] = 0.3

    # If no archetype, ensure we have a research fallback
    if not has_archetype and OutputPackage.RESEARCH_REPORT not in packages:
        packages.append(OutputPackage.RESEARCH_REPORT)
        thresholds[OutputPackage.RESEARCH_REPORT.value] = 0.3

    # -------------------------------------------------------------------
    # Apply explicit constraints
    # -------------------------------------------------------------------

    if "no_bets" in understanding.explicit_constraints or "analysis_only" in understanding.explicit_constraints:
        packages = [p for p in packages if p not in (
            OutputPackage.BET_CARD, OutputPackage.ALTERNATIVE_BETS,
        )]
        betting_included = False

    # Bankroll guidance if bankroll is a subject alongside game
    if Subject.BANKROLL in understanding.subjects and OutputPackage.BANKROLL_GUIDANCE not in packages:
        packages.append(OutputPackage.BANKROLL_GUIDANCE)
        if ExecutionMode.BANKROLL_CALC not in modes:
            modes.append(ExecutionMode.BANKROLL_CALC)

    plan = AnswerPlan(
        execution_modes=modes,
        output_packages=packages,
        simulation_required=sim_required,
        betting_recommendations_included=betting_included,
        quality_thresholds=thresholds,
    )

    # -------------------------------------------------------------------
    # LLM arbitration — strictly limited to 2 ambiguity patterns
    # -------------------------------------------------------------------
    if llm_client is not None:
        ambiguity = _detect_routing_ambiguity(understanding, plan)
        if ambiguity is not None:
            try:
                revised = _llm_arbitrate_mode(understanding, plan, ambiguity, llm_client)
                if revised is not None:
                    return revised
            except Exception:
                logger.debug("LLM arbitration failed, keeping deterministic plan", exc_info=True)

    return plan
