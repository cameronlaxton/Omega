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
from typing import Optional

from omega.core.models import (
    AnswerPlan,
    ExecutionMode,
    OutputPackage,
    QueryUnderstanding,
    Subject,
    UserGoal,
)
from omega.core.simulation.archetypes import get_archetype, LEAGUE_TO_ARCHETYPE

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


def build_answer_plan(understanding: QueryUnderstanding) -> AnswerPlan:
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

    if has_game_subject and has_archetype:
        # Native simulation path available
        modes.append(ExecutionMode.NATIVE_SIM)
        sim_required = True
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

    return AnswerPlan(
        execution_modes=modes,
        output_packages=packages,
        simulation_required=sim_required,
        betting_recommendations_included=betting_included,
        quality_thresholds=thresholds,
    )
