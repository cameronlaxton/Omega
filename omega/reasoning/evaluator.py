"""
Quality Gate — revises the AnswerPlan based on actual data quality.

Principle: degrade the answer type before degrading the math.

If the original plan called for native_sim + bet_card but the data
quality is insufficient, do NOT simulate with defaults. Instead,
drop bet_card, switch to research mode, and keep key_factors.

This module runs AFTER data gathering and BEFORE execution.
"""

from __future__ import annotations

import logging
from typing import List

from omega.core.models import (
    AnswerPlan,
    ExecutionMode,
    GatheredFact,
    InputImportance,
    OutputPackage,
)
from omega.reasoning.gatherer import (
    compute_aggregate_quality,
    critical_inputs_filled,
    important_inputs_filled,
)

logger = logging.getLogger("omega.agent.quality_gate")


def apply_quality_gate(
    plan: AnswerPlan,
    facts: List[GatheredFact],
) -> AnswerPlan:
    """Revise an AnswerPlan based on actual gathered data quality.

    Returns a new AnswerPlan (possibly identical) with adjustments:
    - Drops bet_card if critical inputs missing or quality < threshold
    - Downgrades native_sim to research if data is insufficient
    - Switches to limited_context_answer if almost no data
    """
    aggregate_quality = compute_aggregate_quality(facts)
    has_critical = critical_inputs_filled(facts)
    has_important = important_inputs_filled(facts)

    # If no facts at all, nothing is filled — override vacuous-truth results
    if not facts:
        has_critical = False
        has_important = False

    # Start with a copy of the current plan values
    modes = list(plan.execution_modes)
    packages = list(plan.output_packages)
    sim_required = plan.simulation_required
    betting_included = plan.betting_recommendations_included
    thresholds = dict(plan.quality_thresholds)

    # -----------------------------------------------------------------
    # Gate 1: Check bet_card threshold
    # -----------------------------------------------------------------
    bet_card_threshold = thresholds.get(OutputPackage.BET_CARD.value, 0.7)
    if OutputPackage.BET_CARD in packages:
        if not has_critical or aggregate_quality < bet_card_threshold:
            logger.info(
                "Quality gate: dropping bet_card (critical_filled=%s, quality=%.2f, threshold=%.2f)",
                has_critical, aggregate_quality, bet_card_threshold,
            )
            packages = [p for p in packages if p != OutputPackage.BET_CARD]
            betting_included = False

            # Also drop alternative_bets if bet_card is gone
            packages = [p for p in packages if p != OutputPackage.ALTERNATIVE_BETS]

            # Add research_report as replacement if not already present
            if OutputPackage.RESEARCH_REPORT not in packages:
                packages.append(OutputPackage.RESEARCH_REPORT)

    # -----------------------------------------------------------------
    # Gate 2: Check game_breakdown threshold (with sim data)
    # -----------------------------------------------------------------
    breakdown_threshold = thresholds.get(OutputPackage.GAME_BREAKDOWN.value, 0.5)
    if OutputPackage.GAME_BREAKDOWN in packages and sim_required:
        if not has_critical or aggregate_quality < breakdown_threshold:
            logger.info(
                "Quality gate: game_breakdown switching to narrative-only (quality=%.2f)",
                aggregate_quality,
            )
            # Keep game_breakdown but mark sim as not required —
            # the response composer will produce narrative analysis instead
            sim_required = False

    # -----------------------------------------------------------------
    # Gate 3: Native sim mode feasibility
    # -----------------------------------------------------------------
    if ExecutionMode.NATIVE_SIM in modes:
        if not has_critical:
            # Count what percentage of critical+important we have
            crit_important = [f for f in facts
                              if f.slot.importance in (InputImportance.CRITICAL, InputImportance.IMPORTANT)]
            filled_count = sum(1 for f in crit_important if f.filled)
            total_count = len(crit_important) if crit_important else 1
            fill_rate = filled_count / total_count

            if fill_rate >= 0.5:
                # Directional sim only — switch to MIXED
                logger.info("Quality gate: downgrading native_sim to mixed (fill_rate=%.2f)", fill_rate)
                modes = [ExecutionMode.MIXED if m == ExecutionMode.NATIVE_SIM else m for m in modes]
                sim_required = False  # no formal edges
                betting_included = False
                # Drop bet_card if still present
                packages = [p for p in packages if p != OutputPackage.BET_CARD]
            else:
                # Not enough data for any simulation
                logger.info("Quality gate: downgrading native_sim to research (fill_rate=%.2f)", fill_rate)
                modes = [ExecutionMode.RESEARCH if m == ExecutionMode.NATIVE_SIM else m for m in modes]
                sim_required = False
                betting_included = False
                packages = [p for p in packages if p != OutputPackage.BET_CARD]
                if OutputPackage.RESEARCH_REPORT not in packages:
                    packages.append(OutputPackage.RESEARCH_REPORT)

        elif not has_important:
            # Critical inputs present but some important missing — sim runs
            # but confidence is capped at C. Keep native_sim, just log it.
            logger.info("Quality gate: native_sim allowed with confidence cap (important inputs missing)")

    # -----------------------------------------------------------------
    # Gate 4: Ultra-low data — switch to limited_context_answer
    # -----------------------------------------------------------------
    filled_facts = [f for f in facts if f.filled]
    if len(filled_facts) < 3 and aggregate_quality < 0.3:
        logger.info("Quality gate: ultra-low data (filled=%d, quality=%.2f), switching to limited_context_answer",
                     len(filled_facts), aggregate_quality)
        packages = [OutputPackage.LIMITED_CONTEXT_ANSWER]
        if ExecutionMode.NATIVE_SIM in modes or ExecutionMode.MIXED in modes:
            modes = [ExecutionMode.RESEARCH if m in (ExecutionMode.NATIVE_SIM, ExecutionMode.MIXED) else m
                     for m in modes]
        sim_required = False
        betting_included = False

    # -----------------------------------------------------------------
    # Build the revised plan
    # -----------------------------------------------------------------
    revised = AnswerPlan(
        execution_modes=modes,
        output_packages=packages,
        simulation_required=sim_required,
        betting_recommendations_included=betting_included,
        quality_thresholds=thresholds,
        clarification_needed=plan.clarification_needed,
        clarification_question=plan.clarification_question,
    )

    if revised != plan:
        logger.info("Quality gate revised the answer plan: modes=%s, packages=%s",
                     [m.value for m in revised.execution_modes],
                     [p.value for p in revised.output_packages])

    return revised
