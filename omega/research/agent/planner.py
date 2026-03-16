"""
Requirement Planner — generates typed gather-slots from an AnswerPlan.

For each execution mode in the AnswerPlan, the planner consults the
sport archetype's required_inputs and optional_inputs to produce a
typed gather-list (List[GatherSlot]).

This module is deterministic — the LLM doesn't decide what data is
needed. The archetype and answer plan dictate the requirements.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from omega.core.models import (
    AnswerPlan,
    ExecutionMode,
    GatherSlot,
    InputImportance,
    OutputPackage,
    QueryUnderstanding,
    Subject,
    FRESHNESS_RULES,
)
from omega.core.simulation.archetypes import get_archetype, SportArchetype

logger = logging.getLogger("omega.agent.requirement_planner")


def _freshness_for(data_type: str) -> float:
    """Return freshness TTL in seconds for a data type."""
    return FRESHNESS_RULES.get(data_type, 86400.0)


def _team_stat_slots(
    entity: str,
    prefix: str,
    league: str,
    archetype: SportArchetype,
) -> List[GatherSlot]:
    """Generate GatherSlots for a team's stats based on archetype."""
    slots: List[GatherSlot] = []
    critical = set(archetype.critical_team_keys)
    required = set(archetype.required_team_keys)

    # Critical inputs
    for key in archetype.critical_team_keys:
        slots.append(GatherSlot(
            key=f"{prefix}.{key}",
            data_type="team_stat",
            entity=entity,
            league=league,
            importance=InputImportance.CRITICAL,
            freshness_max=_freshness_for("team_stat"),
        ))

    # Important inputs (required but not critical)
    for key in archetype.required_team_keys:
        if key not in critical:
            slots.append(GatherSlot(
                key=f"{prefix}.{key}",
                data_type="team_stat",
                entity=entity,
                league=league,
                importance=InputImportance.IMPORTANT,
                freshness_max=_freshness_for("team_stat"),
            ))

    # Optional inputs
    for key in archetype.optional_team_keys:
        slots.append(GatherSlot(
            key=f"{prefix}.{key}",
            data_type="team_stat",
            entity=entity,
            league=league,
            importance=InputImportance.OPTIONAL,
            freshness_max=_freshness_for("team_stat"),
        ))

    return slots


def _player_stat_slots(
    entity: str,
    league: str,
    archetype: SportArchetype,
    prop_type: Optional[str] = None,
) -> List[GatherSlot]:
    """Generate GatherSlots for a player's stats."""
    slots: List[GatherSlot] = []

    # Required player keys
    for key in archetype.required_player_keys:
        if key == "name":
            continue  # already known from the entity
        slots.append(GatherSlot(
            key=f"player.{key}",
            data_type="player_stat",
            entity=entity,
            league=league,
            importance=InputImportance.IMPORTANT,
            freshness_max=_freshness_for("player_stat"),
        ))

    # Optional player keys
    for key in archetype.optional_player_keys:
        slots.append(GatherSlot(
            key=f"player.{key}",
            data_type="player_stat",
            entity=entity,
            league=league,
            importance=InputImportance.OPTIONAL,
            freshness_max=_freshness_for("player_stat"),
        ))

    # If there's a specific prop, the player's game log for that stat is critical
    if prop_type:
        slots.append(GatherSlot(
            key=f"player.game_log.{prop_type}",
            data_type="player_game_log",
            entity=entity,
            league=league,
            importance=InputImportance.CRITICAL,
            freshness_max=_freshness_for("player_game_log"),
        ))

    return slots


def _odds_slots(
    understanding: QueryUnderstanding,
) -> List[GatherSlot]:
    """Generate GatherSlots for odds data."""
    if not understanding.league:
        return []

    slots: List[GatherSlot] = []

    # Use the first two entities as the matchup, or a general "matchup" entity
    if len(understanding.entities) >= 2:
        matchup_label = f"{understanding.entities[0].name} vs {understanding.entities[1].name}"
    elif understanding.entities:
        matchup_label = understanding.entities[0].name
    else:
        matchup_label = "matchup"

    markets = understanding.markets or ["moneyline", "spread", "total"]
    for market in markets:
        slots.append(GatherSlot(
            key=f"odds.{market}",
            data_type="odds",
            entity=matchup_label,
            league=understanding.league,
            importance=InputImportance.IMPORTANT,
            freshness_max=_freshness_for("odds"),
        ))

    return slots


def _injury_slots(
    understanding: QueryUnderstanding,
) -> List[GatherSlot]:
    """Generate injury status slots for entities in the query."""
    if not understanding.league:
        return []

    slots: List[GatherSlot] = []
    for ent in understanding.entities:
        if ent.entity_type == "team":
            slots.append(GatherSlot(
                key=f"{ent.name}.injuries",
                data_type="injury",
                entity=ent.name,
                league=understanding.league,
                importance=InputImportance.OPTIONAL,
                freshness_max=_freshness_for("injury"),
            ))
    return slots


def _schedule_slot(understanding: QueryUnderstanding) -> List[GatherSlot]:
    """Generate schedule slot when we need to resolve 'tonight' / 'today'."""
    if not understanding.league:
        return []
    return [GatherSlot(
        key="schedule",
        data_type="schedule",
        entity=understanding.league,
        league=understanding.league,
        importance=InputImportance.IMPORTANT,
        freshness_max=_freshness_for("schedule"),
    )]


def build_gather_list(
    understanding: QueryUnderstanding,
    plan: AnswerPlan,
) -> List[GatherSlot]:
    """Build a typed gather-list from QueryUnderstanding + AnswerPlan.

    This is the main entry point for the requirement planner. It produces
    GatherSlots with proper importance levels and freshness rules based on
    the archetype and answer plan.
    """
    if plan.clarification_needed:
        return []  # nothing to gather yet

    slots: List[GatherSlot] = []
    league = understanding.league
    archetype = get_archetype(league) if league else None

    # -------------------------------------------------------------------
    # Simulation-related slots (native_sim or mixed mode)
    # -------------------------------------------------------------------
    needs_sim_data = any(
        m in (ExecutionMode.NATIVE_SIM, ExecutionMode.MIXED)
        for m in plan.execution_modes
    )

    if needs_sim_data and archetype and league:
        # Team stat slots for each team entity
        team_entities = [e for e in understanding.entities if e.entity_type == "team"]

        if len(team_entities) >= 2:
            slots.extend(_team_stat_slots(
                team_entities[0].name, "home_team", league, archetype,
            ))
            slots.extend(_team_stat_slots(
                team_entities[1].name, "away_team", league, archetype,
            ))
        elif len(team_entities) == 1:
            # Single team — we'll need to resolve the opponent from schedule
            slots.extend(_team_stat_slots(
                team_entities[0].name, "team", league, archetype,
            ))
            slots.extend(_schedule_slot(understanding))

        # Player prop slots
        if Subject.PLAYER_PROP in understanding.subjects:
            player_entities = [e for e in understanding.entities if e.entity_type == "player"]
            for player in player_entities:
                slots.extend(_player_stat_slots(
                    player.name, league, archetype, understanding.prop_type,
                ))

    # -------------------------------------------------------------------
    # Odds slots — needed for edge calculation
    # -------------------------------------------------------------------
    if plan.betting_recommendations_included or OutputPackage.BET_CARD in plan.output_packages:
        slots.extend(_odds_slots(understanding))
    elif needs_sim_data:
        # Even without formal betting recs, odds provide useful context
        slots.extend(_odds_slots(understanding))

    # -------------------------------------------------------------------
    # Injury slots — useful context for any game analysis
    # -------------------------------------------------------------------
    game_subjects = {Subject.GAME, Subject.PLAYER_PROP, Subject.SLATE, Subject.COMPARISON}
    if game_subjects & set(understanding.subjects):
        slots.extend(_injury_slots(understanding))

    # -------------------------------------------------------------------
    # Schedule slots — needed for slate analysis or when date is ambiguous
    # -------------------------------------------------------------------
    if Subject.SLATE in understanding.subjects and league:
        slots.extend(_schedule_slot(understanding))

    # -------------------------------------------------------------------
    # Research mode — looser requirements
    # -------------------------------------------------------------------
    if ExecutionMode.RESEARCH in plan.execution_modes and not needs_sim_data:
        # For research mode without simulation, we still want basic facts
        if league and understanding.entities:
            for ent in understanding.entities:
                if ent.entity_type == "team":
                    slots.append(GatherSlot(
                        key=f"research.{ent.name}.stats",
                        data_type="team_stat",
                        entity=ent.name,
                        league=league,
                        importance=InputImportance.OPTIONAL,
                        freshness_max=_freshness_for("team_stat"),
                    ))
            slots.extend(_injury_slots(understanding))
            slots.extend(_schedule_slot(understanding))

    # -------------------------------------------------------------------
    # Bankroll calc — may need slate-level odds
    # -------------------------------------------------------------------
    if ExecutionMode.BANKROLL_CALC in plan.execution_modes:
        if Subject.SLATE in understanding.subjects and league:
            slots.extend(_schedule_slot(understanding))

    # Deduplicate slots by key
    seen_keys: set[str] = set()
    deduped: List[GatherSlot] = []
    for slot in slots:
        if slot.key not in seen_keys:
            seen_keys.add(slot.key)
            deduped.append(slot)

    logger.info("Requirement planner produced %d gather slots for league=%s modes=%s",
                len(deduped), league, [m.value for m in plan.execution_modes])

    return deduped
