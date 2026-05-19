"""
Frozen artifacts — typed, versioned backtest inputs derived from traces.

A FrozenArtifact captures everything the backtest engine needs to replay
a historical game: team contexts, odds, seed, and outcome. Every field
is decision-time data; post-outcome information is attached only at
grading time via the ``outcome`` field.

Artifacts are derived from persisted ExecutionTraces via ``trace_to_artifact()``.
Legacy HistoricalGame dicts can be converted via ``compat_dict_to_artifact()``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field


class FrozenArtifact(BaseModel):
    """A historically valid, typed input for the backtest engine.

    Derived from a persisted ExecutionTrace + attached outcome.
    Every field is decision-time data; no post-outcome contamination.
    """

    # Identity
    artifact_id: str = Field(description="Deterministic hash of event identity")
    schema_version: int = 1
    source_trace_id: str | None = Field(
        default=None, description="Links back to the originating ExecutionTrace"
    )

    # Event
    home_team: str
    away_team: str
    league: str
    date: str = Field(description="YYYY-MM-DD")

    # Contexts (as used by the sim at decision time)
    home_context: dict[str, Any] = Field(default_factory=dict)
    away_context: dict[str, Any] = Field(default_factory=dict)

    # Odds (decision-time snapshot)
    odds: dict[str, Any] = Field(
        default_factory=dict,
        description="Decision-time odds: moneyline_home, moneyline_away, spread_home, over_under",
    )

    # Deterministic seed (as used by the orchestrator)
    simulation_seed: int = 42

    # Calibration policy reference
    calibration_policy: str = "static_v1"

    # Outcome (attached only at grading time, NOT during simulation)
    outcome: dict[str, Any] | None = Field(
        default=None, description="home_score, away_score — attached at grading time"
    )
    closing_odds: dict[str, Any] | None = Field(
        default=None, description="Closing-line odds for CLV calculation"
    )


def compute_artifact_id(
    home_team: str, away_team: str, league: str, date: str
) -> str:
    """Derive a deterministic artifact ID from event identity.

    Same event always produces the same ID, preventing duplicate artifacts
    from multiple trace replays of the same game.
    """
    raw = f"{home_team}|{away_team}|{league}|{date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def trace_to_artifact(
    trace: dict[str, Any],
    outcome: dict[str, Any] | None = None,
) -> FrozenArtifact:
    """Convert a persisted trace dict into a frozen backtest artifact.

    Extracts decision-time data from the trace. Attaches outcome separately.
    The trace dict is expected to match the shape of ExecutionTrace serialized
    via ``model_dump()`` or as stored in the ``full_trace`` column of TraceStore.

    Args:
        trace: Serialized ExecutionTrace dict.
        outcome: Optional outcome dict with home_score, away_score.

    Returns:
        A FrozenArtifact ready for backtest consumption.
    """
    # Extract event identity from trace
    matchup = trace.get("matchup", "")
    league = trace.get("league", "")
    timestamp = trace.get("timestamp", "")

    # Parse matchup "Away @ Home" → team names
    home_team = ""
    away_team = ""
    if matchup and " @ " in matchup:
        parts = matchup.split(" @ ", 1)
        away_team = parts[0].strip()
        home_team = parts[1].strip()

    # Extract date from timestamp
    date = timestamp[:10] if timestamp else ""

    # Extract contexts from execution_result
    exec_result = trace.get("execution_result") or {}
    home_context = exec_result.get("home_context", {})
    away_context = exec_result.get("away_context", {})

    # If contexts not in execution_result, try gathered_facts
    if not home_context and not away_context:
        home_context, away_context = _extract_contexts_from_facts(
            trace.get("gathered_facts", []), home_team, away_team
        )

    # Extract odds from trace snapshot
    odds = trace.get("odds_snapshot") or {}

    # Seed from trace
    seed = trace.get("simulation_seed", 42)

    # Trace ID
    trace_id = trace.get("trace_id")
    if trace_id is not None:
        trace_id = str(trace_id)

    artifact_id = compute_artifact_id(home_team, away_team, league, date)

    return FrozenArtifact(
        artifact_id=artifact_id,
        source_trace_id=trace_id,
        home_team=home_team,
        away_team=away_team,
        league=league,
        date=date,
        home_context=home_context,
        away_context=away_context,
        odds=odds,
        simulation_seed=seed if seed is not None else 42,
        outcome=outcome,
    )


def compat_dict_to_artifact(game: dict[str, Any]) -> FrozenArtifact:
    """Convert a legacy HistoricalGame dict to FrozenArtifact.

    This shim allows existing backtest code and tests that use
    hand-constructed game dicts to continue working.
    """
    home_team = game.get("home_team", "Home")
    away_team = game.get("away_team", "Away")
    league = game.get("league", "NBA")
    date = game.get("date", "")

    artifact_id = compute_artifact_id(home_team, away_team, league, date)

    return FrozenArtifact(
        artifact_id=artifact_id,
        home_team=home_team,
        away_team=away_team,
        league=league,
        date=date,
        home_context=game.get("home_context", {}),
        away_context=game.get("away_context", {}),
        odds=game.get("odds", {}),
        simulation_seed=game.get("simulation_seed", 42),
        outcome=game.get("outcome"),
        closing_odds=game.get("closing_odds"),
    )


def _extract_contexts_from_facts(
    facts: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Best-effort context extraction from gathered facts.

    Scans gathered_facts for team_stat entries matching home/away teams.
    Returns (home_context, away_context).
    """
    home_ctx: dict[str, Any] = {}
    away_ctx: dict[str, Any] = {}

    for fact in facts:
        slot = fact.get("slot", {})
        result = fact.get("result")
        if not result or slot.get("data_type") != "team_stat":
            continue

        data = result.get("data", {})
        entity = slot.get("entity", "").lower()

        if entity and home_team.lower() in entity:
            home_ctx.update(data)
        elif entity and away_team.lower() in entity:
            away_ctx.update(data)

    return home_ctx, away_ctx
