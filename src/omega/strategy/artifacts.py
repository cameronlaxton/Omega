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
import math
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
    game_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Decision-time situational context used for calibration slice selection",
    )

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


def compute_artifact_id(home_team: str, away_team: str, league: str, date: str) -> str:
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

    input_snapshot = trace.get("input_snapshot") or {}

    # Extract contexts from execution_result or canonical analyze() input_snapshot
    exec_result = trace.get("execution_result") or {}
    home_context = exec_result.get("home_context") or input_snapshot.get("home_context") or {}
    away_context = exec_result.get("away_context") or input_snapshot.get("away_context") or {}
    game_context = input_snapshot.get("game_context") or trace.get("context_labels") or {}

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
        game_context=game_context,
        odds=odds,
        simulation_seed=seed if seed is not None else 42,
        outcome=outcome,
    )


class FrozenPropArtifact(BaseModel):
    """A historically valid, typed prop-plane input for the variant sweep.

    The prop analogue of :class:`FrozenArtifact`: everything needed to
    re-simulate one graded prop decision at an alternative structural knob
    (``nb_k_scale``) is decision-time data recovered from the persisted trace —
    the projection mean and base NB dispersion ``k`` come from the trace's own
    ``distribution_params`` (mu, k), the line/player/stat from the input
    snapshot. Outcomes (the attached ``prop_outcomes`` rows) are one-to-many and
    carry no calibration signal until grading pairs them with a prediction.

    ``nb_dispersion_k`` is the PRE-scale base ``k``. The persisted
    ``distribution_params.k`` is the FINAL ``k`` the production sim ran with;
    when a promoted prop profile applied an ``nb_k_scale`` the backend echoes
    the applied scale into ``distribution_params``, and the builder divides it
    back out — so re-simulating at a candidate scale never double-applies the
    production one. Traces with no echoed scale ran at the 1.0 identity
    (ungoverned, or persisted before the prop consumption path existed).
    """

    # Identity
    artifact_id: str = Field(description="Deterministic hash of prop decision identity")
    schema_version: int = 1
    source_trace_id: str | None = Field(
        default=None, description="Links back to the originating ExecutionTrace"
    )

    # Event
    player_name: str
    league: str
    stat_type: str = Field(description="Canonical prop stat key (e.g. rushing_yards)")
    line: float
    date: str = Field(description="YYYY-MM-DD decision date")

    # Simulation inputs (decision-time, recovered from the persisted sim result)
    projection_mean: float = Field(description="NB mu the production sim ran with")
    nb_dispersion_k: float = Field(
        description=(
            "Base (pre-scale) NB dispersion k: the persisted final k with any "
            "echoed production nb_k_scale divided back out"
        )
    )
    projection_std: float | None = None
    n_iter: int = 2000
    simulation_seed: int | None = None

    # Outcomes (attached only at grading time; one-to-many rows of side/result)
    prop_outcomes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Attached prop_outcomes rows (side, result, ...); push/void excluded at grading",
    )


def compute_prop_artifact_id(
    player_name: str, league: str, stat_type: str, line: float, date: str
) -> str:
    """Derive a deterministic artifact ID from prop decision identity."""
    raw = f"{player_name}|{league}|{stat_type}|{line}|{date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _finite(value: Any) -> float | None:
    """Parse a finite float or None (never raises)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def prop_trace_to_frozen_artifact(trace: dict[str, Any]) -> FrozenPropArtifact | None:
    """Convert one graded prop trace dict into a :class:`FrozenPropArtifact`.

    The single builder (mirrors :func:`trace_to_artifact` — do not rebuild this
    shape elsewhere). Fail-closed: returns None when the trace cannot support a
    faithful re-simulation — no NB ``distribution_params`` (mu, k), no line, no
    date, or no gradeable (non-push/void over/under) outcome row.

    The artifact date prefers the match ``decision_time`` over the trace
    ``timestamp``: a historical-replay batch stamps every trace with the single
    RUN day, which would defeat the sweep's no-leak validation/holdout split.
    """
    from omega.core.simulation.backends import canonical_prop_stat_type

    snap = trace.get("input_snapshot") or {}
    player_name = snap.get("player_name") or trace.get("player_name") or ""
    stat_type = snap.get("prop_type") or snap.get("stat_type") or ""
    league = trace.get("league") or snap.get("league") or ""
    line = _finite(snap.get("line"))
    if not player_name or not stat_type or not league or line is None:
        return None
    # Canonicalize market-key aliases (pass_yds -> passing_yards) so loader
    # filtering and bucket naming see one stat key per market family.
    stat_type = canonical_prop_stat_type(league, stat_type)

    date = str(trace.get("decision_time") or "")[:10] or str(trace.get("timestamp") or "")[:10]
    if not date:
        return None

    # The persisted sim result is the source of truth for the param point the
    # production run actually used (post context-adjustment mean, injected or
    # moment-matched k) — re-deriving it from the raw request would duplicate
    # analyze_player_prop's prior-building logic.
    params: dict[str, Any] = {}
    n_iter = None
    seed = trace.get("simulation_seed")
    projection_std = None
    # full_trace carries the rows inline; store queries also attach the V10 table
    # rows as _simulation_distributions (same shape, params JSON-decoded).
    dist_rows = trace.get("simulation_distributions") or trace.get("_simulation_distributions")
    for row in dist_rows or []:
        if not isinstance(row, dict):
            continue
        row_params = row.get("distribution_params")
        if isinstance(row_params, dict) and row_params.get("mu") is not None:
            params = row_params
            n_iter = row.get("n_iterations")
            projection_std = _finite(row.get("sample_std"))
            if seed is None:
                seed = row.get("seed")
            break
    mu = _finite(params.get("mu"))
    k = _finite(params.get("k"))
    if mu is None or mu < 0 or k is None or k <= 0:
        return None
    # distribution_params.k is the FINAL k the sim ran with. When a promoted prop
    # profile applied an nb_k_scale, the backend echoed the applied scale next to
    # it — divide it back out so nb_dispersion_k is always the PRE-scale base the
    # sweep layers candidate scales onto (never double-applying production's).
    applied_scale = _finite(params.get("nb_k_scale"))
    if applied_scale is not None and applied_scale > 0:
        k /= applied_scale

    outcomes = [dict(r) for r in trace.get("_prop_outcomes") or [] if isinstance(r, dict)]
    gradeable = any(
        (r.get("side") or "").lower() in ("over", "under")
        and r.get("result") not in ("push", "void")
        for r in outcomes
    )
    if not gradeable:
        return None

    trace_id = trace.get("trace_id")
    return FrozenPropArtifact(
        artifact_id=compute_prop_artifact_id(player_name, league, stat_type, line, date),
        source_trace_id=str(trace_id) if trace_id is not None else None,
        player_name=player_name,
        league=league,
        stat_type=stat_type,
        line=line,
        date=date,
        projection_mean=mu,
        nb_dispersion_k=k,
        projection_std=projection_std,
        n_iter=int(n_iter) if n_iter else 2000,
        simulation_seed=int(seed) if seed is not None else None,
        prop_outcomes=outcomes,
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
        game_context=game.get("game_context", {}),
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
