"""
Stable service interface wrapping Omega core internals.

JSON-in/JSON-out service layer. Caller supplies all context;
no data fetching, no config loading, no network calls.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from omega.core.betting.kelly import recommend_stake
from omega.core.betting.odds import (
    edge_percentage,
    expected_value_percent,
    implied_probability,
)
from omega.core.calibration.adjustment_policy import (
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.calibration.market import calibration_market_for_plane
from omega.core.calibration.probability import apply_calibration, apply_calibration_audited
from omega.core.config.leagues import get_league_config
from omega.core.contracts.market_quotes import market_quote
from omega.core.contracts.schemas import (
    AnalysisMetadata,
    BetSlip,
    CalibrationAudit,
    EdgeDetail,
    GameAnalysisRequest,
    GameAnalysisResponse,
    OddsInput,
    PlayerPropRequest,
    PlayerPropResponse,
    SimulationResult,
    SlateAnalysisRequest,
    SlateAnalysisResponse,
)
from omega.core.contracts.seeding import stable_analysis_hash
from omega.core.edge.consumers import resolve_edge_consumer
from omega.core.simulation.archetypes import get_archetype_name
from omega.core.simulation.backends import (
    GameSimulationBackend,
    PropSimulationInput,
    resolve_default_prop_backend,
    resolve_game_backend,
    resolve_prop_backend,
)
from omega.core.simulation.engine import (
    OmegaSimulationEngine,
    select_distribution,
)
from omega.core.simulation.evidence_handlers import (
    AdjustmentRecord,
    PlaneAdjustment,
    compute_game_adjustment,
    compute_player_adjustment,
    resolve_evidence_mode,
)
from omega.core.simulation.evidence_to_modifier import (
    compute_transition_modifier_adjustment,
)
from omega.core.simulation.validation import validate_sim_context
from omega.trace.eligibility import (
    calibration_exclusion_reasons as compute_calibration_exclusion_reasons,
)

UTC = timezone.utc

logger = logging.getLogger("omega.service")

_engine = OmegaSimulationEngine()
MODEL_VERSION = "omega-core-phase6h"


@dataclass(frozen=True)
class EvidenceExecutionPlan:
    """Effective evidence path for one analysis.

    The plan keeps simulation behavior and trace provenance together. Handler
    adjustments, Markov modifiers, and per-signal application records are derived
    once so canonical traces cannot claim a different path than the engine used.
    """

    adjustment: PlaneAdjustment | None = None
    transition_modifiers: dict[str, float] | None = None
    evidence_mode: str = "shadow"
    evidence_application: list[dict[str, Any]] = field(default_factory=list)
    # Per-modifier-key (Markov) or per-family (plane) aggregate math, kept apart
    # from the per-signal applications so grouped/clamped effects are not
    # misattributed to one signal (Issue #22).
    aggregation_records: list[dict[str, Any]] = field(default_factory=list)


def _stable_input_hash(request: Any) -> str | None:
    """Stable 8-char content hash, excluding volatile odds and close snapshots."""
    try:
        return stable_analysis_hash(request)
    except (TypeError, ValueError) as exc:
        logger.debug("Failed to derive stable input hash: %s", exc)
        return None


def _new_trace_id(request: Any = None) -> str:
    """Python-minted trace id for VM/MCP runs.

    The stable hash prefix identifies the simulation context while the nonce
    keeps repeated runs separately persistable. Volatile odds structures are
    intentionally excluded from the hash to avoid cache fracturing.
    """
    if request is not None:
        h = _stable_input_hash(request)
        if h is not None:
            return f"sandbox-{h}-{uuid.uuid4().hex[:4]}"
    return f"sandbox-{uuid.uuid4().hex[:12]}"


def _coerce_request(
    request: dict[str, Any] | GameAnalysisRequest | PlayerPropRequest | SlateAnalysisRequest,
) -> GameAnalysisRequest | PlayerPropRequest | SlateAnalysisRequest:
    """Accept a typed request or a dict and return the canonical typed request."""
    if isinstance(request, (GameAnalysisRequest, PlayerPropRequest, SlateAnalysisRequest)):
        return request
    if not isinstance(request, dict):
        raise TypeError(f"Expected dict or request object, got {type(request).__name__}")

    if "player_name" in request and "prop_type" in request:
        return PlayerPropRequest(**request)
    if "games" in request or (
        "league" in request and "home_team" not in request and "player_name" not in request
    ):
        return SlateAnalysisRequest(**request)
    return GameAnalysisRequest(**request)


def _safe_dump(obj: Any) -> dict[str, Any]:
    """Convert Pydantic models and dicts to JSON-friendly dictionaries."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def _validation_skip_response(
    request: GameAnalysisRequest,
    errors: list[str],
    ran_at: str,
) -> GameAnalysisResponse:
    matchup = f"{request.away_team} @ {request.home_team}"
    return GameAnalysisResponse(
        matchup=matchup,
        league=request.league,
        analyzed_at=ran_at,
        status="skipped",
        skip_reason="Invalid simulation context: " + "; ".join(errors),
        missing_requirements=errors,
        context_source="missing",
    )


def _strict_game_context_errors(request: GameAnalysisRequest) -> list[str]:
    """Validate provided team contexts once at the service boundary.

    Missing contexts are handled by the engine's existing skip/baseline logic.
    Provided contexts are strict-validated here so invalid proxies (for example
    raw basketball FG% passed as off_rating) cannot reach backend dispatch.
    """
    errors: list[str] = []
    for side, context in (("home", request.home_context), ("away", request.away_context)):
        if context is None:
            continue
        try:
            validate_sim_context(context, request.league, side, strict=True)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def _result_downgrades(result: Any) -> list[str]:
    """Expose minimal trace-level downgrades without reviving the lite gate."""
    status = getattr(result, "status", None)
    if status == "skipped":
        return ["engine_skipped"]
    if status == "error":
        return ["engine_error"]
    return []


def _result_status(result: Any) -> str | None:
    return getattr(result, "status", None)


def _result_context_source(result: Any) -> str | None:
    return getattr(result, "context_source", None)


def _result_baseline_used(result: Any) -> bool:
    return bool(getattr(result, "baseline_used", False))


def _identity_status(kind: str, request: Any) -> str:
    if kind == "prop":
        missing = [
            field
            for field in ("player_name", "home_team", "away_team", "game_date", "line")
            if not getattr(request, field, None)
        ]
        return "missing" if missing else "complete"
    if kind == "game":
        has_teams = getattr(request, "home_team", None) and getattr(request, "away_team", None)
        return "complete" if has_teams else "missing"
    return "complete"


def analyze(
    request: dict[str, Any] | GameAnalysisRequest | PlayerPropRequest | SlateAnalysisRequest,
    *,
    session_id: str,
    bankroll: float,
    trace_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a canonical Omega analysis and return an auditable trace envelope.

    This wrapper owns only request typing, trace identity, and audit snapshots.
    Simulation, calibration, edge, EV, Kelly, staking, and confidence tiers stay
    in the deterministic analyzers below.
    """
    if not session_id:
        raise ValueError("session_id is required for canonical analyze()")
    if bankroll is None or bankroll <= 0:
        raise ValueError("bankroll must be a positive explicit value")

    typed_req = _coerce_request(request)
    trace_id = _new_trace_id(typed_req)
    ran_at = datetime.now(UTC).isoformat()

    validation_errors: list[str] = []
    if isinstance(typed_req, GameAnalysisRequest):
        validation_errors = _strict_game_context_errors(typed_req)

    # Structured evidence is planned once here, then passed into the analyzer.
    # The same plan supplies both the applied effect and the per-signal
    # applications recorded on the trace, so there is exactly one evaluation per
    # analysis.
    evidence_plan: EvidenceExecutionPlan | None = None
    if isinstance(typed_req, PlayerPropRequest):
        evidence_plan = _player_evidence_plan_for(typed_req)
    elif isinstance(typed_req, GameAnalysisRequest) and not validation_errors:
        evidence_plan = _game_evidence_plan_for(typed_req)

    result: GameAnalysisResponse | PlayerPropResponse | SlateAnalysisResponse
    if isinstance(typed_req, GameAnalysisRequest) and validation_errors:
        result = _validation_skip_response(typed_req, validation_errors, ran_at)
        kind = "game"
    elif isinstance(typed_req, GameAnalysisRequest):
        result = analyze_game(typed_req, bankroll=bankroll, evidence_plan=evidence_plan)
        kind = "game"
    elif isinstance(typed_req, PlayerPropRequest):
        result = analyze_player_prop(
            typed_req,
            bankroll=bankroll,
            evidence_adjustment=evidence_plan.adjustment if evidence_plan else None,
        )
        kind = "prop"
    elif isinstance(typed_req, SlateAnalysisRequest):
        typed_req = typed_req.model_copy(update={"bankroll": bankroll})
        result = analyze_slate(typed_req)
        kind = "slate"
    else:
        raise TypeError(
            f"Unsupported request type {type(typed_req).__name__}. "
            "Expected GameAnalysisRequest, PlayerPropRequest, or SlateAnalysisRequest."
        )

    # Populate context_labels for calibration slice fitting.
    # Covers both GameAnalysisRequest and PlayerPropRequest via _build_context_labels().
    context_labels: dict[str, Any] = {}
    if isinstance(typed_req, (GameAnalysisRequest, PlayerPropRequest)):
        context_labels = _build_context_labels(typed_req)

    # Evidence application — aligned by index with input_snapshot.evidence so the
    # trace store can explode it into the evidence_signals table (V9).
    evidence_mode = "shadow"
    evidence_application: list[dict[str, Any]] = []
    evidence_aggregation: list[dict[str, Any]] = []
    if evidence_plan is not None:
        evidence_mode = evidence_plan.evidence_mode
        evidence_application = evidence_plan.evidence_application
        evidence_aggregation = evidence_plan.aggregation_records

    # Assemble trace_quality last — pure audit metadata, never influences math above.
    result_downgrades = _result_downgrades(result)
    result_missing_requirements = [
        str(req) for req in (getattr(result, "missing_requirements", None) or [])
    ]
    evidence_signals = getattr(typed_req, "evidence", None) or []
    evidence_status = "present" if evidence_signals else "empty"
    context_source = _result_context_source(result)
    baseline_used = _result_baseline_used(result)
    identity_status = _identity_status(kind, typed_req)
    # "missing" only fires when context was provided: the caller had data to
    # reason about but submitted no structured signals, so the evidence-learning
    # loop has a gap. "not_applicable" covers baseline/engine-skipped runs where
    # no signals are expected. This never gates calibration eligibility — that
    # path is deliberately evidence-agnostic (see omega.trace.eligibility).
    evidence_quality = (
        "present" if evidence_signals
        else "missing" if context_source == "provided"
        else "not_applicable"
    )
    caller_exclusion_reasons = (
        [str(reason) for reason in trace_quality.get("calibration_exclusion_reasons", [])]
        if trace_quality
        else []
    )
    # Single source of truth: omega.trace.eligibility owns the exclusion-reason
    # predicate. No QA verdict exists at analyze time (no sidecar yet); ingest
    # reconciles a failed trace-scoped verdict into this same flag later.
    calibration_exclusion_reasons = compute_calibration_exclusion_reasons(
        result_status=_result_status(result),
        context_source=context_source,
        baseline_used=baseline_used,
        identity_status=identity_status,
        result_downgrades=result_downgrades,
        result_missing_requirements=result_missing_requirements,
        caller_exclusion_reasons=caller_exclusion_reasons,
    )
    calibration_eligible = not calibration_exclusion_reasons

    engine_trace_quality: dict[str, Any] = {
        "aggregate_quality": None,
        "downgrades": result_downgrades,
        "passed": len(result_downgrades) == 0,
        "evidence_status": evidence_status,
        "evidence_quality": evidence_quality,
        "identity_status": identity_status,
        "context_source": context_source,
        "baseline_used": baseline_used,
        "calibration_eligible": calibration_eligible,
        "calibration_exclusion_reasons": calibration_exclusion_reasons,
    }
    if trace_quality:
        merged_downgrades = sorted(
            {*trace_quality.get("downgrades", []), *result_downgrades}
        )
        engine_trace_quality = {
            **engine_trace_quality,
            **trace_quality,
            "downgrades": merged_downgrades,
            "evidence_status": evidence_status,   # engine-computed; caller cannot override
            "evidence_quality": evidence_quality,  # engine-computed; caller cannot override
            "identity_status": identity_status,
            "context_source": context_source,
            "baseline_used": baseline_used,
            "calibration_eligible": calibration_eligible,
            "calibration_exclusion_reasons": calibration_exclusion_reasons,
        }

    return {
        "trace_id": trace_id,
        "model_version": MODEL_VERSION,
        "ran_at": ran_at,
        "kind": kind,
        "session_id": session_id,
        "bankroll": bankroll,
        "input_snapshot": _safe_dump(typed_req),
        "result": _safe_dump(result),
        "downgrades": result_downgrades,
        "context_labels": context_labels,
        "evidence_mode": evidence_mode,
        "evidence_application": evidence_application,
        "evidence_aggregation": evidence_aggregation,
        "trace_quality": engine_trace_quality,
    }


def _calibrate(
    raw_prob: float,
    league: str | None = None,
    context_hints: dict[str, Any] | None = None,
) -> float:
    """Apply calibration. Delegates to apply_calibration() -- the single source of truth."""
    return apply_calibration(raw_prob, league=league, context_hints=context_hints)


def _calibrate_audited(
    raw_prob: float,
    league: str | None = None,
    context_hints: dict[str, Any] | None = None,
    plane: str = "game",
    market: str = "home",
) -> tuple[float, CalibrationAudit]:
    """Like _calibrate() but also returns a CalibrationAudit recording the path taken."""
    # Derive the calibration profile market from the plane in one shared policy.
    cal_market = calibration_market_for_plane(plane, market=market)
    calibrated, d = apply_calibration_audited(
        raw_prob, league=league, context_hints=context_hints, market=cal_market
    )
    audit = CalibrationAudit(
        raw_prob=d["raw_prob"],
        calibrated_prob=d["calibrated_prob"],
        league=league,
        plane=plane,
        market=market,
        method_resolved=d.get("method_resolved"),
        profile_id=d.get("profile_id"),
        context_slice=d.get("context_slice"),
        resolved_slice=d.get("resolved_slice"),
        path=d["path"],
    )
    return calibrated, audit


def _build_context_labels(
    request: GameAnalysisRequest | PlayerPropRequest,
) -> dict[str, Any]:
    """Extract calibration context labels from any request type.

    Passes through all game_context keys so the fitter can use any signal
    the agent supplies, not just the known set at coding time.
    """
    gc: dict[str, Any] | None = getattr(request, "game_context", None)
    if not gc:
        return {}
    labels: dict[str, Any] = {}
    # Type-coerce the known calibration slice keys for safety
    if gc.get("is_playoff") is not None:
        labels["is_playoff"] = bool(gc["is_playoff"])
    if gc.get("rest_days") is not None:
        labels["rest_days"] = int(gc["rest_days"])
    if gc.get("blowout_risk") is not None:
        labels["blowout_risk"] = float(gc["blowout_risk"])
    # Pass through any additional context keys (matchup weaknesses, scheme signals, etc.)
    for k, v in gc.items():
        if k not in labels:
            labels[k] = v
    return labels


# ---------------------------------------------------------------------------
# Structured evidence adjustment (Phase 6i)
# ---------------------------------------------------------------------------


def _load_adjustment_policy() -> AdjustmentPolicy | None:
    """Return the production engine adjustment policy, or None if unavailable.

    Never raises: a missing/corrupt registry degrades to no evidence adjustment.
    """
    try:
        return AdjustmentPolicyRegistry().get_production_policy()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not load adjustment policy: %s", exc)
        return None


def _evidence_identity(signal: Any) -> tuple[tuple[Any, ...] | None, tuple[Any, ...]]:
    """Return full and fallback identities for cross-plane deduplication."""
    signal_type = getattr(signal, "signal_type", None)
    direction = getattr(signal, "direction", None) or "neutral"
    source = getattr(signal, "source", None)
    window = getattr(signal, "window", None)
    stat_key = getattr(signal, "stat_key", None)
    fallback = (signal_type, direction)
    full = None
    if signal_type and source and window and stat_key:
        full = (signal_type, source, window, direction, stat_key)
    return full, fallback


def _suppressed_player_signal_indices(evidence: list[Any]) -> set[int]:
    """Identify player-plane duplicates suppressed by game-plane evidence.

    Game-plane execution has precedence. Suppressed player signals remain in the
    request snapshot for audit, but they cannot affect prediction math.
    """
    game_full: set[tuple[Any, ...]] = set()
    game_fallback: set[tuple[Any, ...]] = set()
    for sig in evidence:
        if getattr(sig, "plane", None) != "game":
            continue
        full, fallback = _evidence_identity(sig)
        if full is not None:
            game_full.add(full)
        game_fallback.add(fallback)

    suppressed: set[int] = set()
    for idx, sig in enumerate(evidence):
        if getattr(sig, "plane", None) != "player":
            continue
        full, fallback = _evidence_identity(sig)
        if (full is not None and full in game_full) or fallback in game_fallback:
            suppressed.add(idx)
    return suppressed


def _signal_with_suppressed_player_plane(signal: Any) -> Any:
    """Make a suppressed player signal skip player handlers without losing order."""
    if hasattr(signal, "model_copy"):
        return signal.model_copy(update={"plane": "game"})
    return signal


def _suppression_record(signal: Any, evidence_mode: str) -> AdjustmentRecord:
    return AdjustmentRecord(
        signal_type=str(getattr(signal, "signal_type", "")),
        target="skip",
        factor=1.0,
        applied=False,
        reason="suppressed_by_game_plane_dedup",
        policy_version="evidence_dedup_v1",
        evidence_mode=evidence_mode,
    )


def _with_suppression_records(
    adjustment: PlaneAdjustment,
    evidence: list[Any],
    suppressed_indices: set[int],
) -> PlaneAdjustment:
    if not suppressed_indices:
        return adjustment
    records = list(adjustment.records)
    for idx in suppressed_indices:
        if idx < len(records):
            records[idx] = _suppression_record(evidence[idx], adjustment.evidence_mode)
    return PlaneAdjustment(
        mean_factor=adjustment.mean_factor,
        std_factor=adjustment.std_factor,
        home_factor=adjustment.home_factor,
        away_factor=adjustment.away_factor,
        records=records,
        evidence_mode=adjustment.evidence_mode,
    )


def _player_evidence_plan_for(request: PlayerPropRequest) -> EvidenceExecutionPlan:
    evidence = list(getattr(request, "evidence", None) or [])
    if not evidence:
        return EvidenceExecutionPlan()
    policy = _load_adjustment_policy()
    if policy is None:
        return EvidenceExecutionPlan()

    evidence_mode = resolve_evidence_mode(policy)
    suppressed = _suppressed_player_signal_indices(evidence)
    handler_evidence = [
        _signal_with_suppressed_player_plane(sig) if idx in suppressed else sig
        for idx, sig in enumerate(evidence)
    ]
    adjustment = compute_player_adjustment(
        player_context=request.player_context or {},
        evidence=handler_evidence,
        league=request.league,
        prop_type=request.prop_type,
        policy=policy,
        evidence_mode=evidence_mode,
    )
    adjustment = _with_suppression_records(adjustment, evidence, suppressed)
    return EvidenceExecutionPlan(
        adjustment=adjustment,
        evidence_mode=adjustment.evidence_mode,
        evidence_application=adjustment.applications(),
    )


def _merge_markov_applications(
    evidence: list[Any],
    suppressed_indices: set[int],
    active_applications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Realign Markov per-signal applications onto the full evidence list.

    Suppressed signals get a suppression record; every other signal pulls its
    real application (mapped or skip) from
    ``compute_transition_modifier_adjustment``, which already ran over the active
    (non-suppressed) evidence in order — so the iterator stays aligned.
    """
    out: list[dict[str, Any]] = []
    active = iter(active_applications)
    for idx, sig in enumerate(evidence):
        if idx in suppressed_indices:
            out.append(_suppression_record(sig, "markov_transition").as_application())
        else:
            out.append(next(active))
    return out


def _effective_game_backend_name(request: GameAnalysisRequest) -> str:
    """Resolve the backend a request will actually run on.

    When the caller leaves ``simulation_backend`` at its ``"fast_score"`` default,
    the league's ``default_game_backend`` (e.g. ``markov_state_wnba`` for WNBA)
    takes over. An explicitly-chosen backend is always honored. This is the one
    place the league default is consulted, so dispatch and evidence-plan
    construction never disagree about which backend is in play.
    """
    name = request.simulation_backend
    if name == "fast_score":
        league_default = get_league_config(request.league).get("default_game_backend")
        if league_default:
            return str(league_default)
    return name


def _uses_transition_modifiers(backend: GameSimulationBackend | None) -> bool:
    """True when *backend* consumes Markov transition modifiers for evidence.

    Reads the backend's ``evidence_mode`` capability rather than sniffing its
    name, so a new Markov-family backend need not be named ``markov_state*`` to
    route evidence correctly. Defaults to plane-adjustment routing for an unknown
    backend (``None``) or any backend predating the attribute.
    """
    return getattr(backend, "evidence_mode", "plane_adjustment") == "markov_transition"


def _game_evidence_plan_for(request: GameAnalysisRequest) -> EvidenceExecutionPlan:
    evidence = list(getattr(request, "evidence", None) or [])
    if not evidence:
        return EvidenceExecutionPlan()

    suppressed = _suppressed_player_signal_indices(evidence)
    backend = resolve_game_backend(_effective_game_backend_name(request))
    if _uses_transition_modifiers(backend):
        active_evidence = [
            sig for idx, sig in enumerate(evidence) if idx not in suppressed
        ]
        markov = compute_transition_modifier_adjustment(
            active_evidence,
            home_team=request.home_team,
            policy=_load_adjustment_policy(),
        )
        applications = _merge_markov_applications(
            evidence, suppressed, markov.applications
        )
        return EvidenceExecutionPlan(
            adjustment=None,
            transition_modifiers=markov.modifiers or None,
            evidence_mode="markov_transition",
            evidence_application=applications,
            aggregation_records=markov.aggregation_records,
        )

    policy = _load_adjustment_policy()
    if policy is None:
        return EvidenceExecutionPlan()
    evidence_mode = resolve_evidence_mode(policy)
    adjustment = compute_game_adjustment(
        evidence=evidence,
        league=request.league,
        policy=policy,
        evidence_mode=evidence_mode,
    )
    adjustment = _with_suppression_records(adjustment, evidence, suppressed)
    return EvidenceExecutionPlan(
        adjustment=adjustment,
        evidence_mode=adjustment.evidence_mode,
        evidence_application=adjustment.applications(),
    )


def _player_adjustment_for(request: PlayerPropRequest) -> PlaneAdjustment | None:
    """Compute the structured-evidence adjustment for a player-prop request.

    Pure given (request, production policy). Returns None when the request
    carries no evidence or no policy is available. Canonical ``analyze()`` uses
    the richer evidence plan so suppression records and behavior stay aligned.
    """
    return _player_evidence_plan_for(request).adjustment


def _game_adjustment_for(request: GameAnalysisRequest) -> PlaneAdjustment | None:
    """Compute handler evidence for direct fast-score callers."""
    return _game_evidence_plan_for(request).adjustment


def _apply_game_evidence(
    home_context: dict[str, Any] | None,
    away_context: dict[str, Any] | None,
    adjustment: PlaneAdjustment | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return team contexts with game-plane evidence factors applied.

    Scales each team's ``off_rating`` by the per-team factor. In shadow mode the
    factors are 1.0, so the contexts are returned unchanged. Never mutates the
    input dicts.
    """
    if adjustment is None:
        return home_context, away_context
    home = dict(home_context) if isinstance(home_context, dict) else home_context
    away = dict(away_context) if isinstance(away_context, dict) else away_context
    if (
        isinstance(home, dict)
        and adjustment.home_factor != 1.0
        and home.get("off_rating") is not None
    ):
        home["off_rating"] = float(home["off_rating"]) * adjustment.home_factor
        home["_evidence_factor_applied"] = round(adjustment.home_factor, 4)
    if (
        isinstance(away, dict)
        and adjustment.away_factor != 1.0
        and away.get("off_rating") is not None
    ):
        away["off_rating"] = float(away["off_rating"]) * adjustment.away_factor
        away["_evidence_factor_applied"] = round(adjustment.away_factor, 4)
    return home, away


def _build_edge(
    side: str,
    team: str,
    true_prob: float,
    calibrated_prob: float,
    market_odds: float,
    bankroll: float,
    n_iterations: int,
    calibration_audit: CalibrationAudit | None = None,
    market: str = "moneyline",
    line: float | None = None,
) -> EdgeDetail:
    """Compute edge detail for one side of a matchup."""
    market_prob = implied_probability(market_odds)
    edge_pct = edge_percentage(calibrated_prob, market_prob)
    ev_pct = expected_value_percent(calibrated_prob, market_odds)
    tier = "A" if n_iterations >= 1000 else "B"
    if abs(edge_pct) < 3.0:
        tier = "Pass"

    stake = recommend_stake(
        true_prob=calibrated_prob,
        odds=market_odds,
        bankroll=bankroll,
        confidence_tier=tier,
    )

    return EdgeDetail(
        side=side,
        team=team,
        market=market,
        line=line,
        true_prob=round(true_prob, 4),
        calibrated_prob=round(calibrated_prob, 4),
        market_implied=round(market_prob, 4),
        edge_pct=round(edge_pct, 2),
        ev_pct=round(ev_pct, 2),
        market_odds=market_odds,
        confidence_tier=tier,
        recommended_units=stake["units"],
        calibration_audit=calibration_audit,
    )


def _edge_selection_label(edge: EdgeDetail) -> str:
    """Human-readable selection string for an edge (shared legacy/portfolio)."""
    if edge.market == "total":
        return edge.team
    if edge.market == "spread" and edge.line is not None:
        return f"{edge.team} {edge.line:+g}"
    return f"{edge.team} {edge.side}"


def _portfolio_selection_enabled() -> bool:
    """Whether the portfolio-aware selector is active (default: off = legacy)."""
    return os.environ.get("OMEGA_PORTFOLIO_SELECTION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _edge_to_candidate(edge: EdgeDetail, *, league: str | None, matchup: str | None):
    """Build a portfolio BetCandidate from an EdgeDetail + game context."""
    from omega.core.betting.portfolio_selection import BetCandidate
    from omega.trace.portfolio_state import entity_keys_for

    descriptor = f"{edge.market}:{edge.side}" + (f":{edge.line:g}" if edge.line is not None else "")
    entity_keys = entity_keys_for(
        {
            "league": league or "",
            "matchup": matchup or "",
            "market": edge.market,
            "selection_descriptor": descriptor,
        }
    )
    return BetCandidate(
        selection=_edge_selection_label(edge),
        selection_descriptor=descriptor,
        market=edge.market,
        calibrated_prob=edge.calibrated_prob,
        odds=edge.market_odds,
        edge_pct=edge.edge_pct,
        ev_pct=edge.ev_pct,
        confidence_tier=edge.confidence_tier,
        entity_keys=entity_keys,
        league=league,
    )


def _pick_best_bet_via_portfolio(
    edges: list[EdgeDetail],
    bankroll: float,
    *,
    league: str | None,
    matchup: str | None,
) -> BetSlip | None:
    """Portfolio-aware path: return the top sized bet as a BetSlip."""
    from omega.core.betting.portfolio_selection import select_portfolio

    candidates = [_edge_to_candidate(e, league=league, matchup=matchup) for e in edges]
    selection = select_portfolio(candidates, bankroll=bankroll)
    if not selection.bets:
        return None
    top = selection.bets[0]
    return BetSlip(
        selection=top.candidate.selection,
        odds=top.candidate.odds,
        edge_pct=top.candidate.edge_pct,
        ev_pct=top.candidate.ev_pct,
        confidence_tier=top.candidate.confidence_tier,
        recommended_units=top.units,
        kelly_fraction=top.kelly_fraction,
    )


def _pick_best_bet(
    edges: list[EdgeDetail],
    bankroll: float,
    *,
    league: str | None = None,
    matchup: str | None = None,
) -> BetSlip | None:
    """Select the strongest edge and build a BetSlip, if any edge qualifies.

    With ``OMEGA_PORTFOLIO_SELECTION`` enabled, routes through the portfolio-aware
    selector and returns its top sized bet; otherwise (default) the legacy greedy
    max-EV pick, bit-for-bit unchanged.
    """
    if _portfolio_selection_enabled():
        return _pick_best_bet_via_portfolio(edges, bankroll, league=league, matchup=matchup)

    actionable = [e for e in edges if e.confidence_tier in ("A", "B") and e.ev_pct > 0]
    if not actionable:
        return None
    best = max(actionable, key=lambda e: e.ev_pct)
    stake = recommend_stake(
        true_prob=best.calibrated_prob,
        odds=best.market_odds,
        bankroll=bankroll,
        confidence_tier=best.confidence_tier,
    )
    return BetSlip(
        selection=_edge_selection_label(best),
        odds=best.market_odds,
        edge_pct=best.edge_pct,
        ev_pct=best.ev_pct,
        confidence_tier=best.confidence_tier,
        recommended_units=stake["units"],
        kelly_fraction=stake["kelly_fraction"],
    )


def _resolve_game_market_odds(
    odds: OddsInput,
    home_team: str,
    away_team: str,
) -> tuple[float | None, float | None]:
    """Resolve home/away moneyline odds from normalized markets first, legacy fields second."""
    home_ml = market_quote(odds, "moneyline", home_team, "Home")
    away_ml = market_quote(odds, "moneyline", away_team, "Away")

    home_odds = home_ml.price if home_ml is not None else odds.moneyline_home
    away_odds = away_ml.price if away_ml is not None else odds.moneyline_away
    return home_odds, away_odds


def _resolve_game_spread_market(
    odds: OddsInput,
    home_team: str,
    away_team: str,
) -> tuple[float | None, float | None, float | None]:
    """Resolve home spread line and both side prices."""
    home_spread = market_quote(odds, "spread", home_team, "Home")
    away_spread = market_quote(odds, "spread", away_team, "Away")
    line = (
        home_spread.line
        if home_spread is not None and home_spread.line is not None
        else odds.spread_home
    )
    if line is None:
        return None, None, None
    home_price = home_spread.price if home_spread is not None else odds.spread_home_price
    away_price = away_spread.price if away_spread is not None else odds.spread_away_price
    return line, home_price, away_price


def _resolve_game_total_market(odds: OddsInput) -> tuple[float | None, float | None, float | None]:
    """Resolve total line and over/under prices."""
    over_q = market_quote(odds, "total", "Over")
    under_q = market_quote(odds, "total", "Under")
    line = (
        over_q.line
        if over_q is not None and over_q.line is not None
        else (under_q.line if under_q is not None and under_q.line is not None else odds.over_under)
    )
    if line is None:
        return None, None, None
    over_price = over_q.price if over_q is not None else odds.total_over_price
    under_price = under_q.price if under_q is not None else odds.total_under_price
    return line, over_price, under_price


# ---------------------------------------------------------------------------
# analyze_game  -- primary entry point
# ---------------------------------------------------------------------------


def analyze_game(
    request: GameAnalysisRequest,
    bankroll: float = 1000.0,
    *,
    evidence_adjustment: PlaneAdjustment | None = None,
    evidence_plan: EvidenceExecutionPlan | None = None,
) -> GameAnalysisResponse:
    """Analyze a single game matchup. Never raises -- returns structured response.

    ``evidence_plan`` is the preferred backend-aware path for structured
    evidence. ``evidence_adjustment`` remains accepted for direct fast-score
    callers, but Markov always ignores handler adjustments and uses transition
    modifiers from the plan.
    """
    now = datetime.now().isoformat()
    matchup = f"{request.away_team} @ {request.home_team}"
    archetype_name = get_archetype_name(request.league)

    # Extract market lines so the engine can compute coverage probabilities.
    # Done before the engine call; odds may be absent.
    spread_value: float | None = None
    total_value: float | None = None
    suppressed_markets: list[str] = []
    suppress_total_market = request.league.upper() == "WNBA"
    if request.odds:
        spread_value, _, _ = _resolve_game_spread_market(
            request.odds, request.home_team, request.away_team
        )
        total_value, _, _ = _resolve_game_total_market(request.odds)
        if suppress_total_market and total_value is not None:
            suppressed_markets.append("WNBA:total")
            total_value = None

    effective_backend_name = _effective_game_backend_name(request)
    backend = resolve_game_backend(effective_backend_name)
    if backend is None:
        return GameAnalysisResponse(
            matchup=matchup,
            league=request.league,
            analyzed_at=now,
            status="skipped",
            skip_reason=f"Unsupported simulation_backend={effective_backend_name!r}",
            missing_requirements=["simulation_backend"],
            context_source="missing",
        )

    use_markov = _uses_transition_modifiers(backend)
    if evidence_plan is None:
        evidence_plan = _game_evidence_plan_for(request)
    effective_adjustment = None if use_markov else (
        evidence_plan.adjustment if evidence_plan else evidence_adjustment
    )
    if not use_markov and effective_adjustment is None:
        effective_adjustment = evidence_adjustment
    home_ctx, away_ctx = _apply_game_evidence(
        request.home_context, request.away_context, effective_adjustment
    )

    transition_modifiers: dict | None = (
        evidence_plan.transition_modifiers if use_markov and evidence_plan else None
    )
    if transition_modifiers:
        logger.debug(
            "Markov modifiers from %d evidence signals: %s",
            len(request.evidence),
            transition_modifiers,
        )

    try:
        sim_result = _engine.run_fast_game_simulation(
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league,
            n_iterations=request.n_iterations,
            home_context=home_ctx,
            away_context=away_ctx,
            seed=request.seed,
            spread_home=spread_value,
            over_under=total_value,
            allow_baseline=request.allow_baseline,
            transition_modifiers=transition_modifiers,
            prior_payload=request.prior_payload,
            backend=backend,
        )
    except Exception as exc:
        logger.warning("Simulation error for %s: %s", matchup, exc)
        return GameAnalysisResponse(
            matchup=matchup,
            league=request.league,
            analyzed_at=now,
            status="error",
            skip_reason=f"Simulation error: {exc}",
        )

    # Skipped -- propagate missing_requirements
    if not sim_result.get("success"):
        return GameAnalysisResponse(
            matchup=matchup,
            league=request.league,
            analyzed_at=now,
            status="skipped",
            skip_reason=sim_result.get("skip_reason", "Unknown skip"),
            missing_requirements=sim_result.get("missing_requirements"),
        )

    # Build simulation result
    simulation = SimulationResult(
        iterations=sim_result.get("iterations", request.n_iterations),
        home_win_prob=sim_result["home_win_prob"],
        away_win_prob=sim_result["away_win_prob"],
        draw_prob=sim_result.get("draw_prob"),
        predicted_spread=sim_result["predicted_spread"],
        predicted_total=sim_result["predicted_total"],
        predicted_home_score=sim_result.get("predicted_home_score", 0),
        predicted_away_score=sim_result.get("predicted_away_score", 0),
        context_source=sim_result.get("context_source", "provided"),
        baseline_used=bool(sim_result.get("baseline_used", False)),
        simulation_backend=sim_result.get("simulation_backend"),
        component_version=sim_result.get("component_version"),
    )

    # Edge analysis -- requires odds
    edges: list[EdgeDetail] = []
    data_sources = ["simulation"]

    if request.odds:
        data_sources.append("user_provided")
        home_prob = sim_result["home_win_prob"] / 100.0
        away_prob = sim_result["away_win_prob"] / 100.0
        draw_prob_raw = (sim_result.get("draw_prob") or 0.0) / 100.0

        # Normalize probabilities if draw exists
        total_prob = home_prob + away_prob + draw_prob_raw
        if total_prob > 0 and draw_prob_raw > 0:
            home_prob /= total_prob
            away_prob /= total_prob
            draw_prob_raw /= total_prob

        home_odds, away_odds = _resolve_game_market_odds(
            request.odds,
            request.home_team,
            request.away_team,
        )

        gc = request.game_context
        if home_odds is not None:
            cal_home, home_audit = _calibrate_audited(
                home_prob, league=request.league, context_hints=gc, plane="game", market="home"
            )
            home_edge = _build_edge(
                "home",
                request.home_team,
                home_prob,
                cal_home,
                home_odds,
                bankroll,
                request.n_iterations,
                calibration_audit=home_audit,
                market="moneyline",
            )
            edges.append(home_edge)

        if away_odds is not None:
            cal_away, away_audit = _calibrate_audited(
                away_prob, league=request.league, context_hints=gc, plane="game", market="away"
            )
            edges.append(
                _build_edge(
                    "away",
                    request.away_team,
                    away_prob,
                    cal_away,
                    away_odds,
                    bankroll,
                    request.n_iterations,
                    calibration_audit=away_audit,
                    market="moneyline",
                )
            )

        spread_line, home_spread_odds, away_spread_odds = _resolve_game_spread_market(
            request.odds, request.home_team, request.away_team
        )
        if spread_line is not None and "home_cover_prob" in sim_result:
            if home_spread_odds is not None:
                cover_prob = sim_result["home_cover_prob"] / 100.0
                cal_cover, cover_audit = _calibrate_audited(
                    cover_prob,
                    league=request.league,
                    context_hints=gc,
                    plane="game",
                    market="cover",
                )
                home_edge = _build_edge(
                    "home",
                    request.home_team,
                    cover_prob,
                    cal_cover,
                    home_spread_odds,
                    bankroll,
                    request.n_iterations,
                    calibration_audit=cover_audit,
                    market="spread",
                    line=spread_line,
                )
                edges.append(
                    home_edge.model_copy(update={"spread_coverage_prob": cover_prob})
                )
            if away_spread_odds is not None:
                away_cover_prob = sim_result["away_cover_prob"] / 100.0
                cal_away_cover, away_cover_audit = _calibrate_audited(
                    away_cover_prob,
                    league=request.league,
                    context_hints=gc,
                    plane="game",
                    market="cover",
                )
                away_edge = _build_edge(
                    "away",
                    request.away_team,
                    away_cover_prob,
                    cal_away_cover,
                    away_spread_odds,
                    bankroll,
                    request.n_iterations,
                    calibration_audit=away_cover_audit,
                    market="spread",
                    line=-spread_line,
                )
                edges.append(
                    away_edge.model_copy(update={"spread_coverage_prob": away_cover_prob})
                )

        total_line, over_odds, under_odds = _resolve_game_total_market(request.odds)
        if (
            not suppress_total_market
            and total_line is not None
            and "over_prob" in sim_result
            and "under_prob" in sim_result
        ):
            if over_odds is not None:
                over_prob = sim_result["over_prob"] / 100.0
                cal_over, over_audit = _calibrate_audited(
                    over_prob,
                    league=request.league,
                    context_hints=gc,
                    plane="game",
                    market="over",
                )
                edges.append(
                    _build_edge(
                        "over",
                        f"Over {total_line:g}",
                        over_prob,
                        cal_over,
                        over_odds,
                        bankroll,
                        request.n_iterations,
                        calibration_audit=over_audit,
                        market="total",
                        line=total_line,
                    )
                )
            if under_odds is not None:
                under_prob = sim_result["under_prob"] / 100.0
                cal_under, under_audit = _calibrate_audited(
                    under_prob,
                    league=request.league,
                    context_hints=gc,
                    plane="game",
                    market="under",
                )
                edges.append(
                    _build_edge(
                        "under",
                        f"Under {total_line:g}",
                        under_prob,
                        cal_under,
                        under_odds,
                        bankroll,
                        request.n_iterations,
                        calibration_audit=under_audit,
                        market="total",
                        line=total_line,
                    )
                )

        # 3-way moneyline (hockey regulation, soccer)
        if request.odds.moneyline_draw is not None and draw_prob_raw > 0:
            cal_draw, draw_audit = _calibrate_audited(
                draw_prob_raw, league=request.league, context_hints=gc, plane="game", market="draw"
            )
            edges.append(
                _build_edge(
                    "draw",
                    "Draw",
                    draw_prob_raw,
                    cal_draw,
                    request.odds.moneyline_draw,
                    bankroll,
                    request.n_iterations,
                    calibration_audit=draw_audit,
                    market="draw",
                )
            )

        # Exotic 3-way markets (soccer): double chance, draw-no-bet, BTTS,
        # correct score. Each builds an edge only when both a price and the
        # corresponding simulated probability are present. Probabilities reuse
        # the game calibration plane (cold-start) until exotic profiles exist.
        # (price_field, sim_prob_key, side, team_label, market_name)
        _exotic_specs = [
            ("dc_home_draw", "double_chance_home_draw_prob", "home_draw", "Home or Draw", "double_chance"),
            ("dc_home_away", "double_chance_home_away_prob", "home_away", "Home or Away", "double_chance"),
            ("dc_away_draw", "double_chance_away_draw_prob", "away_draw", "Away or Draw", "double_chance"),
            ("dnb_home", "dnb_home_prob", "home", request.home_team, "draw_no_bet"),
            ("dnb_away", "dnb_away_prob", "away", request.away_team, "draw_no_bet"),
            ("btts_yes", "btts_yes_prob", "yes", "Both Teams To Score", "both_teams_to_score"),
            ("btts_no", "btts_no_prob", "no", "No Both Teams To Score", "both_teams_to_score"),
        ]
        for price_field, prob_key, side, team_label, market_name in _exotic_specs:
            price = getattr(request.odds, price_field, None)
            sim_prob = sim_result.get(prob_key)
            if price is None or sim_prob is None:
                continue
            prob = sim_prob / 100.0
            if prob <= 0:
                continue
            cal_prob, audit = _calibrate_audited(
                prob, league=request.league, context_hints=gc, plane="game", market=market_name
            )
            edges.append(
                _build_edge(
                    side,
                    team_label,
                    prob,
                    cal_prob,
                    price,
                    bankroll,
                    request.n_iterations,
                    calibration_audit=audit,
                    market=market_name,
                )
            )

        # Sport-specific exotic/derivative markets (soccer Asian handicap +
        # first-half total today; NFL Wong teasers next) are priced by the
        # archetype's registered EdgeConsumer rather than an inline per-sport
        # ladder. The consumer resolves its own market lines from request.odds,
        # evaluates them against the backend pmfs in sim_result, and bridges the
        # exact EV into the binary edge framework via the injected calibrate /
        # build_edge helpers. Unmapped or no-consumer archetypes are a no-op.
        consumer = resolve_edge_consumer(archetype_name)
        if consumer is not None:
            edges.extend(
                consumer.consume(
                    sim_result,
                    request,
                    bankroll,
                    _calibrate_audited,
                    _build_edge,
                )
            )

        # Correct score: a map of scoreline -> price.
        cs_probs = sim_result.get("correct_score_probs") or {}
        if request.odds.correct_score and cs_probs:
            for scoreline, price in request.odds.correct_score.items():
                sim_prob = cs_probs.get(scoreline)
                if price is None or sim_prob is None:
                    continue
                prob = sim_prob / 100.0
                if prob <= 0:
                    continue
                cal_prob, audit = _calibrate_audited(
                    prob,
                    league=request.league,
                    context_hints=gc,
                    plane="game",
                    market="correct_score",
                )
                edges.append(
                    _build_edge(
                        scoreline,
                        f"Correct Score {scoreline}",
                        prob,
                        cal_prob,
                        price,
                        bankroll,
                        request.n_iterations,
                        calibration_audit=audit,
                        market="correct_score",
                    )
                )

    best_bet = (
        _pick_best_bet(edges, bankroll, league=request.league, matchup=matchup)
        if edges
        else None
    )

    return GameAnalysisResponse(
        matchup=matchup,
        league=request.league,
        analyzed_at=now,
        status="success",
        simulation=simulation,
        edges=edges,
        best_bet=best_bet,
        missing_requirements=[],
        context_source=sim_result.get("context_source", "provided"),
        baseline_used=bool(sim_result.get("baseline_used", False)),
        simulation_backend=sim_result.get("simulation_backend"),
        component_version=sim_result.get("component_version"),
        simulation_distributions=sim_result.get("simulation_distributions") or [],
        metadata=AnalysisMetadata(
            data_sources=data_sources,
            archetype=archetype_name,
            suppressed_markets=suppressed_markets,
        ),
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Context-adjusted input assembly
# ---------------------------------------------------------------------------

# Per-league, per-stat playoff suppression factors (conservative empirical defaults).
# Missing league falls back to _DEFAULT_PLAYOFF_FACTOR; missing stat within a
# known league also falls back to the default.
_PLAYOFF_STAT_FACTORS: dict[str, dict[str, float]] = {
    "NBA": {"pts": 0.96, "reb": 0.97, "ast": 0.96, "3pm": 0.94, "stl": 0.95, "blk": 0.95},
    "NHL": {"goals": 0.94, "assists": 0.95, "shots": 0.94, "saves": 0.97},
    "MLB": {"hits": 0.97, "hr": 0.93, "rbis": 0.96, "strikeouts": 1.02, "total_bases": 0.95},
    "NFL": {"pass_yds": 0.97, "rush_yds": 0.97, "rec_yds": 0.97, "receptions": 0.97},
    "EPL": {"goals": 0.96, "assists": 0.97, "shots": 0.95},
    "MLS": {"goals": 0.96, "assists": 0.97, "shots": 0.95},
}
_DEFAULT_PLAYOFF_FACTOR = 0.97
# Back-to-back fatigue -- only meaningful in sports with consecutive-night scheduling.
_B2B_FATIGUE: dict[str, float] = {"NBA": 0.94, "NHL": 0.95}
# MLB park factor applies to power/extra-base counting stats only.
_MLB_PARK_FACTOR_STATS = frozenset({"hr", "total_bases", "rbis"})
_MLB_PITCHER_DISTRIBUTION_PROPS = frozenset(
    {"strikeouts_pitched", "strikeouts", "k", "outs_recorded", "pitching_outs"}
)


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected_season_from_game_date(game_date: str) -> int | None:
    try:
        return int(game_date[:4])
    except (TypeError, ValueError):
        return None


def _pitcher_prop_distribution_requirements(
    request: PlayerPropRequest,
    player_context: dict[str, Any],
) -> list[str]:
    """Return missing input paths for MLB pitcher K/outs distributions."""
    if request.league.upper() != "MLB":
        return []
    if request.prop_type not in _MLB_PITCHER_DISTRIBUTION_PROPS:
        return []

    missing: list[str] = []
    mean_key = f"{request.prop_type}_mean"
    std_key = f"{request.prop_type}_std"
    if _coerce_optional_float(player_context.get(mean_key)) is None:
        missing.append(f"player_context.{mean_key}")
    if _coerce_optional_float(player_context.get(std_key)) is None:
        missing.append(f"player_context.{std_key}")

    sample_size = _coerce_optional_int(player_context.get("sample_size"))
    if sample_size is None or sample_size < 5:
        missing.append("player_context.sample_size>=5")

    expected_season = _expected_season_from_game_date(request.game_date)
    sample_season = _coerce_optional_int(
        player_context.get("sample_season", player_context.get("season"))
    )
    if expected_season is None:
        missing.append("game_date.year")
    elif sample_season != expected_season:
        missing.append(f"player_context.sample_season={expected_season}")

    return missing


def _apply_game_context(
    player_context: dict[str, Any],
    game_context: dict[str, Any] | None,
    prop_type: str,
    league: str,
    *,
    evidence_adjustment: PlaneAdjustment | None = None,
) -> dict[str, Any]:
    """Return an adjusted copy of player_context using game context + evidence.

    Two adjustment layers, applied in order to {prop_type}_mean / {prop_type}_std:
      1. Legacy ``game_context`` factor (playoff / B2B / pace / park) — unchanged.
      2. Structured evidence factor from ``evidence_adjustment`` — new. In shadow
         mode the evidence factors are 1.0 so this layer is a no-op.

    Never mutates the input dict. Records ``_context_factor_applied`` and (when
    evidence is present) ``_evidence_factor_applied`` + ``_adjustments_applied``
    for audit. Returns the original dict unchanged if mean_key is absent.
    """
    game_context = game_context or {}
    mean_key = f"{prop_type}_mean"
    std_key = f"{prop_type}_std"
    if mean_key not in player_context:
        return player_context

    ctx = dict(player_context)
    league_uc = league.upper()
    factor = 1.0

    if game_context.get("is_playoff"):
        league_factors = _PLAYOFF_STAT_FACTORS.get(league_uc, {})
        factor *= league_factors.get(prop_type, _DEFAULT_PLAYOFF_FACTOR)

    # B2B fatigue: rest_days=0 means the team played the previous night.
    # Only applied for sports with consecutive-night schedules (NBA, NHL).
    _rest_days = game_context.get("rest_days")
    rest_days = _coerce_optional_float(_rest_days)
    if rest_days is not None and int(rest_days) == 0 and league_uc in _B2B_FATIGUE:
        factor *= _B2B_FATIGUE[league_uc]

    # Pace scales counting stats proportionally (NBA, NHL, soccer).
    if (paf := game_context.get("pace_adjustment_factor")) is not None:
        pace_factor = _coerce_optional_float(paf)
        if pace_factor is not None:
            factor *= pace_factor

    # MLB park factor boosts/suppresses power stats in hitter/pitcher-friendly parks.
    if league_uc == "MLB" and prop_type in _MLB_PARK_FACTOR_STATS:
        if (park := game_context.get("park_factor")) is not None:
            park_factor = _coerce_optional_float(park)
            if park_factor is not None:
                factor *= park_factor

    mean_value = _coerce_optional_float(ctx[mean_key])
    if mean_value is None:
        return ctx

    ctx[mean_key] = mean_value * factor
    if std_key in ctx:
        std_value = _coerce_optional_float(ctx[std_key])
        if std_value is not None:
            ctx[std_key] = std_value * factor  # preserve coefficient of variation
    ctx["_context_factor_applied"] = round(factor, 4)

    # Structured evidence factor — applied after the legacy game_context factor.
    # The evidence handlers compute their factors against the raw {stat}_mean,
    # so the two layers compose multiplicatively and order-independently.
    if evidence_adjustment is not None:
        ctx[mean_key] = ctx[mean_key] * evidence_adjustment.mean_factor
        if std_key in ctx:
            ctx[std_key] = ctx[std_key] * evidence_adjustment.std_factor
        ctx["_evidence_factor_applied"] = {
            "mean": round(evidence_adjustment.mean_factor, 4),
            "std": round(evidence_adjustment.std_factor, 4),
            "mode": evidence_adjustment.evidence_mode,
        }
        ctx["_adjustments_applied"] = evidence_adjustment.applications()
    return ctx


# analyze_player_prop
# ---------------------------------------------------------------------------


def analyze_player_prop(
    request: PlayerPropRequest,
    bankroll: float = 1000.0,
    *,
    evidence_adjustment: PlaneAdjustment | None = None,
) -> PlayerPropResponse:
    """Analyze a single player prop. Never raises.

    Dispatches through the prop-backend registry (resolve_default_prop_backend
    -> resolve_prop_backend); the default distribution router performs
    archetype-aware Poisson/Normal sampling over a player rolling-stat
    distribution (mean, std) against the prop line. Caller must supply
    player_context with `{stat}_mean` and optionally `{stat}_std` keys (e.g. for
    pts: pts_mean=24.3, pts_std=6.1).

    ``evidence_adjustment`` is the precomputed structured-evidence adjustment;
    when omitted it is derived from ``request.evidence`` so direct callers and
    the ``analyze()`` wrapper get identical, deterministic behavior.
    """
    player_ctx = dict(request.player_context or {})
    stat_key = request.prop_type
    mean_key = f"{stat_key}_mean"
    std_key = f"{stat_key}_std"

    recent_perf = player_ctx.get("recent_performances")
    recent_mins = player_ctx.get("recent_minutes")
    if recent_perf and isinstance(recent_perf, list) and len(recent_perf) > 0:
        dud_count = 0
        non_zero_sum = 0.0
        non_zero_count = 0
        for i, val in enumerate(recent_perf):
            try:
                val_f = float(val)
                is_dud = (val_f <= 0.0)
                if recent_mins and isinstance(recent_mins, list) and i < len(recent_mins):
                    try:
                        mins_f = float(recent_mins[i])
                        if mins_f < 5.0:
                            is_dud = True
                    except (TypeError, ValueError):
                        pass

                if is_dud:
                    dud_count += 1
                else:
                    non_zero_sum += val_f
                    non_zero_count += 1
            except (TypeError, ValueError):
                continue

        total_valid = dud_count + non_zero_count
        if total_valid > 0:
            dud_prob = dud_count / total_valid
            true_mean = non_zero_sum / non_zero_count if non_zero_count > 0 else 0.0
            player_ctx[mean_key] = true_mean
            player_ctx["dud_prob"] = dud_prob

    pitcher_missing = _pitcher_prop_distribution_requirements(request, player_ctx)
    if pitcher_missing:
        return PlayerPropResponse(
            player_name=request.player_name,
            league=request.league,
            prop_type=request.prop_type,
            line=request.line,
            status="skipped",
            skip_reason=(
                "Missing required MLB pitcher prop distribution inputs: "
                + ", ".join(pitcher_missing)
            ),
            missing_requirements=pitcher_missing,
            context_source="provided",
            baseline_used=False,
        )

    if evidence_adjustment is None:
        evidence_adjustment = _player_adjustment_for(request)

    if request.game_context or evidence_adjustment is not None:
        player_ctx = _apply_game_context(
            player_ctx,
            request.game_context,
            stat_key,
            request.league,
            evidence_adjustment=evidence_adjustment,
        )

    mean = player_ctx.get(mean_key)
    if mean is None:
        return PlayerPropResponse(
            player_name=request.player_name,
            league=request.league,
            prop_type=request.prop_type,
            line=request.line,
            status="skipped",
            skip_reason=f"Missing required input player_context.{mean_key}",
            missing_requirements=[f"player_context.{mean_key}"],
        )

    try:
        mean_f = float(mean)
        std_raw = player_ctx.get(std_key)
        std_f = float(std_raw) if std_raw is not None else max(1.0, mean_f * 0.25)
    except (TypeError, ValueError) as exc:
        return PlayerPropResponse(
            player_name=request.player_name,
            league=request.league,
            prop_type=request.prop_type,
            line=request.line,
            status="error",
            skip_reason=f"Non-numeric player_context values: {exc}",
        )

    # B1: surface optional distribution override from caller. select_distribution()
    # also auto-routes low-mean basketball count stats (blk/stl/3pm/oreb/dreb/to
    # with mean < 3.0) to Poisson where Normal would understate right-tail mass.
    distribution_override = player_ctx.get("distribution")

    notes: list[str] = []

    # Dispatch through the prop-backend registry. The (league, stat_type) routing
    # table selects the model; an unregistered target (e.g. prop_neg_binom /
    # tennis_prop_serve pending Milestone 3) falls back to the distribution router
    # so the prop still prices, with the substitution recorded in notes. The
    # router forwards distribution + dud_prob from prior_payload, so this is
    # bit-identical to the prior direct run_player_simulation call.
    backend_name = resolve_default_prop_backend(request.league, stat_key)
    prop_backend = resolve_prop_backend(backend_name)
    if prop_backend is None:
        prop_backend = resolve_prop_backend("prop_distribution_router")
        notes.append(f"prop backend {backend_name!r} unregistered; using distribution router")

    prior_payload: dict[str, Any] = {
        "distribution": distribution_override,
        "dud_prob": float(player_ctx.get("dud_prob", 0.0)),
    }
    if backend_name == "prop_neg_binom":
        # Prefer a fitted/shrunk dispersion injected by the gatherer
        # (inject_prop_priors -> player_context.nb_dispersion_k) over the
        # per-request method-of-moments derivation, recording its provenance.
        injected_k = player_ctx.get("nb_dispersion_k")
        if injected_k is not None:
            try:
                injected_k_f = float(injected_k)
            except (TypeError, ValueError):
                injected_k_f = None
            if injected_k_f is not None and math.isfinite(injected_k_f) and injected_k_f > 0:
                prior_payload["nb_dispersion_k"] = injected_k_f
                notes.append(f"nb_k_source:{player_ctx.get('nb_k_source', 'supplied')}")
        if "nb_dispersion_k" not in prior_payload:
            variance = std_f**2
            if variance > mean_f:
                prior_payload["nb_dispersion_k"] = (mean_f**2) / (variance - mean_f)
            else:
                prior_payload["nb_dispersion_k"] = 100.0
    # Whitelisted sport-specific prop priors travel from player_context into
    # prior_payload (tennis serve model keys; harmless no-ops elsewhere).
    for prior_key in ("ace_rate", "serve_win_pct", "match_format", "expected_total_games"):
        if player_ctx.get(prior_key) is not None:
            prior_payload[prior_key] = player_ctx[prior_key]

    sim_input = PropSimulationInput(
        player_name=request.player_name,
        league=request.league,
        stat_type=stat_key,
        line=request.line,
        projection_mean=mean_f,
        n_iter=request.n_iterations,
        seed=request.seed,
        projection_std=std_f,
        prior_payload=prior_payload,
    )

    try:
        sim_result = prop_backend.run(sim_input)
    except Exception as exc:
        logger.warning("Prop simulation error for %s: %s", request.player_name, exc)
        return PlayerPropResponse(
            player_name=request.player_name,
            league=request.league,
            prop_type=request.prop_type,
            line=request.line,
            status="error",
            skip_reason=f"Simulation error: {exc}",
        )

    # run_player_simulation returns probabilities as decimals (0-1)
    over_prob = sim_result.get("over_prob", 0.0)
    under_prob = sim_result.get("under_prob", 0.0)
    resolved_dist = select_distribution(
        stat_key,
        request.league,
        mean=mean_f,
        override=distribution_override,
    )
    if distribution_override in {"normal", "poisson"}:
        notes.append(f"distribution_override:{resolved_dist}")

    sim_distribution = {
        "target": "player_stat",
        "market": "player_prop",
        "stat_key": stat_key,
        "distribution_type": sim_result.get("distribution_type", resolved_dist),
        "distribution_params": sim_result.get("distribution_params") or {},
        "params_schema_version": 1,
        "sample_mean": sim_result.get("mean"),
        "sample_std": sim_result.get("std"),
        "p10": sim_result.get("p10"),
        "p50": sim_result.get("p50"),
        "p90": sim_result.get("p90"),
        "n_iterations": request.n_iterations,
        "seed": request.seed,
        "context_hash": _stable_input_hash(
            {
                "league": request.league,
                "prop_type": stat_key,
                "player_context": player_ctx,
                "game_context": request.game_context,
            }
        ),
        "component_version": "player_prop_fast_v1",
    }

    # B2: imputation provenance -- LLM declares which observation slots are
    # imputed and (ideally) the underlying sample size. We cap tiers based on
    # the imputed fraction so a 4-of-5-imputed std cannot ride an A-tier edge.
    raw_imputed = player_ctx.get("imputed_keys") or []
    imputed_keys = [str(k) for k in raw_imputed if k is not None]
    sample_size_raw = player_ctx.get("sample_size")
    try:
        sample_size = int(sample_size_raw) if sample_size_raw is not None else None
    except (TypeError, ValueError):
        sample_size = None

    imputed_fraction: float | None
    if imputed_keys and (sample_size is None or sample_size <= 0):
        imputed_fraction = 1.0
        notes.append("imputed_keys_provided_without_sample_size")
    elif imputed_keys:
        assert sample_size is not None
        imputed_fraction = min(1.0, len(imputed_keys) / float(sample_size))
    else:
        imputed_fraction = 0.0

    # B4: single-side odds -- "implied opposite" is forbidden. If only one
    # side is sourced, compute that side's edge only and annotate the other.
    edge_over: float | None = None
    edge_under: float | None = None
    recommendation = "pass"
    tier: str | None = None
    over_audit: CalibrationAudit | None = None
    under_audit: CalibrationAudit | None = None
    cal_over = over_prob
    cal_under = under_prob

    _ctx_hints = request.game_context or None
    if request.odds_over is not None:
        market_over = implied_probability(request.odds_over)
        cal_over, over_audit = _calibrate_audited(
            over_prob, league=request.league, context_hints=_ctx_hints, plane="prop", market="over"
        )
        edge_over = round(edge_percentage(cal_over, market_over), 2)
    else:
        notes.append("odds_unsourced_over")

    if request.odds_under is not None:
        market_under = implied_probability(request.odds_under)
        cal_under, under_audit = _calibrate_audited(
            under_prob,
            league=request.league,
            context_hints=_ctx_hints,
            plane="prop",
            market="under",
        )
        edge_under = round(edge_percentage(cal_under, market_under), 2)
    else:
        notes.append("odds_unsourced_under")

    if edge_over is not None and edge_under is not None:
        if edge_over > 3.0 and edge_over > edge_under:
            recommendation = "over"
            tier = "A" if request.n_iterations >= 1000 else "B"
        elif edge_under > 3.0:
            recommendation = "under"
            tier = "A" if request.n_iterations >= 1000 else "B"
    elif edge_over is not None and edge_over > 3.0:
        recommendation = "over"
        tier = "A" if request.n_iterations >= 1000 else "B"
    elif edge_under is not None and edge_under > 3.0:
        recommendation = "under"
        tier = "A" if request.n_iterations >= 1000 else "B"

    # B2 cap: imputation discipline supersedes engine tier
    if imputed_fraction is not None and imputed_fraction > 0.4:
        tier = None
        recommendation = "pass"
        notes.append("insufficient_real_observations")
    elif imputed_fraction is not None and imputed_fraction > 0.2 and tier == "A":
        tier = "B"
        notes.append("tier_capped_imputation")

    kelly: float | None = None
    units: float | None = None
    bet_side_odds: float | None = None
    if recommendation == "over" and request.odds_over is not None and tier is not None:
        stake = recommend_stake(
            true_prob=cal_over,
            odds=request.odds_over,
            bankroll=bankroll,
            confidence_tier=tier,
        )
        kelly = stake["kelly_fraction"]
        units = stake["units"]
        bet_side_odds = request.odds_over
    elif recommendation == "under" and request.odds_under is not None and tier is not None:
        stake = recommend_stake(
            true_prob=cal_under,
            odds=request.odds_under,
            bankroll=bankroll,
            confidence_tier=tier,
        )
        kelly = stake["kelly_fraction"]
        units = stake["units"]
        bet_side_odds = request.odds_under

    return PlayerPropResponse(
        player_name=request.player_name,
        league=request.league,
        prop_type=request.prop_type,
        line=request.line,
        status="success",
        over_prob=round(over_prob, 4),
        under_prob=round(under_prob, 4),
        projection_mean=round(float(sim_result.get("mean", 0.0)), 4),
        projection_std=round(float(sim_result.get("std", 0.0)), 4),
        projection_p10=round(float(sim_result.get("p10", 0.0)), 4),
        projection_p50=round(float(sim_result.get("p50", 0.0)), 4),
        projection_p90=round(float(sim_result.get("p90", 0.0)), 4),
        distribution_type=sim_result.get("distribution_type", resolved_dist),
        edge_over=edge_over,
        edge_under=edge_under,
        recommendation=recommendation,
        confidence_tier=tier,
        kelly_fraction=kelly,
        recommended_units=units,
        bet_side_odds=bet_side_odds,
        missing_requirements=[],
        notes=notes,
        imputed_fraction=imputed_fraction,
        over_calibration_audit=over_audit,
        under_calibration_audit=under_audit,
        context_source="provided",
        baseline_used=False,
        simulation_distributions=[sim_distribution],
    )


# ---------------------------------------------------------------------------
# analyze_slate
# ---------------------------------------------------------------------------


def analyze_slate(
    request: SlateAnalysisRequest,
    games: list[dict[str, Any]] | None = None,
) -> SlateAnalysisResponse:
    """Analyze a slate of games. Loops analyze_game per game; catches errors per-game.
    Does not fetch games; caller must supply request.games or games argument."""
    date_str = request.date or datetime.now().strftime("%Y-%m-%d")

    games = games if games is not None else request.games
    if not games:
        games = []

    analyses: list[GameAnalysisResponse] = []
    for game in games:
        home = _extract_team(game, "home_team") or _extract_team(game, "home")
        away = _extract_team(game, "away_team") or _extract_team(game, "away")
        if not home or not away:
            continue

        odds_dict = game.get("odds") or game.get("markets") or {}
        odds_input = None
        if odds_dict:
            odds_input = OddsInput(
                spread_home=odds_dict.get("spread_home"),
                spread_home_price=odds_dict.get("spread_home_price", -110),
                moneyline_home=odds_dict.get("moneyline_home"),
                moneyline_away=odds_dict.get("moneyline_away"),
                over_under=odds_dict.get("over_under"),
                moneyline_draw=odds_dict.get("moneyline_draw"),
                markets=odds_dict.get("markets"),
            )

        game_kwargs: dict[str, Any] = {
            "home_team": home,
            "away_team": away,
            "league": request.league,
            "odds": odds_input,
            "home_context": game.get("home_context"),
            "away_context": game.get("away_context"),
            "game_context": game.get("game_context"),
            "allow_baseline": bool(game.get("allow_baseline", False)),
            "simulation_backend": game.get("simulation_backend", "fast_score"),
        }
        if game.get("n_iterations") is not None:
            game_kwargs["n_iterations"] = game.get("n_iterations")
        if game.get("seed") is not None:
            game_kwargs["seed"] = game.get("seed")

        game_request = GameAnalysisRequest(**game_kwargs)
        result = analyze_game(game_request, bankroll=request.bankroll)
        analyses.append(result)

    games_with_edge = sum(1 for a in analyses if a.best_bet is not None)

    return SlateAnalysisResponse(
        league=request.league,
        date=date_str,
        total_games=len(games),
        games_analyzed=len(analyses),
        games_with_edge=games_with_edge,
        analyses=analyses,
    )


def _extract_team(game: dict[str, Any], key: str) -> str | None:
    """Safely extract team name from various game dict shapes."""
    val = game.get(key)
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("name") or val.get("team") or val.get("full_name")
    return None
