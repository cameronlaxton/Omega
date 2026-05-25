"""
Stable service interface wrapping Omega core internals.

JSON-in/JSON-out service layer. Caller supplies all context;
no data fetching, no config loading, no network calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
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
from omega.core.calibration.probability import apply_calibration, apply_calibration_audited
from omega.core.contracts.schemas import (
    AnalysisMetadata,
    BetSlip,
    CalibrationAudit,
    EdgeDetail,
    GameAnalysisRequest,
    GameAnalysisResponse,
    MarketQuote,
    OddsInput,
    PlayerPropRequest,
    PlayerPropResponse,
    SimulationResult,
    SlateAnalysisRequest,
    SlateAnalysisResponse,
)
from omega.core.simulation.archetypes import get_archetype_name
from omega.core.simulation.engine import (
    MarkovGameSimulationBackend,
    OmegaSimulationEngine,
    run_player_simulation,
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
    MAPPED_SIGNAL_TYPES,
    signals_to_transition_modifiers,
)

UTC = timezone.utc

logger = logging.getLogger("omega.service")

_engine = OmegaSimulationEngine()
MODEL_VERSION = "omega-core-phase6h"
_TRACE_HASH_EXCLUDE = {
    "odds",
    "odds_over",
    "odds_under",
    "markets",
    "odds_snapshot",
    "market_snapshots",
    "closing_line",
    "closing_lines",
}


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


def _sanitize_for_trace_hash(value: Any) -> Any:
    """Drop volatile market fields before deriving trace hash identity."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(k): _sanitize_for_trace_hash(v)
            for k, v in value.items()
            if str(k) not in _TRACE_HASH_EXCLUDE
        }
    if isinstance(value, list):
        return [_sanitize_for_trace_hash(item) for item in value]
    return value


def _stable_input_hash(request: Any) -> str | None:
    """Stable 8-char content hash, excluding volatile odds and close snapshots."""
    try:
        payload = _sanitize_for_trace_hash(request)
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8]
    except Exception:
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

    # Structured evidence is planned once here, then passed into the analyzer.
    # The same plan supplies both the applied effect and the per-signal
    # applications recorded on the trace, so there is exactly one evaluation per
    # analysis.
    evidence_plan: EvidenceExecutionPlan | None = None
    if isinstance(typed_req, PlayerPropRequest):
        evidence_plan = _player_evidence_plan_for(typed_req)
    elif isinstance(typed_req, GameAnalysisRequest):
        evidence_plan = _game_evidence_plan_for(typed_req)

    result: GameAnalysisResponse | PlayerPropResponse | SlateAnalysisResponse
    if isinstance(typed_req, GameAnalysisRequest):
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
    if evidence_plan is not None:
        evidence_mode = evidence_plan.evidence_mode
        evidence_application = evidence_plan.evidence_application

    # Assemble trace_quality last — pure audit metadata, never influences math above.
    result_downgrades = _result_downgrades(result)
    evidence_signals = getattr(typed_req, "evidence", None) or []
    evidence_status = "present" if evidence_signals else "empty"
    context_source = _result_context_source(result)
    baseline_used = _result_baseline_used(result)
    identity_status = _identity_status(kind, typed_req)
    calibration_exclusion_reasons: list[str] = []
    if _result_status(result) != "success":
        calibration_exclusion_reasons.append("engine_skipped")
    if context_source != "provided":
        calibration_exclusion_reasons.append(
            "baseline_default_context" if baseline_used else "legacy_missing_context_source"
        )
    if identity_status != "complete":
        calibration_exclusion_reasons.append("legacy_missing_identity")
    calibration_exclusion_reasons.extend(str(d) for d in result_downgrades)
    calibration_exclusion_reasons = sorted(set(calibration_exclusion_reasons))
    calibration_eligible = not calibration_exclusion_reasons

    engine_trace_quality: dict[str, Any] = {
        "aggregate_quality": None,
        "downgrades": result_downgrades,
        "passed": len(result_downgrades) == 0,
        "evidence_status": evidence_status,
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
            "evidence_status": evidence_status,  # engine-computed; caller cannot override
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
    calibrated, d = apply_calibration_audited(raw_prob, league=league, context_hints=context_hints)
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


def _markov_evidence_applications(
    evidence: list[Any],
    active_indices: set[int],
    suppressed_indices: set[int],
) -> list[dict[str, Any]]:
    applications: list[dict[str, Any]] = []
    for idx, sig in enumerate(evidence):
        signal_type = str(getattr(sig, "signal_type", ""))
        if idx in suppressed_indices:
            applications.append(_suppression_record(sig, "markov_transition").as_application())
        elif idx in active_indices and signal_type in MAPPED_SIGNAL_TYPES:
            applications.append(
                {
                    "signal_type": signal_type,
                    "target": "markov_transition",
                    "applied": True,
                    "factor": None,
                    "reason": "mapped_to_markov_transition_modifiers",
                    "policy_version": "markov_state_v1",
                    "evidence_mode": "markov_transition",
                }
            )
        else:
            applications.append(
                {
                    "signal_type": signal_type,
                    "target": "skip",
                    "applied": False,
                    "factor": 1.0,
                    "reason": "no Markov transition mapping for signal_type",
                    "policy_version": "markov_state_v1",
                    "evidence_mode": "markov_transition",
                }
            )
    return applications


def _game_evidence_plan_for(request: GameAnalysisRequest) -> EvidenceExecutionPlan:
    evidence = list(getattr(request, "evidence", None) or [])
    if not evidence:
        return EvidenceExecutionPlan()

    suppressed = _suppressed_player_signal_indices(evidence)
    if request.simulation_backend == "markov_state":
        active_indices = {idx for idx in range(len(evidence)) if idx not in suppressed}
        active_evidence = [sig for idx, sig in enumerate(evidence) if idx in active_indices]
        transition_modifiers = signals_to_transition_modifiers(
            active_evidence, home_team=request.home_team
        )
        applications = _markov_evidence_applications(evidence, active_indices, suppressed)
        return EvidenceExecutionPlan(
            adjustment=None,
            transition_modifiers=transition_modifiers or None,
            evidence_mode="markov_transition",
            evidence_application=applications,
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


def _pick_best_bet(
    edges: list[EdgeDetail],
    bankroll: float,
) -> BetSlip | None:
    """Select the strongest edge and build a BetSlip, if any edge qualifies."""
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
        selection=f"{best.team} {best.side}",
        odds=best.market_odds,
        edge_pct=best.edge_pct,
        ev_pct=best.ev_pct,
        confidence_tier=best.confidence_tier,
        recommended_units=stake["units"],
        kelly_fraction=stake["kelly_fraction"],
    )


def _quote_matches_selection(quote: MarketQuote, *labels: str) -> bool:
    selection = quote.selection.strip().casefold()
    return any(selection == label.strip().casefold() for label in labels if label)


def _market_quote(
    odds: OddsInput,
    market_type: str,
    *labels: str,
) -> MarketQuote | None:
    for quote in odds.markets or []:
        if quote.market_type != market_type:
            continue
        if _quote_matches_selection(quote, *labels):
            return quote
    return None


def _resolve_game_market_odds(
    odds: OddsInput,
    home_team: str,
    away_team: str,
) -> tuple[float | None, float | None]:
    """Resolve home/away odds from normalized markets first, legacy fields second."""
    home_spread = _market_quote(odds, "spread", home_team, "Home")
    home_ml = _market_quote(odds, "moneyline", home_team, "Home")
    away_ml = _market_quote(odds, "moneyline", away_team, "Away")

    # Only use spread_home_price when spread_home was explicitly supplied;
    # otherwise fall through to moneyline_home. The spread_home_price field
    # defaults to -110 which would otherwise shadow a real moneyline_home value.
    legacy_home = odds.spread_home_price if odds.spread_home is not None else odds.moneyline_home
    home_odds = (
        home_spread.price
        if home_spread is not None
        else (home_ml.price if home_ml is not None else legacy_home)
    )
    away_odds = away_ml.price if away_ml is not None else odds.moneyline_away
    return home_odds, away_odds


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

    # Extract spread value so the engine can compute coverage probabilities.
    # Done before the engine call; odds may be absent.
    spread_value: float | None = None
    if request.odds:
        _hsq = _market_quote(request.odds, "spread", request.home_team, "Home")
        if _hsq and _hsq.line is not None:
            spread_value = _hsq.line
        elif request.odds.spread_home is not None:
            spread_value = request.odds.spread_home

    if request.simulation_backend not in {"fast_score", "markov_state"}:
        return GameAnalysisResponse(
            matchup=matchup,
            league=request.league,
            analyzed_at=now,
            status="skipped",
            skip_reason=f"Unsupported simulation_backend={request.simulation_backend!r}",
            missing_requirements=["simulation_backend"],
            context_source="missing",
        )

    use_markov = request.simulation_backend == "markov_state"
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
        engine = (
            OmegaSimulationEngine(game_backend=MarkovGameSimulationBackend())
            if use_markov
            else _engine
        )
        sim_result = engine.run_fast_game_simulation(
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league,
            n_iterations=request.n_iterations,
            home_context=home_ctx,
            away_context=away_ctx,
            seed=request.seed,
            spread_home=spread_value,
            allow_baseline=request.allow_baseline,
            transition_modifiers=transition_modifiers,
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

        # Detect whether the home market is a spread/run-line so we can
        # substitute coverage probability for outright win probability.
        home_spread_q = _market_quote(request.odds, "spread", request.home_team, "Home")
        home_ml_q = _market_quote(request.odds, "moneyline", request.home_team, "Home")
        is_home_spread_market = (home_spread_q is not None) or (
            request.odds.spread_home is not None
            and home_ml_q is None
            and request.odds.moneyline_home is None
        )

        gc = request.game_context
        if home_odds is not None:
            if is_home_spread_market and "home_cover_prob" in sim_result:
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
                    home_odds,
                    bankroll,
                    request.n_iterations,
                    calibration_audit=cover_audit,
                )
                home_edge = home_edge.model_copy(update={"spread_coverage_prob": cover_prob})
            else:
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
                )
            )

    best_bet = _pick_best_bet(edges, bankroll) if edges else None

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


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    Uses run_player_simulation: archetype-aware Poisson/Normal sampling over
    a player rolling-stat distribution (mean, std) against the prop line.
    Caller must supply player_context with `{stat}_mean` and optionally
    `{stat}_std` keys (e.g. for pts: pts_mean=24.3, pts_std=6.1).

    ``evidence_adjustment`` is the precomputed structured-evidence adjustment;
    when omitted it is derived from ``request.evidence`` so direct callers and
    the ``analyze()`` wrapper get identical, deterministic behavior.
    """
    player_ctx = request.player_context or {}
    stat_key = request.prop_type
    mean_key = f"{stat_key}_mean"
    std_key = f"{stat_key}_std"

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

    player_proj = {
        "league": request.league,
        "stat_key": stat_key,
        "mean": mean_f,
        "variance": std_f**2,
        "market_line": request.line,
        "distribution": distribution_override,
    }

    try:
        sim_result = run_player_simulation(
            player_proj,
            n_iter=request.n_iterations,
            seed=request.seed,
        )
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

    notes: list[str] = []
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
