"""
Sandbox entry point — what an LLM analysis tool actually calls.

Wraps service-layer analyzers with:
    * A sandbox trace_id so output is distinguishable from a real local run
    * Input echo for audit
    * Quality-gate enforcement (refuses bet cards when sim was skipped)
    * Optional plan-level quality gate for callers that want to "score" a
      request the way the canonical orchestrator would
"""

from __future__ import annotations

import hashlib
import json as _json_for_trace_hash
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from omega_lite._quality_helpers import build_data_completeness, compute_aggregate_quality
from omega_lite.archetypes import (
    get_archetype,
    get_critical_inputs,
    get_important_inputs,
)
from omega_lite.models import (
    AnswerPlan,
    ExecutionMode,
    GatheredFact,
    GatherSlot,
    InputImportance,
    OutputPackage,
    ProviderResult,
)
from omega_lite.quality_gate import apply_quality_gate
from omega_lite.schemas import (
    GameAnalysisRequest,
    GameAnalysisResponse,
    OddsInput,
    PlayerPropRequest,
    PlayerPropResponse,
    SlateAnalysisRequest,
    SlateAnalysisResponse,
)
from omega_lite.service import (
    analyze_game,
    analyze_player_prop,
    analyze_slate,
)


MODEL_VERSION = "omega-lite-v1"


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

_TRACE_HASH_EXCLUDE = {"odds_over", "odds_under"}


def _input_hash(request: Any) -> Optional[str]:
    """Stable 8-char content hash of a request, excluding volatile odds fields.

    Returns None if the request cannot be serialized.
    """
    try:
        if hasattr(request, "model_dump"):
            payload = request.model_dump(exclude=_TRACE_HASH_EXCLUDE)
        elif isinstance(request, dict):
            payload = {k: v for k, v in request.items() if k not in _TRACE_HASH_EXCLUDE}
        else:
            return None
        encoded = _json_for_trace_hash.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return None


def _new_trace_id(request: Any = None) -> str:
    """Sandbox-prefixed trace id. The 'sandbox-' prefix is the honesty signal:
    a consumer can tell at a glance this came from omega_lite, not the
    canonical pipeline.

    When a request is supplied, the id encodes a stable input hash so reruns
    of identical inputs share the hash prefix: ``sandbox-<hash[:8]>-<nonce[:4]>``.
    Odds fields are excluded from the hash because mid-flight juice changes
    should not invalidate the underlying simulation context.
    """
    if request is not None:
        h = _input_hash(request)
        if h is not None:
            return f"sandbox-{h}-{uuid.uuid4().hex[:4]}"
    return f"sandbox-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Synthetic GatheredFact construction
# ---------------------------------------------------------------------------

def _synth_facts_from_game_request(req: GameAnalysisRequest) -> List[GatheredFact]:
    """Build synthetic GatheredFact list from a GameAnalysisRequest so the
    plan-level quality gate can score it.

    Each archetype-required team key becomes one GatherSlot per side. We mark
    the slot filled if the corresponding value is present and numeric in the
    provided context dict.
    """
    archetype = get_archetype(req.league)
    if archetype is None:
        return []

    facts: List[GatheredFact] = []
    critical_keys = set(archetype.critical_team_keys)

    sides = (
        ("home", req.home_team, req.home_context),
        ("away", req.away_team, req.away_context),
    )

    for side, team, ctx in sides:
        ctx = ctx or {}
        for key in archetype.required_team_keys:
            importance = (
                InputImportance.CRITICAL if key in critical_keys
                else InputImportance.IMPORTANT
            )
            slot = GatherSlot(
                key=f"{side}_team.{key}",
                data_type="team_stat",
                entity=team,
                league=req.league,
                importance=importance,
            )
            value = ctx.get(key)
            filled = isinstance(value, (int, float)) and value != 0
            result = None
            quality = 0.0
            if filled:
                result = ProviderResult(
                    data={key: value},
                    source="user_supplied",
                    method="user_input",
                    confidence=1.0,
                )
                quality = 1.0
            facts.append(GatheredFact(slot=slot, result=result, filled=filled, quality_score=quality))

    # Odds slot — important but not critical for the sim itself
    odds_slot = GatherSlot(
        key="odds",
        data_type="odds",
        entity=f"{req.away_team} @ {req.home_team}",
        league=req.league,
        importance=InputImportance.IMPORTANT,
    )
    odds_filled = req.odds is not None and (
        req.odds.moneyline_home is not None
        or req.odds.moneyline_away is not None
        or req.odds.spread_home is not None
        or req.odds.over_under is not None
    )
    odds_result = None
    if odds_filled:
        odds_result = ProviderResult(
            data=req.odds.model_dump(exclude_none=True) if req.odds else {},
            source="user_supplied",
            method="user_input",
            confidence=1.0,
        )
    facts.append(GatheredFact(
        slot=odds_slot,
        result=odds_result,
        filled=bool(odds_filled),
        quality_score=1.0 if odds_filled else 0.0,
    ))

    return facts


def _synth_facts_from_prop_request(req: PlayerPropRequest) -> List[GatheredFact]:
    """Same idea for prop requests. The single CRITICAL slot is
    player_context.{stat}_mean — without it analyze_player_prop refuses.

    B2: when the caller declares ``imputed_keys`` in player_context, the
    mean and std are derived from a partially-imputed sample. Decay their
    quality_score and mark the provider method as ``imputed``.
    """
    ctx = req.player_context or {}
    stat = req.prop_type
    mean_key = f"{stat}_mean"
    std_key = f"{stat}_std"

    raw_imputed = ctx.get("imputed_keys") or []
    imputed_keys = [str(k) for k in raw_imputed if k is not None]
    sample_size_raw = ctx.get("sample_size")
    try:
        sample_size = int(sample_size_raw) if sample_size_raw is not None else None
    except (TypeError, ValueError):
        sample_size = None

    if imputed_keys and sample_size and sample_size > 0:
        imputed_fraction = min(1.0, len(imputed_keys) / float(sample_size))
    elif imputed_keys:
        imputed_fraction = 1.0
    else:
        imputed_fraction = 0.0

    quality_when_filled = max(0.5, 1.0 - 0.5 * imputed_fraction)
    method_label = "imputed" if imputed_fraction > 0.0 else "user_input"

    facts: List[GatheredFact] = []

    mean_slot = GatherSlot(
        key=f"player_context.{mean_key}",
        data_type="player_stat",
        entity=req.player_name,
        league=req.league,
        importance=InputImportance.CRITICAL,
    )
    mean_filled = isinstance(ctx.get(mean_key), (int, float))
    facts.append(GatheredFact(
        slot=mean_slot,
        result=ProviderResult(
            data={mean_key: ctx[mean_key]},
            source="user_supplied",
            method=method_label,
            confidence=quality_when_filled,
        ) if mean_filled else None,
        filled=mean_filled,
        quality_score=quality_when_filled if mean_filled else 0.0,
    ))

    std_slot = GatherSlot(
        key=f"player_context.{std_key}",
        data_type="player_stat",
        entity=req.player_name,
        league=req.league,
        importance=InputImportance.IMPORTANT,
    )
    std_filled = isinstance(ctx.get(std_key), (int, float))
    # std is more sensitive to imputation than mean (median imputation
    # systematically compresses std), so decay it harder.
    std_quality = max(0.3, 1.0 - 0.7 * imputed_fraction) if std_filled else 0.0
    facts.append(GatheredFact(
        slot=std_slot,
        result=ProviderResult(
            data={std_key: ctx.get(std_key)},
            source="user_supplied",
            method=method_label,
            confidence=std_quality,
        ) if std_filled else None,
        filled=std_filled,
        quality_score=std_quality,
    ))

    odds_slot = GatherSlot(
        key="odds",
        data_type="odds",
        entity=req.player_name,
        league=req.league,
        importance=InputImportance.IMPORTANT,
    )
    odds_filled = req.odds_over is not None or req.odds_under is not None
    facts.append(GatheredFact(
        slot=odds_slot,
        result=ProviderResult(
            data={"odds_over": req.odds_over, "odds_under": req.odds_under},
            source="user_supplied",
            method="user_input",
            confidence=1.0,
        ) if odds_filled else None,
        filled=odds_filled,
        quality_score=1.0 if odds_filled else 0.0,
    ))

    return facts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    request: Union[Dict[str, Any], GameAnalysisRequest, PlayerPropRequest, SlateAnalysisRequest],
    bankroll: float = 1000.0,
    apply_plan_gate: bool = True,
) -> Dict[str, Any]:
    """Run an omega_lite analysis from inside an LLM sandbox.

    Dispatches by request type. Returns a dict with:
        - trace_id          (sandbox-XXXX)
        - model_version     ("omega-lite-v1")
        - ran_at            (ISO timestamp)
        - kind              ("game" | "prop" | "slate")
        - input_snapshot    (echo of the request, for audit)
        - result            (GameAnalysisResponse / PlayerPropResponse /
                             SlateAnalysisResponse as a dict)
        - quality_gate      (downgrade summary, optional)

    Args:
        request: Either a dict matching the appropriate Pydantic schema, or
                 an already-constructed request object.
        bankroll: Bankroll for Kelly staking (ignored for prop requests today).
        apply_plan_gate: If True, run the plan-level quality gate over the
                         request inputs and include downgrades in the output.
                         Set False for raw service-layer behavior.
    """
    # Coerce dicts into typed requests
    typed_req = _coerce_request(request)

    trace_id = _new_trace_id(typed_req)
    ran_at = datetime.now(timezone.utc).isoformat()

    if isinstance(typed_req, GameAnalysisRequest):
        result = analyze_game(typed_req, bankroll=bankroll)
        kind = "game"
        facts = _synth_facts_from_game_request(typed_req) if apply_plan_gate else []
    elif isinstance(typed_req, PlayerPropRequest):
        result = analyze_player_prop(typed_req, bankroll=bankroll)
        kind = "prop"
        facts = _synth_facts_from_prop_request(typed_req) if apply_plan_gate else []
    elif isinstance(typed_req, SlateAnalysisRequest):
        result = analyze_slate(typed_req)
        kind = "slate"
        facts = []  # slate gate is per-game inside analyze_slate
    else:
        raise TypeError(
            f"Unsupported request type {type(typed_req).__name__}. "
            "Expected GameAnalysisRequest, PlayerPropRequest, or SlateAnalysisRequest."
        )

    output: Dict[str, Any] = {
        "trace_id": trace_id,
        "model_version": MODEL_VERSION,
        "ran_at": ran_at,
        "kind": kind,
        "input_snapshot": _safe_dump(typed_req),
        "result": _safe_dump(result),
    }

    if apply_plan_gate and facts:
        starter_plan = _starter_plan_for(kind)
        revised = apply_quality_gate(starter_plan, facts)
        output["quality_gate"] = {
            "applied": True,
            "aggregate_quality": round(compute_aggregate_quality(facts), 3),
            "data_completeness": build_data_completeness(facts),
            "downgrades": list(revised.downgrades),
            "final_packages": [p.value for p in revised.output_packages],
            "final_modes": [m.value for m in revised.execution_modes],
        }
    elif apply_plan_gate:
        output["quality_gate"] = {"applied": False, "reason": "no synthetic facts for this kind"}

    return output


def _starter_plan_for(kind: str) -> AnswerPlan:
    """The default 'wanted' plan before the quality gate runs."""
    if kind == "slate":
        return AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.BET_CARD, OutputPackage.COMPACT_SUMMARY],
            simulation_required=True,
            betting_recommendations_included=True,
            quality_thresholds={OutputPackage.BET_CARD.value: 0.7},
        )
    if kind == "prop":
        return AnswerPlan(
            execution_modes=[ExecutionMode.NATIVE_SIM],
            output_packages=[OutputPackage.BET_CARD, OutputPackage.KEY_FACTORS],
            simulation_required=True,
            betting_recommendations_included=True,
            quality_thresholds={OutputPackage.BET_CARD.value: 0.7},
        )
    return AnswerPlan(
        execution_modes=[ExecutionMode.NATIVE_SIM],
        output_packages=[OutputPackage.BET_CARD, OutputPackage.GAME_BREAKDOWN, OutputPackage.KEY_FACTORS],
        simulation_required=True,
        betting_recommendations_included=True,
        quality_thresholds={
            OutputPackage.BET_CARD.value: 0.7,
            OutputPackage.GAME_BREAKDOWN.value: 0.5,
        },
    )


def _coerce_request(request):
    """Accept a typed request OR a dict and return the typed form."""
    if isinstance(request, (GameAnalysisRequest, PlayerPropRequest, SlateAnalysisRequest)):
        return request
    if not isinstance(request, dict):
        raise TypeError(f"Expected dict or request object, got {type(request).__name__}")

    # Pick the schema by which keys are present
    if "player_name" in request and "prop_type" in request:
        return PlayerPropRequest(**request)
    if "games" in request or ("league" in request and "home_team" not in request and "player_name" not in request):
        return SlateAnalysisRequest(**request)
    return GameAnalysisRequest(**request)


def _safe_dump(obj) -> Dict[str, Any]:
    """Convert a Pydantic model (or dict) to a plain dict for JSON output."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
