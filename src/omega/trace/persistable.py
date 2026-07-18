"""Canonical adapter for traces that are ready for TraceStore persistence."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from omega.core.contracts.schemas import (
    EventIdentityV1,
    coerce_engine_auto_ledger_mode,
    coerce_presentation_mode,
)
from omega.trace.eligibility import (
    evidence_learning_eligibility,
    probability_calibration_eligibility,
)


def _validated_event_identity(value: Any) -> dict[str, Any] | None:
    """Fail-closed event identity: anything that doesn't validate becomes None.

    A malformed identity must not merge unrelated traces, so it is dropped (the
    trace stays separately visible with an identity warning) rather than kept
    as unvalidated payload.
    """
    if not isinstance(value, dict):
        return None
    try:
        return EventIdentityV1(**value).model_dump(mode="json")
    except Exception:  # noqa: BLE001 - fail closed on any validation error
        return None


class PersistableTrace(BaseModel):
    """Versioned trace shape accepted by the SQLite trace ledger.

    This model is the named contract between `analyze()` trace envelopes and
    `TraceStore.persist()`. It keeps the raw request/result snapshots while
    also carrying denormalized columns used by trace queries.
    """

    model_config = ConfigDict(extra="allow")

    # v2 (Matchup Intelligence Phase 0): additive presentation_mode /
    # engine_auto_ledger_mode / event_identity / decision_support_presentation
    # fields. v1 payloads remain readable — every new field has a fail-closed
    # default, so a v1 dict validates as the restrictive decision_support /
    # disabled configuration.
    schema_version: int = 2
    trace_id: str
    run_id: str
    timestamp: str
    kind: Literal["game", "prop", "slate", "unknown"] = "unknown"
    session_id: str | None = None
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    context_labels: dict[str, Any] = Field(default_factory=dict)
    calibration_audit: list[dict[str, Any]] = Field(default_factory=list)
    downgrades: list[Any] = Field(default_factory=list)

    # Structured-evidence application (Phase 6i). evidence_application is aligned
    # by index with input_snapshot.evidence; TraceStore.persist() explodes both
    # into the evidence_signals table.
    evidence_mode: str | None = None
    evidence_application: list[dict[str, Any]] = Field(default_factory=list)
    # Per-key/per-family aggregate math (Issue #22). Stored in the full_trace JSON
    # blob only — no DB column — so grouped effects stay auditable for the
    # qualitative feedback report without a schema migration.
    evidence_aggregation: list[dict[str, Any]] = Field(default_factory=list)
    simulation_distributions: list[dict[str, Any]] = Field(default_factory=list)

    prompt: str = ""
    league: str | None = None
    matchup: str = ""
    execution_mode: str = "sandbox_unknown"
    simulation_seed: int | None = None
    aggregate_quality: float | None = None
    predictions: dict[str, Any] | list[Any] | None = None
    recommendations: dict[str, Any] | list[Any] | None = None
    odds_snapshot: dict[str, Any] | None = None
    model_version: str | None = None
    bankroll: float | None = None
    trace_quality: dict[str, Any] = Field(default_factory=dict)
    quality_gate: dict[str, Any] | None = (
        None  # backward-compat read alias; never written by new code
    )

    # LLM reasoning fields — populated by the agent orchestrator before filing the trace.
    reasoning_narrative: str | None = None
    reasoning_inputs: dict[str, Any] | None = None
    reasoning_downgrade_rationale: str | None = None
    # Analyst-note prose (thesis/market_read/why/risks/verdict) — qualitative only, no
    # protected values. Rides the full_trace JSON blob; surfaced into the saved session card.
    reasoning_presentation: dict[str, Any] | None = None

    # Matchup Intelligence Phase 0 (schema v2, all additive; ride full_trace JSON).
    # presentation_mode frames how authorized values are shown; it never widens
    # what output_mode authorizes. engine_auto_ledger_mode is the per-run
    # autolog authority consumed by omega.trace.autolog_policy.
    presentation_mode: str = "decision_support"
    engine_auto_ledger_mode: str = "disabled"
    # EventIdentityV1 dict (provider-anchored); None for legacy/unknown identity.
    event_identity: dict[str, Any] | None = None
    # DecisionSupportPresentationV1 dict — the primary presentation contract.
    decision_support_presentation: dict[str, Any] | None = None

    @classmethod
    def from_analyze_output(cls, analyze_out: dict[str, Any]) -> PersistableTrace:
        """Build a persistable trace from the canonical service `analyze()` output."""
        trace_id = str(analyze_out.get("trace_id", ""))
        # Fall back to the legacy top-level `timestamp` key for pre-Phase-6h
        # direct-analyze() exports that predate the ran_at/analyzed_at fields
        # (see docs/bugs/ export-wrapper timestamp gap).
        ran_at = str(
            analyze_out.get("ran_at")
            or analyze_out.get("analyzed_at")
            or analyze_out.get("timestamp")
            or ""
        )
        kind = str(analyze_out.get("kind", "unknown"))
        if kind not in {"game", "prop", "slate"}:
            kind = "unknown"

        input_snap = analyze_out.get("input_snapshot") or {}
        result = analyze_out.get("result") or {}
        # Prefer new trace_quality key; fall back to legacy quality_gate for old traces.
        gate = analyze_out.get("trace_quality") or analyze_out.get("quality_gate") or {}
        league = input_snap.get("league") or result.get("league") or None
        matchup = _derive_matchup(kind, input_snap, result)
        downgrades = gate.get("downgrades") or analyze_out.get("downgrades") or []

        return cls(
            trace_id=trace_id,
            run_id=str(analyze_out.get("run_id") or trace_id),
            timestamp=ran_at,
            kind=kind,  # type: ignore[arg-type]
            session_id=analyze_out.get("session_id"),
            input_snapshot=input_snap,
            result=result,
            context_labels=analyze_out.get("context_labels") or {},
            calibration_audit=_extract_calibration_audit(result),
            downgrades=downgrades,
            evidence_mode=analyze_out.get("evidence_mode"),
            evidence_application=analyze_out.get("evidence_application") or [],
            evidence_aggregation=analyze_out.get("evidence_aggregation") or [],
            simulation_distributions=(result.get("simulation_distributions") or []),
            prompt=_derive_prompt(kind, input_snap, str(league or ""), matchup),
            league=league,
            matchup=matchup,
            execution_mode=f"sandbox_{kind}",
            simulation_seed=input_snap.get("seed"),
            aggregate_quality=gate.get("aggregate_quality"),
            predictions=_derive_predictions(kind, result),
            recommendations=result.get("edges")
            or result.get("best_bet")
            or _prop_recommendation(result),
            odds_snapshot=input_snap.get("odds") or _prop_odds_snapshot(input_snap),
            model_version=analyze_out.get("model_version"),
            bankroll=analyze_out.get("bankroll"),
            trace_quality=gate,
            reasoning_narrative=analyze_out.get("reasoning_narrative"),
            reasoning_inputs=analyze_out.get("reasoning_inputs"),
            reasoning_downgrade_rationale=analyze_out.get("reasoning_downgrade_rationale"),
            reasoning_presentation=analyze_out.get("reasoning_presentation"),
            presentation_mode=coerce_presentation_mode(analyze_out.get("presentation_mode")),
            engine_auto_ledger_mode=coerce_engine_auto_ledger_mode(
                analyze_out.get("engine_auto_ledger_mode")
            ),
            event_identity=_validated_event_identity(analyze_out.get("event_identity")),
            decision_support_presentation=analyze_out.get("decision_support_presentation"),
        )

    def calibration_eligibility(self) -> dict[str, bool]:
        """Diagnostic report on which calibration paths this trace is eligible for.

        Not an enforcement gate — callers decide whether to act on the result.
        Falls back to quality_gate when trace_quality is empty (pre-rename traces).
        Delegates to omega.trace.eligibility (the single source of truth).
        """
        tq = self.trace_quality or self.quality_gate or {}
        identity_status = tq.get("identity_status")
        prob = probability_calibration_eligibility(predictions=self.predictions, trace_quality=tq)
        evidence = evidence_learning_eligibility(trace_quality=tq)
        return {
            "probability_calibration": prob.eligible,
            "evidence_scoring": evidence.eligible,
            "context_slice_fitting": identity_status not in ("missing", "backfilled"),
        }

    def to_store_record(self) -> dict[str, Any]:
        """Return the dict shape consumed by `TraceStore.persist()`."""
        return self.model_dump(mode="json")


def _derive_matchup(kind: str, input_snap: dict[str, Any], result: dict[str, Any]) -> str:
    if kind == "game":
        home = input_snap.get("home_team") or ""
        away = input_snap.get("away_team") or ""
        if home and away:
            return f"{away} @ {home}"
    if kind == "prop":
        home = input_snap.get("home_team") or ""
        away = input_snap.get("away_team") or ""
        if home and away:
            return f"{away} @ {home}"
        player = input_snap.get("player_name") or ""
        prop = input_snap.get("prop_type") or ""
        line = input_snap.get("line")
        if player and prop:
            line_str = f" {line}" if line is not None else ""
            return f"{player} {prop}{line_str}"
    return result.get("matchup", "") or ""


def _derive_prompt(kind: str, input_snap: dict[str, Any], league: str, matchup: str) -> str:
    base = f"{league} {kind}: {matchup}".strip()
    return base or json.dumps(input_snap, default=str)[:200]


def _derive_predictions(kind: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if kind == "game":
        return result.get("simulation")
    prop_predictions = {
        k: result.get(k) for k in ("over_prob", "under_prob") if result.get(k) is not None
    }
    return prop_predictions or None


def _prop_recommendation(result: dict[str, Any]) -> dict[str, Any] | None:
    recommendation = result.get("recommendation")
    if recommendation is None:
        return None
    return {
        "recommendation": recommendation,
        "confidence_tier": result.get("confidence_tier"),
        "recommended_units": result.get("recommended_units"),
        "kelly_fraction": result.get("kelly_fraction"),
        "bet_side_odds": result.get("bet_side_odds"),
    }


def _prop_odds_snapshot(input_snap: dict[str, Any]) -> dict[str, Any] | None:
    over = input_snap.get("odds_over")
    under = input_snap.get("odds_under")
    if over is None and under is None:
        return None
    return {"odds_over": over, "odds_under": under}


def _extract_calibration_audit(result: dict[str, Any]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for edge in result.get("edges") or []:
        audit = edge.get("calibration_audit") if isinstance(edge, dict) else None
        if isinstance(audit, dict):
            audits.append(audit)
    for key in ("over_calibration_audit", "under_calibration_audit"):
        audit = result.get(key)
        if isinstance(audit, dict):
            audits.append(audit)
    return audits
