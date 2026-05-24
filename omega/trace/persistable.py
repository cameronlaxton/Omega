"""Canonical adapter for traces that are ready for TraceStore persistence."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PersistableTrace(BaseModel):
    """Versioned trace shape accepted by the SQLite trace ledger.

    This model is the named contract between `analyze()` trace envelopes and
    `TraceStore.persist()`. It keeps the raw request/result snapshots while
    also carrying denormalized columns used by trace queries.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
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
    quality_gate: dict[str, Any] | None = None  # backward-compat read alias; never written by new code

    # LLM reasoning fields — populated by the agent orchestrator before filing the trace.
    reasoning_narrative: str | None = None
    reasoning_inputs: dict[str, Any] | None = None
    reasoning_downgrade_rationale: str | None = None

    @classmethod
    def from_analyze_output(cls, analyze_out: dict[str, Any]) -> PersistableTrace:
        """Build a persistable trace from the canonical service `analyze()` output."""
        trace_id = str(analyze_out.get("trace_id", ""))
        ran_at = str(analyze_out.get("ran_at") or analyze_out.get("analyzed_at") or "")
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
        )

    def calibration_eligibility(self) -> dict[str, bool]:
        """Diagnostic report on which calibration paths this trace is eligible for.

        Not an enforcement gate — callers decide whether to act on the result.
        Falls back to quality_gate when trace_quality is empty (pre-rename traces).
        """
        tq = self.trace_quality or self.quality_gate or {}
        identity_status = tq.get("identity_status")
        context_source = tq.get("context_source")
        calibration_eligible = bool(tq.get("calibration_eligible"))
        return {
            "probability_calibration": (
                self.predictions is not None
                and calibration_eligible
                and context_source == "provided"
                and identity_status == "complete"
            ),
            "evidence_scoring": tq.get("evidence_status") == "present",
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
