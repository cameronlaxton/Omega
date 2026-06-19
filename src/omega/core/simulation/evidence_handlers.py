"""
Deterministic evidence handlers — translate structured EvidenceSignals into
capped, attributable simulation-input adjustments.

This module is the deterministic half of the structured-reasoning loop. The LLM
supplies evidence *values*; the handlers here decide *how* the engine applies
them, using coefficients from a versioned ``AdjustmentPolicy``. Every handler is
a pure function (no RNG, no clock, no I/O), so the same signal + policy always
yields the same factor — a hard requirement for reproducible traces.

Key properties:
  - Unknown ``signal_type`` -> no handler -> recorded as skipped, never applied.
  - A signal is gated on sport (``signal_applies``) and on plane (player vs game).
  - Every factor is capped to ``1 +/- coeffs['cap']`` so one signal can never
    swing a mean unboundedly.
  - Shadow mode computes and records factors but the effective mean/std factor
    returned to the engine is 1.0 — predictions are unchanged until a policy is
    promoted to ``mode='live'``.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from omega.core.calibration.adjustment_policy import AdjustmentPolicy
from omega.core.contracts.evidence import (
    EvidenceSignal,
    resolve_archetype,
    signal_applies,
)
from omega.core.simulation.evidence_aggregation import cap_factor as _cap_factor

# Environment override for the rollout gate. When unset, the policy's own
# ``mode`` field governs. Accepts "shadow" or "live" (case-insensitive).
_ENV_MODE_VAR = "OMEGA_EVIDENCE_MODE"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjustmentRecord:
    """Per-signal record of what the engine did (or would do) with one signal.

    ``factor`` is always the computed, capped factor — even in shadow mode — so
    retrospective scoring can backtest counterfactually. ``applied`` is True only
    when the engine actually multiplied the factor into the live prediction.
    """

    signal_type: str
    target: str  # "mean" | "std" | "skip"
    factor: float
    applied: bool
    reason: str
    policy_version: str
    evidence_mode: str

    def as_application(self) -> dict[str, Any]:
        """Serialize to the per-signal dict the trace store persists (V9)."""
        return {
            "signal_type": self.signal_type,
            "target": self.target,
            "applied": self.applied,
            "factor": self.factor,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "evidence_mode": self.evidence_mode,
        }


@dataclass
class PlaneAdjustment:
    """Aggregate result of evaluating every signal for one analysis.

    Player plane uses ``mean_factor`` / ``std_factor`` (the team-factor fields
    stay 1.0). Game plane uses ``home_factor`` / ``away_factor`` (the mean/std
    fields stay 1.0). All factors are effective — 1.0 in shadow mode.
    """

    mean_factor: float = 1.0
    std_factor: float = 1.0
    home_factor: float = 1.0
    away_factor: float = 1.0
    records: list[AdjustmentRecord] = field(default_factory=list)
    evidence_mode: str = "shadow"

    def applications(self) -> list[dict[str, Any]]:
        """Per-signal application dicts, aligned by index with the evidence list."""
        return [r.as_application() for r in self.records]


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


def resolve_evidence_mode(policy: AdjustmentPolicy | None) -> str:
    """Resolve the effective rollout mode.

    Order: ``OMEGA_EVIDENCE_MODE`` env override, else the policy's ``mode``,
    else ``shadow``. An invalid env value is ignored (falls through to policy).
    """
    env = (os.environ.get(_ENV_MODE_VAR) or "").strip().lower()
    if env in ("shadow", "live"):
        return env
    if policy is not None and policy.mode in ("shadow", "live"):
        return policy.mode
    return "shadow"


# ---------------------------------------------------------------------------
# Cap helper
# ---------------------------------------------------------------------------


# ``_cap_factor`` is the canonical ``evidence_aggregation.cap_factor`` (imported
# above). Re-exported under the legacy private name so existing call sites and
# tests keep working while the cap semantics live in exactly one place.


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Handlers — pure (EvidenceSignal, coeffs, baseline) -> (target, raw_factor)
# ---------------------------------------------------------------------------

# A handler returns (target, raw_factor). The orchestrator applies the cap.
Handler = Callable[[EvidenceSignal, dict[str, Any], float], "tuple[str, float]"]


def _h_flat_mult(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Flat multiplier triggered when the signal value is truthy."""
    trigger = signal.value
    if isinstance(trigger, (int, float, bool)) and not trigger:
        return "mean", 1.0
    return "mean", float(coeffs.get("mean_mult", 1.0))


def _h_b2b(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Back-to-back fatigue. Triggered when value is truthy.

    ``default_mult`` is pre-resolved to the per-league value by
    ``_resolve_b2b_coeffs`` before this handler runs, so the per-stat lookup is
    a simple read here.
    """
    if isinstance(signal.value, (int, float, bool)) and not signal.value:
        return "mean", 1.0
    return "mean", float(coeffs.get("default_mult", 1.0))


def _h_ratio(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Value is a ratio vs baseline (e.g. pace 1.05); scaled toward 1.0."""
    ratio = _as_float(signal.value, 1.0)
    scale = float(coeffs.get("scale", 1.0))
    return "mean", 1.0 + (ratio - 1.0) * scale


def _h_usage_spike(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Value is a fractional usage increase (0.12 = +12%)."""
    delta = _as_float(signal.value, 0.0)
    scale = float(coeffs.get("scale", 1.0))
    return "mean", 1.0 + delta * scale


def _h_per_unit(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Value is a signed magnitude; positive favors more output."""
    units = _as_float(signal.value, 0.0)
    per_unit = float(coeffs.get("per_unit", 0.0))
    return "mean", 1.0 + units * per_unit


def _h_opponent_rank(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Value is an opponent defensive rank (1=toughest .. 30=weakest)."""
    rank = _as_float(signal.value, coeffs.get("pivot_rank", 15.5))
    pivot = float(coeffs.get("pivot_rank", 15.5))
    per_rank = float(coeffs.get("per_rank", 0.0))
    # rank below pivot (tougher) suppresses; above pivot (weaker) boosts.
    return "mean", 1.0 + (rank - pivot) * per_rank


def _h_series_blend(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Blend a recent observed level toward the season baseline.

    Value may be a series (list of recent values) or a single scalar average.
    """
    if isinstance(signal.value, list) and signal.value:
        nums = [_as_float(v) for v in signal.value]
        recent = sum(nums) / len(nums) if nums else baseline
    else:
        recent = _as_float(signal.value, baseline)
    if baseline <= 0:
        return "mean", 1.0
    weight = float(coeffs.get("weight", 0.0))
    return "mean", 1.0 + weight * (recent / baseline - 1.0)


def _h_categorical(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Value is a category label resolved through a coefficient map."""
    mapping: dict[str, Any] = coeffs.get("map", {})
    label = str(signal.value)
    return "mean", float(mapping.get(label, 1.0))


def _h_std_compress(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Blowout risk compresses the distribution (lower std -> tighter floors)."""
    risk = _as_float(signal.value, 0.0)
    risk = max(0.0, min(1.0, risk))
    compression = float(coeffs.get("std_compression", 0.0))
    return "std", 1.0 - risk * compression


def _h_std_widen(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """An outlier-flagged prior game widens the distribution (more uncertainty)."""
    if isinstance(signal.value, (int, float, bool)) and not signal.value:
        return "std", 1.0
    return "std", 1.0 + float(coeffs.get("std_widen", 0.0))


def _h_audit_only(signal: EvidenceSignal, coeffs: dict[str, Any], baseline: float):
    """Audit-only signals: registered so the validator is silent; never applied."""
    return "skip", 1.0


HANDLER_REGISTRY: dict[str, Handler] = {
    # player-form
    "recent_form": _h_series_blend,
    "series_avg": _h_series_blend,
    "home_away_split": _h_series_blend,
    "last_game_outlier": _h_std_widen,
    # matchup
    "opponent_stat_rank": _h_opponent_rank,
    "def_matchup_weak": _h_flat_mult,
    "def_matchup_strong": _h_flat_mult,
    "pitcher_matchup": _h_per_unit,
    "starter_era": _h_per_unit,
    "formation_mismatch": _h_per_unit,
    "surface_edge": _h_per_unit,
    "course_fit": _h_per_unit,
    "stylistic_matchup": _h_per_unit,
    "map_pool_edge": _h_per_unit,
    # situational
    "rest_advantage": _h_per_unit,
    "elimination_game": _h_flat_mult,
    "motivation_edge": _h_per_unit,
    "blowout_risk": _h_std_compress,
    "b2b_fatigue": _h_b2b,
    "weather_wind": _h_per_unit,
    "weather_cold": _h_flat_mult,
    "dome_effect": _h_flat_mult,
    "park_factor_evidence": _h_ratio,
    # usage / pace
    "usage_spike": _h_usage_spike,
    "usage_role_change": _h_categorical,
    "pace_up": _h_ratio,
    "pace_down": _h_ratio,
    # team-form / matchup (game-plane momentum)
    "win_streak": _h_per_unit,
    "series_lead": _h_per_unit,
    # Audit-only: registered in SIGNAL_REGISTRY and policy but always return "skip"
    # so the engine records them for retrospective scoring without applying them.
    "season_record": _h_audit_only,
    "season_baseline": _h_audit_only,
    "defensive_scheme": _h_audit_only,
}


# ---------------------------------------------------------------------------
# Per-signal evaluation
# ---------------------------------------------------------------------------


def _evaluate_signal(
    signal: EvidenceSignal,
    *,
    policy: AdjustmentPolicy,
    archetype: str | None,
    baseline: float,
    evidence_mode: str,
) -> AdjustmentRecord:
    """Evaluate one signal into a capped, attributable AdjustmentRecord.

    Pure given (signal, policy, archetype, baseline, evidence_mode).
    """
    policy_version = policy.policy_id
    handler = HANDLER_REGISTRY.get(signal.signal_type)
    coeffs = policy.coeffs_for(signal.signal_type)

    if handler is None or not coeffs:
        return AdjustmentRecord(
            signal.signal_type, "skip", 1.0, False,
            "no handler or coefficients for signal_type", policy_version, evidence_mode,
        )
    if not signal_applies(signal.signal_type, archetype):
        return AdjustmentRecord(
            signal.signal_type, "skip", 1.0, False,
            f"signal does not apply to sport archetype {archetype!r}",
            policy_version, evidence_mode,
        )

    target, raw_factor = handler(signal, coeffs, baseline)
    # reliability_weight is the single seam that closes the Phase C -> Phase B
    # loop: omega-fit-adjustment-policy sets it per signal_type from
    # measured empirical accuracy, damping the handler's deviation toward the
    # 1.0 no-op for signals that scored as noise. Absent => 1.0 (full trust),
    # so the hand-seeded v1 policy behaves exactly as the raw handler.
    reliability = max(0.0, min(1.0, float(coeffs.get("reliability_weight", 1.0))))
    damped_factor = 1.0 + reliability * (raw_factor - 1.0)
    factor = _cap_factor(damped_factor, float(coeffs.get("cap", 0.0)))
    applied = evidence_mode == "live" and target != "skip" and factor != 1.0
    reason = (
        f"{signal.signal_type}: {target} x{factor:.4f} "
        f"(raw {raw_factor:.4f}, mode={evidence_mode})"
    )
    return AdjustmentRecord(
        signal.signal_type, target, factor, applied, reason, policy_version, evidence_mode
    )


def _resolve_b2b_coeffs(policy: AdjustmentPolicy, league: str) -> dict[str, Any]:
    """b2b_fatigue is per-league; pre-resolve its coefficient before evaluation."""
    coeffs = policy.coeffs_for("b2b_fatigue")
    if not coeffs:
        return coeffs
    by_league: dict[str, Any] = coeffs.get("by_league", {})
    resolved = dict(coeffs)
    resolved["default_mult"] = float(
        by_league.get(league.upper(), coeffs.get("default_mult", 1.0))
    )
    return resolved


# ---------------------------------------------------------------------------
# Plane orchestration
# ---------------------------------------------------------------------------


def compute_player_adjustment(
    *,
    player_context: dict[str, Any],
    evidence: list[EvidenceSignal],
    league: str,
    prop_type: str,
    policy: AdjustmentPolicy,
    evidence_mode: str,
) -> PlaneAdjustment:
    """Evaluate every player-plane signal for a player-prop analysis.

    Returns a PlaneAdjustment whose ``records`` are aligned by index with the
    full ``evidence`` list (game-plane signals get a ``skip`` record). The
    effective ``mean_factor`` / ``std_factor`` are 1.0 in shadow mode.
    """
    archetype = resolve_archetype(league)
    baseline = _as_float((player_context or {}).get(f"{prop_type}_mean"), 0.0)

    records: list[AdjustmentRecord] = []
    mean_factor = 1.0
    std_factor = 1.0
    for signal in evidence:
        if signal.plane != "player":
            records.append(
                AdjustmentRecord(
                    signal.signal_type, "skip", 1.0, False,
                    "game-plane signal not applied on a player-prop analysis",
                    policy.policy_id, evidence_mode,
                )
            )
            continue
        rec = _evaluate_record_for_signal(signal, policy, archetype, baseline, evidence_mode, league)
        records.append(rec)
        if rec.applied and rec.target == "mean":
            mean_factor *= rec.factor
        elif rec.applied and rec.target == "std":
            std_factor *= rec.factor

    return PlaneAdjustment(
        mean_factor=mean_factor,
        std_factor=std_factor,
        records=records,
        evidence_mode=evidence_mode,
    )


def compute_game_adjustment(
    *,
    evidence: list[EvidenceSignal],
    league: str,
    policy: AdjustmentPolicy,
    evidence_mode: str,
) -> PlaneAdjustment:
    """Evaluate every game-plane signal for a game analysis.

    Game-plane mean-target factors scale a team's ``off_rating`` — which team is
    chosen by the signal's ``direction`` (home / away / neutral-both). ``std``-
    target signals (e.g. blowout_risk) have no game-plane application target, so
    they are recorded for scoring but marked unapplied.
    """
    archetype = resolve_archetype(league)
    records: list[AdjustmentRecord] = []
    home_factor = 1.0
    away_factor = 1.0
    for signal in evidence:
        if signal.plane != "game":
            records.append(
                AdjustmentRecord(
                    signal.signal_type, "skip", 1.0, False,
                    "player-plane signal not applied on a game analysis",
                    policy.policy_id, evidence_mode,
                )
            )
            continue
        rec = _evaluate_record_for_signal(signal, policy, archetype, 0.0, evidence_mode, league)
        if rec.target == "std" and rec.applied:
            # No team-context std to scale; honestly mark it unapplied.
            rec = dataclasses.replace(
                rec,
                applied=False,
                reason="std-target signal has no game-plane application target",
            )
        records.append(rec)
        if rec.applied and rec.target == "mean":
            if signal.direction == "home":
                home_factor *= rec.factor
            elif signal.direction == "away":
                away_factor *= rec.factor
            else:  # neutral / None -> shifts the whole scoring environment
                home_factor *= rec.factor
                away_factor *= rec.factor

    return PlaneAdjustment(
        home_factor=home_factor,
        away_factor=away_factor,
        records=records,
        evidence_mode=evidence_mode,
    )


def _evaluate_record_for_signal(
    signal: EvidenceSignal,
    policy: AdjustmentPolicy,
    archetype: str | None,
    baseline: float,
    evidence_mode: str,
    league: str,
) -> AdjustmentRecord:
    """Evaluate one signal, pre-resolving per-league coefficients where needed."""
    if signal.signal_type == "b2b_fatigue":
        resolved = _resolve_b2b_coeffs(policy, league)
        # Build a shallow policy view with the resolved coefficient.
        patched = policy.model_copy(
            update={"coefficients": {**policy.coefficients, "b2b_fatigue": resolved}}
        )
        return _evaluate_signal(
            signal, policy=patched, archetype=archetype,
            baseline=baseline, evidence_mode=evidence_mode,
        )
    return _evaluate_signal(
        signal, policy=policy, archetype=archetype,
        baseline=baseline, evidence_mode=evidence_mode,
    )
