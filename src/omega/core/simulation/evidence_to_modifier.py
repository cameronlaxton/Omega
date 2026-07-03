"""Deterministic mapping from EvidenceSignal types to Markov transition modifiers.

Design rules:
- This module contains a strict lookup dict. The LLM decides WHICH signal types
  to emit; this module decides the SCALAR effect of each signal on the Markov
  transition matrix. No LLM inference occurs here.
- Unknown signal types are silently ignored (they remain persisted in the trace
  for audit but do not touch the simulation).
- Multiple signals targeting the same modifier key are multiplied together and
  then clamped so that no single attribute can shift more than ±MAX_CUMULATIVE_SHIFT.
- Directional signals (direction="home" vs direction="away") are applied to
  the appropriate side's modifier key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omega.core.simulation.evidence_aggregation import (
    FamilyMember,
    damp_family,
    reliability_adjusted_factor,
    resolve_confidence,
)

if TYPE_CHECKING:
    from omega.core.calibration.adjustment_policy import AdjustmentPolicy
    from omega.core.contracts.evidence import EvidenceSignal

_log = logging.getLogger(__name__)

# Maximum cumulative multiplicative shift allowed for any single modifier key.
# 1.15 means the engine can boost at most +15% or suppress at most ~13% (-1/1.15)
# regardless of how many overlapping signals pile up.
_MAX_CUMULATIVE_SHIFT: float = 1.15

# Strict signal_type → (modifier_key, scalar) mapping.
# All keys must be members of SIGNAL_REGISTRY in omega.core.contracts.evidence.
# Validated at import time via _validate_registry_membership().
_SIGNAL_TO_MODIFIER: dict[str, tuple[str, float]] = {
    # Pace signals
    "pace_up": ("pace_scalar", 1.06),
    "pace_down": ("pace_scalar", 0.92),
    # Rest / fatigue
    "rest_advantage": ("home_score_rate_scalar", 1.04),
    "b2b_fatigue": ("home_score_rate_scalar", 0.94),
    # Matchup quality signals (applied to the defensive opponent's concession rate)
    "def_matchup_weak": ("away_score_rate_scalar", 1.05),
    "def_matchup_strong": ("away_score_rate_scalar", 0.95),
    # Role / injury-driven usage change
    "usage_role_change": ("home_score_rate_scalar", 0.93),
    # Blowout risk suppresses momentum variance
    "blowout_risk": ("home_momentum_scalar", 0.98),
}

# Exported for introspection (e.g. champion/challenger reporting).
MAPPED_SIGNAL_TYPES: frozenset[str] = frozenset(_SIGNAL_TO_MODIFIER)

# Markov-plane probation. These signal types map to mechanisms that were dead
# code until 2026-07-02 (``pace_scalar`` was produced here but consumed by
# nothing in MarkovSimulator; the momentum-scalar branches were unreachable
# because the alternating possession loop overwrote the cross-team momentum
# marker every possession). The mechanisms are now wired, but no CLV evidence
# exists for their Markov-plane effect, so they are recorded/scored and NOT
# applied to the transition matrix. Graduation is operator-gated through the
# existing issue-28 channel: an explicit ``signal_lifecycle`` override of
# "active" in the production AdjustmentPolicy lifts a type out of this set.
# The declared registry lifecycle stays "active" because the same signals have
# working (and long-applied) fast_score plane handlers — this probation is
# scoped to the Markov transition path only.
MARKOV_PLANE_PROBATION_SIGNALS: frozenset[str] = frozenset(
    {"pace_up", "pace_down", "blowout_risk"}
)

# ---------------------------------------------------------------------------
# Public vocabulary table — single source of truth for prompt generation.
# ---------------------------------------------------------------------------
# Each entry: signal_type -> (modifier_key, scalar, plain-English description)
# Used by omega_markov_evidence_guide() in the MCP server and by the cowork
# prompt section so the LLM's vocabulary is always in sync with this dict.
MARKOV_SIGNAL_VOCABULARY: tuple[tuple[str, str, float, str], ...] = (
    ("pace_up", "pace_scalar", 1.06, "+6% total possessions; matchup faster than baseline"),
    ("pace_down", "pace_scalar", 0.92, "-8% total possessions; matchup slower than baseline"),
    (
        "rest_advantage",
        "home_score_rate_scalar",
        1.04,
        "+4% home scoring rate; directional (home/away)",
    ),
    (
        "b2b_fatigue",
        "home_score_rate_scalar",
        0.94,
        "-6% scoring rate for the fatigued team; directional",
    ),
    (
        "def_matchup_weak",
        "away_score_rate_scalar",
        1.05,
        "+5% offensive scoring vs. weak defender; directional",
    ),
    (
        "def_matchup_strong",
        "away_score_rate_scalar",
        0.95,
        "-5% offensive scoring vs. strong defender; directional",
    ),
    (
        "usage_role_change",
        "home_score_rate_scalar",
        0.93,
        "-7% team scoring rate when key player role is restricted; directional",
    ),
    (
        "blowout_risk",
        "home_momentum_scalar",
        0.98,
        "-2% hot-hand follow-up scoring; suppresses runaway variance",
    ),
)


def build_markov_vocabulary_table(lifecycle_overrides: dict[str, str] | None = None) -> str:
    """Return a formatted text table of Markov-eligible signal types for prompt injection.

    Called by the MCP server's omega_markov_evidence_guide prompt and by
    OMEGA_COWORK.md section generation so the LLM vocabulary is always derived
    from the single source of truth here, never hand-edited in prompts.

    Lifecycle filtering (issue #28 WS3): ``deprecated``/``rejected`` signals drop
    out entirely so the agent stops emitting them; ``probation`` signals stay but
    are flagged "scored, not applied". ``lifecycle_overrides`` is the operator-
    approved ``AdjustmentPolicy.signal_lifecycle`` map (None = declared defaults).
    """
    from omega.core.contracts.evidence import (  # noqa: PLC0415
        effective_lifecycle,
        is_vocabulary_visible,
    )

    lines = [
        "Markov-eligible signal types (simulation_backend='markov_state' only):",
        "",
        f"  {'signal_type':<24} {'modifier':<26} {'scalar':>6}  description",
        f"  {'-' * 24} {'-' * 26} {'-' * 6}  {'-' * 45}",
    ]
    for sig, mod, scalar, desc in MARKOV_SIGNAL_VOCABULARY:
        lifecycle = effective_lifecycle(sig, lifecycle_overrides)
        if not is_vocabulary_visible(lifecycle):
            continue  # deprecated / rejected — dropped from the agent vocabulary
        if lifecycle != "active":
            tag = "   [probation: scored, NOT applied]"
        elif sig in MARKOV_PLANE_PROBATION_SIGNALS and not (
            (lifecycle_overrides or {}).get(sig) == "active"
        ):
            tag = "   [markov probation: scored, NOT applied on this backend]"
        else:
            tag = ""
        lines.append(f"  {sig:<24} {mod:<26} {scalar:>6.2f}  {desc}{tag}")
    lines += [
        "",
        "Rules:",
        "  - All other signal_types are valid for audit/fast_score paths but have NO",
        "    effect on the Markov transition matrix (silently ignored, still persisted).",
        "  - Cumulative cap: no single modifier attribute can shift by more than +/-15%",
        "    regardless of how many overlapping signals are stacked.",
        "  - Use direction='home' or direction='away' for rest_advantage, b2b_fatigue,",
        "    def_matchup_weak/strong, and usage_role_change to target the correct team.",
        "  - [probation] signals: keep emitting them (CLV needs the data) but the engine",
        "    will NOT apply them to predictions until an operator graduates them.",
    ]
    return "\n".join(lines)


def _validate_registry_membership() -> None:
    """Fail loudly at import time if any key is absent from SIGNAL_REGISTRY."""
    from omega.core.contracts.evidence import SIGNAL_REGISTRY  # noqa: PLC0415

    unknown = MAPPED_SIGNAL_TYPES - frozenset(SIGNAL_REGISTRY)
    if unknown:
        raise ImportError(f"evidence_to_modifier: keys not in SIGNAL_REGISTRY: {sorted(unknown)}")


_validate_registry_membership()


@dataclass(frozen=True)
class TransitionModifierAdjustment:
    """Rich result of mapping evidence signals to Markov transition modifiers.

    ``modifiers`` is the clamped per-key scalar dict the backend consumes — it is
    bit-identical to the legacy ``signals_to_transition_modifiers`` output when
    the policy flags are off. ``applications`` carries one *real* per-signal
    record (replacing the service layer's fabricated ``factor=None`` rows).
    ``aggregation_records`` describes the per-modifier-key cumulative math so a
    clamped or damped key effect is never misattributed to a single signal.
    """

    modifiers: dict[str, float]
    applications: list[dict[str, Any]]
    aggregation_records: list[dict[str, Any]]


def compute_transition_modifier_adjustment(
    signals: list[EvidenceSignal],
    home_team: str,
    *,
    policy: AdjustmentPolicy | None = None,
) -> TransitionModifierAdjustment:
    """Map EvidenceSignals to Markov transition modifiers with full attribution.

    The modifier math is the legacy one (per-key product, clamped to
    ±MAX_CUMULATIVE_SHIFT) unless a policy enables the Issue #22 guardrails:

    - ``enable_confidence_weighting`` scales each signal's scalar deviation from
      1.0 by the agent's confidence before aggregation.
    - ``enable_correlation_damping`` collapses the multiple signals on one
      modifier key with the sign-preserving family damper instead of a raw
      product (the modifier key is the natural correlation group here).

    Graduated-apply (default posture): when a ``policy`` is supplied, each
    signal's scalar is additionally damped toward the 1.0 no-op by its per-signal
    reliability — its fitted ``reliability_weight`` if the fit has measured one,
    else the policy's ``unfitted_reliability_prior``. Fit-proved noise (weight 0)
    collapses to a no-op; an unscored signal moves the matrix only a sliver. With
    **no** policy (the legacy ``signals_to_transition_modifiers`` wrapper) the
    reliability is 1.0, so that output stays bit-identical to the legacy mapping.
    Directional signals (home/away) resolve to the correct side.
    """
    enable_confidence = bool(getattr(policy, "enable_confidence_weighting", False))
    enable_damping = bool(getattr(policy, "enable_correlation_damping", False))
    damping_weight = float(getattr(policy, "correlation_damping_weight", 0.5))

    applications: list[dict[str, Any]] = []
    key_contribs: dict[str, list[tuple[int, float]]] = {}

    from omega.core.contracts.evidence import (  # noqa: PLC0415
        effective_lifecycle,
        is_applicable_lifecycle,
    )

    overrides = getattr(policy, "signal_lifecycle", None)

    for idx, sig in enumerate(signals):
        signal_type = str(getattr(sig, "signal_type", ""))
        # Issue #28 WS3 lifecycle gate, mirroring the handler path: a non-active
        # signal is recorded but never applied to the transition matrix.
        if not is_applicable_lifecycle(effective_lifecycle(signal_type, overrides)):
            confidence, confidence_defaulted = resolve_confidence(getattr(sig, "confidence", None))
            applications.append(
                {
                    "signal_type": signal_type,
                    "target": "skip",
                    "applied": False,
                    "factor": 1.0,
                    "effective_scalar": 1.0,
                    "confidence": confidence,
                    "confidence_defaulted": confidence_defaulted,
                    "reason": f"lifecycle={effective_lifecycle(signal_type, overrides)}: "
                    "not applied to Markov transition",
                    "policy_version": "markov_state_v1",
                    "evidence_mode": "markov_transition",
                }
            )
            continue
        # Markov-plane probation: mechanism newly wired, no CLV evidence yet.
        # Recorded (CLV scoring needs the data) but not applied, unless an
        # operator override explicitly graduates the type to "active".
        if (
            signal_type in MARKOV_PLANE_PROBATION_SIGNALS
            and not (overrides or {}).get(signal_type) == "active"
        ):
            confidence, confidence_defaulted = resolve_confidence(getattr(sig, "confidence", None))
            applications.append(
                {
                    "signal_type": signal_type,
                    "target": "skip",
                    "applied": False,
                    "factor": 1.0,
                    "effective_scalar": 1.0,
                    "confidence": confidence,
                    "confidence_defaulted": confidence_defaulted,
                    "reason": "markov_plane_probation: mechanism newly wired, not applied "
                    "until an operator graduates it (signal_lifecycle override 'active')",
                    "policy_version": "markov_state_v1",
                    "evidence_mode": "markov_transition",
                }
            )
            continue
        entry = _SIGNAL_TO_MODIFIER.get(signal_type)
        if entry is None:
            # Carry the no-op enrichment markers so a trace whose signals are all
            # unmapped is still recognized as enriched-pipeline output by the
            # qualitative-feedback gate (not misclassified as insufficient).
            confidence, confidence_defaulted = resolve_confidence(getattr(sig, "confidence", None))
            applications.append(
                {
                    "signal_type": signal_type,
                    "target": "skip",
                    "applied": False,
                    "factor": 1.0,
                    "effective_scalar": 1.0,
                    "confidence": confidence,
                    "confidence_defaulted": confidence_defaulted,
                    "reason": "no Markov transition mapping for signal_type",
                    "policy_version": "markov_state_v1",
                    "evidence_mode": "markov_transition",
                }
            )
            continue

        modifier_key, base_scalar = entry
        direction = getattr(sig, "direction", None)
        effective_key, effective_scalar = modifier_key, base_scalar
        if direction in ("home", "away"):
            effective_key, effective_scalar = _resolve_direction(
                modifier_key, base_scalar, direction
            )
        raw_scalar = effective_scalar
        confidence, confidence_defaulted = resolve_confidence(getattr(sig, "confidence", None))
        if enable_confidence:
            effective_scalar = 1.0 + confidence * (raw_scalar - 1.0)
        # Graduated-apply: damp each signal's scalar toward the 1.0 no-op by its
        # reliability (fitted weight, else the policy's unfitted prior), mirroring
        # the plane handler path. No policy => reliability 1.0 (legacy parity).
        reliability = _signal_reliability(policy, signal_type)
        effective_scalar = reliability_adjusted_factor(effective_scalar, reliability)
        # A factor that reliability-damps to exactly 1.0 (e.g. reliability_weight
        # 0.0) is a mathematical no-op: mirror the plane handler and record it as
        # not applied, so the audit trail and aggregation metadata never claim a
        # signal moved the transition matrix when it did not.
        applied = effective_scalar != 1.0

        if applied:
            key_contribs.setdefault(effective_key, []).append((idx, effective_scalar))
        applications.append(
            {
                "signal_type": signal_type,
                "target": "markov_transition",
                "modifier_key": effective_key,
                "applied": applied,
                "raw_scalar": raw_scalar,
                "effective_scalar": effective_scalar,
                "factor": effective_scalar,  # real factor, replacing the legacy None
                "reliability_weight": reliability,
                "direction": direction,
                "confidence": confidence,
                "confidence_defaulted": confidence_defaulted,
                "reason": "mapped_to_markov_transition_modifiers",
                "policy_version": "markov_state_v1",
                "evidence_mode": "markov_transition",
            }
        )

    modifiers, aggregation_records = _aggregate_modifier_keys(
        key_contribs, enable_damping=enable_damping, damping_weight=damping_weight
    )
    return TransitionModifierAdjustment(
        modifiers=modifiers,
        applications=applications,
        aggregation_records=aggregation_records,
    )


def _signal_reliability(policy: AdjustmentPolicy | None, signal_type: str) -> float:
    """Per-signal trust for the Markov path, in [0, 1].

    Mirrors the plane handler: a signal's fitted ``reliability_weight`` if
    ``omega-fit-adjustment-policy`` has measured one, else the policy's
    ``unfitted_reliability_prior``. No policy => 1.0 (legacy parity for the
    ``signals_to_transition_modifiers`` wrapper).
    """
    if policy is None:
        return 1.0
    weight = policy.coeffs_for(signal_type).get("reliability_weight")
    if weight is None:
        weight = policy.unfitted_reliability_prior
    try:
        return max(0.0, min(1.0, float(weight)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(policy.unfitted_reliability_prior)))


def _aggregate_modifier_keys(
    key_contribs: dict[str, list[tuple[int, float]]],
    *,
    enable_damping: bool,
    damping_weight: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Collapse per-key contributions into clamped modifiers + aggregation records.

    Each key's contributions are combined by a raw product (legacy) or, when
    correlation damping is enabled and two or more signals share the key, by the
    sign-preserving family damper. The result is clamped to
    [1/MAX_CUMULATIVE_SHIFT, MAX_CUMULATIVE_SHIFT].
    """
    lo = 1.0 / _MAX_CUMULATIVE_SHIFT
    hi = _MAX_CUMULATIVE_SHIFT
    modifiers: dict[str, float] = {}
    aggregation_records: list[dict[str, Any]] = []
    for key, contribs in key_contribs.items():
        damped = enable_damping and len(contribs) > 1
        if damped:
            members = [FamilyMember(key=i, factor=s) for i, s in contribs]
            raw_value = damp_family(members, damping_weight).family_damped_factor
        else:
            raw_value = 1.0
            for _, scalar in contribs:
                raw_value *= scalar
        clamped = max(lo, min(hi, raw_value))
        if clamped != raw_value:
            _log.warning(
                "cumulative modifier %r=%r clamped to %r (cap ±%.0f%%)",
                key,
                raw_value,
                clamped,
                (_MAX_CUMULATIVE_SHIFT - 1) * 100,
            )
        modifiers[key] = clamped
        aggregation_records.append(
            {
                "modifier_key": key,
                "family_size": len(contribs),
                "damped": damped,
                "raw_value": raw_value,
                "clamped_value": clamped,
                "cap": _MAX_CUMULATIVE_SHIFT,
            }
        )
    return modifiers, aggregation_records


def signals_to_transition_modifiers(
    signals: list[EvidenceSignal],
    home_team: str,
) -> dict[str, float]:
    """Map a list of EvidenceSignals to a Markov transition_modifiers dict.

    Thin backward-compatible wrapper over
    :func:`compute_transition_modifier_adjustment` returning only the modifier
    dict. Bit-identical to the legacy implementation: multiple signals on one key
    compound (product) and each key is clamped to ±MAX_CUMULATIVE_SHIFT;
    directional signals resolve home/away; unmapped types are skipped.
    """
    if not signals:
        return {}
    return compute_transition_modifier_adjustment(signals, home_team).modifiers


def _resolve_direction(modifier_key: str, scalar: float, direction: str) -> tuple[str, float]:
    """Swap home/away modifier key when the signal targets the non-default side.

    The default mapping table is written from the home team's perspective.
    When direction="away", we flip the key to target the other team.
    """
    if direction == "away":
        if modifier_key == "home_score_rate_scalar":
            return "away_score_rate_scalar", scalar
        if modifier_key == "away_score_rate_scalar":
            return "home_score_rate_scalar", scalar
        if modifier_key == "home_momentum_scalar":
            return "away_momentum_scalar", scalar
    # For pace_scalar and unknown keys: no directional flip needed.
    return modifier_key, scalar
