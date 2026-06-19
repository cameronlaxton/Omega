"""
Shared, pure evidence-aggregation helpers (Issue #22, Phase 1 scaffolding).

This module is a leaf: it imports only the standard library so both
``evidence_handlers.py`` (player/game plane factors) and
``evidence_to_modifier.py`` (Markov transition modifiers) can depend on it
without creating an import cycle. It holds *no* policy, RNG, clock, or I/O —
every function here is a pure deterministic transform, which is what lets the
trace stay reproducible.

It encodes two things the engine needs once confidence weighting and
correlation damping are switched on (both gated behind ``AdjustmentPolicy``
flags that default to ``False``):

1. The **strict factor sequence** from Issue #22, decomposed into one pure
   function per step so each step is independently testable and the wiring in
   later phases is a straight composition:

       1. raw_factor                          (handler output)
       2. reliability_adjusted_factor         reliability_adjusted_factor()
       3. per_signal_capped_factor            cap_factor(..., per_signal_cap)
       4. family_damped_factor                damp_family()
       5. family_capped_factor                cap_factor(..., family_cap)
       6. confidence_adjusted_factor          confidence_adjusted_factor()
       7. plane_aggregated_factor             plane_aggregate()
       8. final_applied_factor                cap_factor(..., plane_cap)

2. The **sign-preserving co-occurrence damping** of correlated signals that
   share a ``damping_family``. ``abs()`` is used only to pick the dominant
   ("primary") record; it never becomes the magnitude of the delta, so a
   family of suppressing signals stays a suppression and a family of boosting
   signals stays a boost.

Nothing in this module is wired into a live path in Phase 1 — it exists so the
behaviour-gated phases that follow have a single, tested source of truth for
the math and the enriched per-signal trace payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "FamilyRole",
    "FamilyMember",
    "FamilyDampingResult",
    "SignalApplication",
    "DEFAULT_CONFIDENCE",
    "cap_factor",
    "reliability_adjusted_factor",
    "confidence_adjusted_factor",
    "resolve_confidence",
    "per_signal_capped_factor",
    "damp_family",
    "plane_aggregate",
]

FamilyRole = Literal["primary", "secondary", "singleton"]

# Confidence assumed when a legacy persisted dict / replay reconstruction is
# missing the field. 1.0 makes confidence weighting a no-op, so replaying a
# pre-confidence trace reproduces the pre-confidence engine exactly — no silent
# inflation. The defaulting is flagged (confidence_defaulted) so the feedback
# report can treat such signals cautiously.
DEFAULT_CONFIDENCE = 1.0


# ---------------------------------------------------------------------------
# Step helpers — one pure function per stage of the strict factor sequence
# ---------------------------------------------------------------------------


def cap_factor(factor: float, cap: float) -> float:
    """Clamp ``factor`` to ``[1 - cap, 1 + cap]``. ``cap`` is a fraction (0..1).

    Canonical implementation reused by ``evidence_handlers`` (steps 3, 5 and 8
    of the factor sequence) so the cap semantics live in exactly one place. A
    non-positive cap collapses the factor to the ``1.0`` no-op, matching the
    legacy handler behaviour bit-for-bit.
    """
    if cap <= 0:
        return 1.0
    lo, hi = 1.0 - cap, 1.0 + cap
    return max(lo, min(hi, factor))


def reliability_adjusted_factor(raw_factor: float, reliability_weight: float) -> float:
    """Step 2: damp a raw handler factor toward 1.0 by the reliability weight.

    ``reliability_weight`` is the empirical-trust coefficient (0 = treat the
    signal as noise, 1 = full strength). This mirrors the existing handler
    damping exactly so moving the math here is behaviour-neutral.
    """
    return 1.0 + reliability_weight * (raw_factor - 1.0)


def per_signal_capped_factor(
    raw_factor: float, reliability_weight: float, per_signal_cap: float
) -> float:
    """Steps 2+3 composed: reliability-adjust then per-signal cap.

    Convenience for callers that do not need the intermediate value separately;
    the trace still records both via :class:`SignalApplication`.
    """
    return cap_factor(
        reliability_adjusted_factor(raw_factor, reliability_weight), per_signal_cap
    )


def confidence_adjusted_factor(factor: float, confidence: float) -> float:
    """Step 6: scale a factor's deviation from 1.0 by the agent's confidence.

    Reliability (how often this *signal type* has been right) and confidence
    (how sure the agent is about *this* instance) are deliberately separate
    coefficients applied at different stages, and stay separately traceable.
    """
    return 1.0 + confidence * (factor - 1.0)


def resolve_confidence(value: float | None) -> tuple[float, bool]:
    """Resolve a signal's confidence, flagging legacy defaulting.

    Live ``EvidenceSignal`` validation requires confidence, so the live engine
    path always passes a value and the second element is ``False``. Only legacy
    persisted dicts / replay reconstructions can be missing it; those fall back
    to :data:`DEFAULT_CONFIDENCE` (a confidence-weighting no-op) and are flagged
    so downstream feedback can treat them cautiously.
    """
    if value is None:
        return DEFAULT_CONFIDENCE, True
    return float(value), False


def plane_aggregate(factors: list[float]) -> float:
    """Step 7: multiply the confidence-adjusted factors of orthogonal families.

    Families that share a ``damping_family`` are collapsed to a single factor by
    :func:`damp_family` *before* this step, so the inputs here are assumed
    orthogonal and compounding them is correct.
    """
    product = 1.0
    for f in factors:
        product *= f
    return product


# ---------------------------------------------------------------------------
# Step 4 — sign-preserving co-occurrence damping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyMember:
    """One correlated signal contributing to a ``damping_family``.

    ``key`` is an opaque caller-chosen handle (e.g. the evidence-list index or
    the signal_type) used only to map roles back onto the caller's records.
    ``factor`` is the *per-signal-capped* factor (sequence step 3).
    """

    key: Any
    factor: float


@dataclass(frozen=True)
class FamilyDampingResult:
    """Outcome of damping one ``damping_family``.

    ``family_damped_factor`` is shared by every member of the family and flows
    into the family cap (step 5). ``roles`` assigns each member's ``key`` a
    ``family_role`` for the trace; ``primary_key`` is the dominant member.
    """

    family_damped_factor: float
    roles: dict[Any, FamilyRole]
    primary_key: Any
    family_size: int


def damp_family(members: list[FamilyMember], damping_weight: float) -> FamilyDampingResult:
    """Step 4: collapse a correlated signal family into one factor, sign-preserved.

    The dominant ("primary") member is the one whose factor deviates most from
    1.0 — ``abs()`` selects it but never becomes the delta's magnitude. The
    primary keeps its full delta; every other member contributes its *signed*
    delta scaled by ``damping_weight`` (0 = ignore correlated stackers, 1 = no
    damping). A singleton family passes its factor through unchanged.

        primary_delta    = primary.factor - 1.0
        secondary_delta  = sum(m.factor - 1.0 for non-primary) * damping_weight
        family_damped    = 1.0 + primary_delta + secondary_delta

    Raises ``ValueError`` on an empty family — callers must not form one.
    """
    if not members:
        raise ValueError("damp_family requires at least one member")
    if len(members) == 1:
        only = members[0]
        return FamilyDampingResult(
            family_damped_factor=only.factor,
            roles={only.key: "singleton"},
            primary_key=only.key,
            family_size=1,
        )

    # First record with the largest absolute deviation wins (deterministic on
    # ties because max() keeps the earliest maximal element).
    primary_idx = max(
        range(len(members)), key=lambda i: abs(members[i].factor - 1.0)
    )
    primary = members[primary_idx]
    primary_delta = primary.factor - 1.0
    secondary_delta = (
        sum(m.factor - 1.0 for i, m in enumerate(members) if i != primary_idx)
        * damping_weight
    )
    family_damped_factor = 1.0 + primary_delta + secondary_delta

    roles: dict[Any, FamilyRole] = {}
    for i, m in enumerate(members):
        roles[m.key] = "primary" if i == primary_idx else "secondary"

    return FamilyDampingResult(
        family_damped_factor=family_damped_factor,
        roles=roles,
        primary_key=primary.key,
        family_size=len(members),
    )


# ---------------------------------------------------------------------------
# Enriched per-signal trace payload (scaffolding — populated in later phases)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalApplication:
    """Full, attributable record of one signal's journey through the sequence.

    Every stage is stored separately so the trace can prove that grouped family
    math was not falsely attributed to a single signal, and so reliability and
    confidence stay independently auditable. This is the enriched payload the
    closed-loop feedback report (Phase 6) reads; in Phase 1 it is only the
    scaffold — no live path emits it yet.
    """

    signal_type: str
    target: str  # "mean" | "std" | "skip"
    applied: bool
    reason: str
    policy_version: str
    evidence_mode: str

    # Strict factor sequence (Issue #22)
    raw_factor: float
    reliability_weight: float
    reliability_adjusted_factor: float
    per_signal_capped_factor: float
    damping_family: str | None
    family_size: int
    family_role: FamilyRole
    family_damped_factor: float
    family_capped_factor: float
    confidence: float
    confidence_defaulted: bool
    confidence_adjusted_factor: float
    final_applied_factor: float

    def as_dict(self) -> dict[str, Any]:
        """Serialize to the per-signal application dict persisted in trace JSON.

        ``factor`` is kept as a stable alias of ``final_applied_factor`` because
        existing consumers (the V9 ``evidence_signals`` explode, the analyze
        envelope, replay tooling) read ``factor`` for the effective applied
        value. New fields are purely additive.
        """
        return {
            "signal_type": self.signal_type,
            "target": self.target,
            "applied": self.applied,
            "factor": self.final_applied_factor,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "evidence_mode": self.evidence_mode,
            "raw_factor": self.raw_factor,
            "reliability_weight": self.reliability_weight,
            "reliability_adjusted_factor": self.reliability_adjusted_factor,
            "per_signal_capped_factor": self.per_signal_capped_factor,
            "damping_family": self.damping_family,
            "family_size": self.family_size,
            "family_role": self.family_role,
            "family_damped_factor": self.family_damped_factor,
            "family_capped_factor": self.family_capped_factor,
            "confidence": self.confidence,
            "confidence_defaulted": self.confidence_defaulted,
            "confidence_adjusted_factor": self.confidence_adjusted_factor,
            "final_applied_factor": self.final_applied_factor,
        }


def is_finite_factor(value: float) -> bool:
    """True when ``value`` is a finite real factor (guards bad handler output)."""
    return isinstance(value, (int, float)) and math.isfinite(value)
