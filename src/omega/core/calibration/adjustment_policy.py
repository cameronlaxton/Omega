"""
Engine adjustment policy — versioned coefficients for structured evidence signals.

A calibration profile corrects probability *outputs* after simulation. An
adjustment policy corrects simulation *inputs* before the fact: it holds the
deterministic coefficients the evidence handlers
(``omega/core/simulation/evidence_handlers.py``) use to translate an
``EvidenceSignal`` into a mean/std factor.

This module deliberately mirrors ``profiles.py`` / ``registry.py``:
  - ``AdjustmentPolicy`` is versioned, attributable, and lifecycle-tracked.
  - ``AdjustmentPolicyRegistry`` is a single JSON file with a promotion workflow.
  - ``ProfileStatus`` (CANDIDATE/PRODUCTION/ARCHIVED/REJECTED) is reused.

Unlike calibration profiles, an adjustment policy is global (not per-league):
its ``coefficients`` are keyed by ``signal_type`` and the handlers apply
sport-awareness internally. At most one PRODUCTION policy exists at a time.

``mode`` gates the rollout along a graduated ladder (see :data:`EvidenceMode`):
``disabled`` < ``observe`` < ``score_only`` < ``bounded_live`` < ``live``. In the
three non-applying modes the handlers compute and record adjustments but the
engine does not move the live prediction; in ``bounded_live`` the factor is
applied under hard per-signal/family/plane caps; in ``live`` it is applied under
the policy's own caps. ``score_only`` is the safe default because the seed
coefficients are unfitted priors — they are recorded and may feed retrospective
learning, but never move a prediction until an operator flips the mode.

Storage: omega/core/calibration/adjustment_policies.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from omega.core.calibration.profiles import ProfileStatus

UTC = timezone.utc

logger = logging.getLogger("omega.core.calibration.adjustment_policy")

_DEFAULT_PATH = Path(__file__).parent / "adjustment_policies.json"
_SCHEMA_VERSION = 1

# Graduated evidence-application ladder. Each rung strictly contains the audit
# guarantees of the one before it and adds (at most) one new effect:
#   disabled     -> evidence is neither computed nor recorded.
#   observe      -> computed + recorded for audit only; no learning, no math.
#   score_only   -> computed + recorded; feeds retrospective evidence learning;
#                   never moves a prediction. (legacy ``shadow`` maps here.)
#   bounded_live -> applied to simulation inputs under HARD single/family/plane
#                   caps; cannot lift confidence to A unless the policy's
#                   evidence metrics pass promotion gates.
#   live         -> applied under the policy's own caps (reserved for a fully
#                   gated, fitted policy).
EvidenceMode = Literal["disabled", "observe", "score_only", "bounded_live", "live"]

# Backward-compatible alias. The historical binary type was
# ``Literal["shadow", "live"]``; external importers keep working, but new code
# should reference :data:`EvidenceMode`.
AdjustmentMode = EvidenceMode

# Legacy persisted/env values -> graduated mode. ``shadow`` becomes
# ``score_only`` (records + learning, no math) — the behaviour-preserving map
# confirmed for this remediation. ``live`` is unchanged.
LEGACY_MODE_MAP: dict[str, str] = {"shadow": "score_only", "live": "live"}

# The five valid graduated modes (mirrors the Literal above for runtime checks).
EVIDENCE_MODES: frozenset[str] = frozenset(
    {"disabled", "observe", "score_only", "bounded_live", "live"}
)

# Modes in which evidence actually moves the prediction math.
APPLYING_MODES: frozenset[str] = frozenset({"bounded_live", "live"})

# Hard caps enforced when mode == "bounded_live", supplied as effective
# overrides when the policy itself leaves them unset. Expressed as max absolute
# fractional deviation of a factor from 1.0.
BOUNDED_LIVE_SINGLE_CAP = 0.10
BOUNDED_LIVE_FAMILY_CAP = 0.15
BOUNDED_LIVE_PLANE_CAP = 0.20

# Minimum fitted sample size before an adjustment policy's evidence metrics are
# considered to have "passed gates" — the gate that lets bounded_live evidence
# contribute to an A-tier recommendation.
MIN_EVIDENCE_SAMPLES_FOR_GATE = 100

# Minimum mean reliability weight (from the fit) for the same gate. A policy can
# be fitted on plenty of data yet have its signals score as noise (reliability
# weight 0 => directional accuracy <= 0.50). Sample size alone must NOT unlock an
# A; the fitted evidence must also be predictive. 0.10 ~ average directional
# accuracy >= 0.55 across the fitted signal types.
MIN_EVIDENCE_RELIABILITY_FOR_GATE = 0.10


def normalize_evidence_mode(value: str | None, *, default: str = "score_only") -> str:
    """Map a raw mode string (incl. legacy ``shadow``) to a valid graduated mode.

    Unknown values fall back to ``default``. Case-insensitive.
    """
    if not value:
        return default
    v = value.strip().lower()
    v = LEGACY_MODE_MAP.get(v, v)
    return v if v in EVIDENCE_MODES else default


class AdjustmentPolicy(BaseModel):
    """A versioned, attributable set of evidence-handler coefficients.

    ``coefficients`` maps ``signal_type`` -> a handler-specific param dict. Every
    handler also reads a ``cap`` (max absolute fractional deviation of the
    factor from 1.0) so a single signal can never swing a mean unboundedly.
    """

    # Identity
    policy_id: str = Field(description="Unique ID, e.g. 'adj_v1_seed'")
    schema_version: int = 1
    version: int = Field(ge=1, description="Monotonically increasing")
    status: ProfileStatus = ProfileStatus.CANDIDATE
    mode: EvidenceMode = Field(
        default="score_only",
        description=(
            "Graduated rollout: disabled | observe | score_only | bounded_live | "
            "live. Non-applying modes (disabled/observe/score_only) record but "
            "never move predictions; bounded_live applies under hard caps; live "
            "applies under the policy's own caps. Legacy 'shadow' maps to "
            "'score_only' on load."
        ),
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _map_legacy_mode(cls, v: Any) -> Any:
        """Map legacy ``shadow``/``live`` strings to graduated modes on load.

        Runs before Literal validation so a policy persisted with mode='shadow'
        parses cleanly as 'score_only' instead of raising.
        """
        if isinstance(v, str):
            return LEGACY_MODE_MAP.get(v.strip().lower(), v)
        return v

    # Coefficients consumed by evidence_handlers.py
    coefficients: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="signal_type -> handler param dict (always includes 'cap').",
    )

    # Feature flags (Issue #22). All default False so a policy persisted before
    # these fields existed parses unchanged and behaves bit-identically: pydantic
    # fills the defaults on load, and no engine path reads a flag until the phase
    # that wires it lands. Each flag gates one behaviour-changing layer.
    enable_confidence_weighting: bool = Field(
        default=False,
        description="Scale each signal's family-capped factor by the agent's "
        "stated confidence before plane aggregation (sequence step 6).",
    )
    enable_correlation_damping: bool = Field(
        default=False,
        description="Damp correlated signals that share a damping_family so "
        "co-occurring evidence cannot stack unbounded (sequence step 4).",
    )
    enable_competition_strength_index: bool = Field(
        default=False,
        description="Apply the structural soccer competition_strength_index to "
        "team-context inputs before Bivariate Poisson lambda derivation.",
    )

    # Damping / cap parameters (Issue #22). Only consulted on the paths the flags
    # above enable, so defaults are behaviour-preserving: correlation_damping_weight
    # is unused unless enable_correlation_damping is set, and the two caps are
    # skipped entirely while None (no clamp = legacy behaviour).
    correlation_damping_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight applied to each secondary signal's signed delta when "
        "damping a co-occurring family (0 = keep only the primary, 1 = no damping).",
    )
    family_cap: float | None = Field(
        default=None,
        ge=0.0,
        description="Max absolute fractional deviation of a damped family's factor "
        "from 1.0 (sequence step 5). None = no family cap; must be >= 0 when set.",
    )
    plane_cap: float | None = Field(
        default=None,
        ge=0.0,
        description="Max absolute fractional deviation of the aggregated plane "
        "factor from 1.0 (sequence step 8). None = no plane cap; must be >= 0 when set.",
    )
    single_cap_ceiling: float | None = Field(
        default=None,
        ge=0.0,
        description="Hard ceiling on every signal's per-signal cap (sequence step "
        "3): the effective cap is min(coeff cap, ceiling). None = no extra "
        "ceiling (legacy). Set by bounded_live to bound single-signal impact.",
    )

    # Training provenance (empty for the hand-seeded v1 priors)
    training_window: str = ""
    sample_size: int = Field(default=0, ge=0)
    dataset_hash: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str = ""

    # Lifecycle
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    promoted_at: str | None = None
    rejected_at: str | None = None
    reject_reason: str | None = None
    incumbent_id: str | None = None

    def coeffs_for(self, signal_type: str) -> dict[str, Any]:
        """Return a copy of the coefficient dict for one signal type (may be empty)."""
        return dict(self.coefficients.get(signal_type, {}))

    def evidence_metrics_passed(self) -> bool:
        """Whether this policy's evidence metrics clear the gate.

        Gate for "bounded_live evidence may contribute to an A-tier rec". The
        policy must clear ALL of:
          * fitted on enough data (``sample_size >= MIN_EVIDENCE_SAMPLES_FOR_GATE``);
          * carry recorded fit ``metrics``;
          * those signals must be *predictive*, not just plentiful —
            ``mean_reliability_weight >= MIN_EVIDENCE_RELIABILITY_FOR_GATE``.

        The hand-seeded v1 priors (sample_size=0, metrics={}) fail on the first
        bar. A policy fitted on noise (e.g. recent_form scoring directional
        accuracy 0.42 -> reliability weight 0.0) has plenty of samples but fails
        the reliability bar, so bounded_live can still never manufacture an A
        from un-predictive evidence.
        """
        if self.sample_size < MIN_EVIDENCE_SAMPLES_FOR_GATE or not self.metrics:
            return False
        mean_reliability = self.metrics.get("mean_reliability_weight")
        return (
            mean_reliability is not None
            and mean_reliability >= MIN_EVIDENCE_RELIABILITY_FOR_GATE
        )

    def bounded_live_effective(self) -> AdjustmentPolicy:
        """Return a copy with bounded_live's hard caps forced on.

        bounded_live guarantees the three caps regardless of how the policy was
        configured: per-signal ceiling, family damping + cap, and plane cap. Caps
        the policy already sets tighter than the bounded_live default are kept
        (``min``); looser or unset ones are pulled to the bounded_live default.
        Used by the plane orchestrators when the resolved mode is bounded_live.
        """
        family_cap = self.family_cap
        family_cap = (
            BOUNDED_LIVE_FAMILY_CAP if family_cap is None else min(family_cap, BOUNDED_LIVE_FAMILY_CAP)
        )
        plane_cap = self.plane_cap
        plane_cap = (
            BOUNDED_LIVE_PLANE_CAP if plane_cap is None else min(plane_cap, BOUNDED_LIVE_PLANE_CAP)
        )
        ceiling = self.single_cap_ceiling
        ceiling = (
            BOUNDED_LIVE_SINGLE_CAP if ceiling is None else min(ceiling, BOUNDED_LIVE_SINGLE_CAP)
        )
        return self.model_copy(
            update={
                "enable_correlation_damping": True,
                "family_cap": family_cap,
                "plane_cap": plane_cap,
                "single_cap_ceiling": ceiling,
            }
        )


class AdjustmentPolicyRegistry:
    """Stores versioned adjustment policies with a promotion workflow.

    Storage is a single JSON file, re-read on every call (no caching) to avoid
    stale-state bugs — the expected policy count is tiny.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"schema_version": _SCHEMA_VERSION, "policies": []}
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read adjustment policy registry at %s: %s", self._path, exc)
            return {"schema_version": _SCHEMA_VERSION, "policies": []}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False)
        os.replace(str(tmp_path), str(self._path))

    def register(self, policy: AdjustmentPolicy) -> None:
        """Add a policy to the registry. Validates policy_id uniqueness."""
        data = self._load()
        existing_ids = {p["policy_id"] for p in data["policies"]}
        if policy.policy_id in existing_ids:
            raise ValueError(f"Adjustment policy ID already exists: {policy.policy_id}")
        data["policies"].append(policy.model_dump())
        self._save(data)
        logger.info(
            "Registered adjustment policy %s (version=%s, status=%s)",
            policy.policy_id,
            policy.version,
            policy.status.value,
        )

    def get_production_policy(self) -> AdjustmentPolicy | None:
        """Return the single active PRODUCTION policy, or None."""
        data = self._load()
        for p in data["policies"]:
            if p.get("status") == ProfileStatus.PRODUCTION.value:
                return AdjustmentPolicy(**p)
        return None

    def get_policy(self, policy_id: str) -> AdjustmentPolicy | None:
        """Retrieve a policy by ID."""
        data = self._load()
        for p in data["policies"]:
            if p["policy_id"] == policy_id:
                return AdjustmentPolicy(**p)
        return None

    def promote(self, policy_id: str) -> None:
        """Promote a candidate to production. Archives the current production policy."""
        data = self._load()
        target = next((p for p in data["policies"] if p["policy_id"] == policy_id), None)
        if target is None:
            raise ValueError(f"Adjustment policy not found: {policy_id}")
        if target["status"] != ProfileStatus.CANDIDATE.value:
            raise ValueError(
                f"Cannot promote policy with status={target['status']} "
                f"(must be {ProfileStatus.CANDIDATE.value})"
            )
        now = datetime.now(UTC).isoformat()
        for p in data["policies"]:
            if p.get("status") == ProfileStatus.PRODUCTION.value:
                p["status"] = ProfileStatus.ARCHIVED.value
                logger.info("Archived incumbent adjustment policy %s", p["policy_id"])
        target["status"] = ProfileStatus.PRODUCTION.value
        target["promoted_at"] = now
        self._save(data)
        logger.info("Promoted adjustment policy %s to production", policy_id)

    def reject(self, policy_id: str, reason: str) -> None:
        """Reject a candidate policy with a documented reason."""
        data = self._load()
        for p in data["policies"]:
            if p["policy_id"] == policy_id:
                p["status"] = ProfileStatus.REJECTED.value
                p["rejected_at"] = datetime.now(UTC).isoformat()
                p["reject_reason"] = reason
                self._save(data)
                logger.info("Rejected adjustment policy %s: %s", policy_id, reason)
                return
        raise ValueError(f"Adjustment policy not found: {policy_id}")

    def set_mode(self, policy_id: str, mode: str) -> None:
        """Set a policy's graduated rollout mode.

        The single behavior-changing edit in the rollout: flipping the
        production policy to 'bounded_live' (or 'live') is what makes structured
        evidence affect predictions. Kept as an explicit, auditable registry
        operation. Legacy 'shadow' is accepted and normalized to 'score_only'.
        """
        normalized = normalize_evidence_mode(mode, default="")
        if normalized not in EVIDENCE_MODES:
            raise ValueError(
                f"mode must be one of {sorted(EVIDENCE_MODES)} (or legacy "
                f"'shadow'), got {mode!r}"
            )
        data = self._load()
        for p in data["policies"]:
            if p["policy_id"] == policy_id:
                p["mode"] = normalized
                self._save(data)
                logger.info("Set adjustment policy %s mode=%s", policy_id, normalized)
                return
        raise ValueError(f"Adjustment policy not found: {policy_id}")

    def list_policies(self, status: str | None = None) -> list[AdjustmentPolicy]:
        """List policies, optionally filtered by status."""
        data = self._load()
        results = []
        for p in data["policies"]:
            if status and p.get("status") != status:
                continue
            results.append(AdjustmentPolicy(**p))
        return results
