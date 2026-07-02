"""
Calibration profiles — versioned, attributable calibration configurations.

Each profile captures a calibration method + parameters, training provenance,
quality metrics, and lifecycle status. Profiles are league-specific.

Storage and selection are handled by CalibrationRegistry (registry.py).
Fitting is handled by CalibrationFitter (fitter.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

UTC = timezone.utc


class ProfileStatus(str, Enum):
    """Lifecycle status of a calibration profile."""

    CANDIDATE = "candidate"  # Newly fitted, not yet evaluated
    PRODUCTION = "production"  # Active — used for calibration
    ARCHIVED = "archived"  # Replaced by a newer production profile
    REJECTED = "rejected"  # Failed evaluation or manually rejected


class ProfileMaturity(str, Enum):
    """Trust level of an *active* calibration profile (orthogonal to status).

    ``status`` is the promotion state machine (candidate→production→archived).
    ``maturity`` grades how much a *production*-status profile may move a
    probability and how high a confidence it may support. This breaks the
    all-or-nothing trap: a sparse market can apply a small, capped correction at
    ``provisional`` maturity instead of waiting forever for full ``production``.

    none        -> not trusted to apply (no correction).
    provisional -> applies a small capped correction; confidence capped low.
    probation   -> larger capped correction; confidence capped medium.
    production  -> full correction; no maturity-based confidence cap.
    retired     -> previously trusted, now withdrawn (no correction).
    """

    NONE = "none"
    PROVISIONAL = "provisional"
    PROBATION = "probation"
    PRODUCTION = "production"
    RETIRED = "retired"


# Maturities whose calibration correction is trusted to be applied to live math.
APPLYING_MATURITIES: frozenset[ProfileMaturity] = frozenset(
    {ProfileMaturity.PROVISIONAL, ProfileMaturity.PROBATION, ProfileMaturity.PRODUCTION}
)


# Sentinel ID for the hardcoded static policy (pre-Phase-6c behavior)
STATIC_FALLBACK_ID = "__static_v1__"


class BindingStatus(str, Enum):
    """How a profile's raw-probability substrate identity was declared (P8.3).

    bound    -> the fit recorded at least one substrate identity field; the
                profile is applied only when the live substrate matches.
    unpinned -> the fit checked its source traces and found NO substrate
                identity (homogeneously unpinned dataset); declared explicitly,
                applies anywhere but is flagged in the calibration audit.
    legacy   -> the profile predates P8.3 (no backend_binding recorded);
                applies anywhere, flagged in the calibration audit.
    """

    BOUND = "bound"
    UNPINNED = "unpinned"
    LEGACY = "legacy"


# Identity fields a binding may constrain; also the keys of a live substrate ref
# (see omega.trace.parameter_profiles.substrate_ref_for_trace).
BINDING_FIELDS = ("backend_name", "backend_component_version", "param_profile_id")


class CalibrationBackendBinding(BaseModel):
    """Raw-probability substrate a calibration profile was fit against (P8.3).

    A calibration profile corrects the residual miscalibration of one specific
    probability-producing substrate: a backend (name + component version) running
    one governed :class:`BackendParameterProfile` (``param_profile_id``). Fields
    left None were unknown in the source traces' provenance and constrain
    nothing — the fitter never guesses or synthesizes identity beyond what the
    traces recorded.
    """

    backend_name: str | None = Field(
        default=None, description="Simulation backend the fit traces ran, e.g. 'prop_neg_binom'."
    )
    backend_component_version: str | None = Field(
        default=None, description="Backend component version of the fit traces, e.g. 'prop_nb_v1'."
    )
    param_profile_id: str | None = Field(
        default=None,
        description=(
            "profile_id of the governed BackendParameterProfile whose structural "
            "knobs produced the fit traces' raw probabilities; None when the fit "
            "dataset was (homogeneously) unpinned."
        ),
    )

    def is_pinned(self) -> bool:
        """True when at least one identity field constrains application."""
        return any(getattr(self, f) is not None for f in BINDING_FIELDS)

    def mismatch_reason(self, substrate: dict[str, Any] | None) -> str | None:
        """Why this binding rejects a live substrate ref, or None if compatible.

        ``substrate`` uses the BINDING_FIELDS keys. Fail-closed semantics: a
        pinned binding never applies against an unknown (None) substrate or one
        missing a constrained field — a mismatch is returned rather than
        silently applying a profile fit on a different raw-probability
        substrate. Unconstrained (None) binding fields match anything.
        """
        if not self.is_pinned():
            return None
        if substrate is None:
            return "substrate_unknown"
        for field in BINDING_FIELDS:
            want = getattr(self, field)
            if want is None:
                continue
            live = substrate.get(field)
            if live is None:
                return f"substrate_missing:{field}"
            if str(live) != str(want):
                return f"{field}_mismatch:fit={want},live={live}"
        return None


class CalibrationProfile(BaseModel):
    """A versioned, attributable calibration configuration.

    Every profile is league-specific and captures enough metadata to
    reproduce, evaluate, and audit its creation.
    """

    # Identity
    profile_id: str = Field(description="Unique ID, e.g. 'iso_nba_v3' or 'iso_nba_playoff_v1'")
    schema_version: int = 1
    version: int = Field(ge=1, description="Monotonically increasing per league")
    method: str = Field(description="shrinkage | cap | isotonic | combined | none")
    league: str = Field(description="League code, e.g. 'NBA'")
    context_slice: str | None = Field(
        default=None,
        description=(
            "Optional sub-population this profile was fitted on. "
            "None = base profile (all contexts). "
            "Examples: 'playoff', 'regular', 'back_to_back'. "
            "Registry falls back to base profile when no slice-specific profile exists."
        ),
    )
    market: str = Field(
        default="game",
        description=(
            "Calibration market plane this profile applies to. "
            "'game' = moneyline/spread/total win probabilities (default, "
            "covers legacy profiles). 'prop' = player-prop over/under "
            "probabilities (fit on the prop plane, applied only to props). "
            "'draw' = 3-way draw probabilities (soccer, hockey regulation). "
            "A profile is only applied to the plane it was fit on; selection "
            "falls back from a prop/draw market to the 'game' profile, then to "
            "the static policy, when the specific market is absent."
        ),
    )
    status: ProfileStatus = ProfileStatus.CANDIDATE
    maturity: ProfileMaturity | None = Field(
        default=None,
        description=(
            "Trust level of an active profile, orthogonal to status. None on "
            "legacy profiles -> derived from status via effective_maturity() "
            "(PRODUCTION status -> PRODUCTION maturity, else NONE). provisional/"
            "probation apply capped corrections and cap confidence; production "
            "applies the full correction."
        ),
    )

    # Method parameters — passed as kwargs to calibrate_probability()
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific: {shrink_factor}, {cap_max, cap_min}, or {calibration_map: {...}}",
    )

    # Training provenance
    training_window: str = Field(description="e.g. '2024-01-01/2024-12-31'")
    sample_size: int = Field(ge=0)
    dataset_hash: str = Field(description="sha256 of sorted prediction+outcome arrays")

    # Quality metrics (measured on held-out set)
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="brier_score, calibration_error, log_loss, n_eval",
    )

    # Lifecycle timestamps
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    promoted_at: str | None = None
    rejected_at: str | None = None
    reject_reason: str | None = None

    # Backend binding (P8.3): the raw-probability substrate this profile was fit
    # against. None on profiles persisted (or fitted by paths not yet threading a
    # binding) before P8.3 -> BindingStatus.LEGACY, applied as before but flagged.
    backend_binding: CalibrationBackendBinding | None = Field(
        default=None,
        description=(
            "Substrate identity (backend name/component version + governed "
            "param_profile_id) of the traces this profile was fit on. Runtime "
            "selection skips the profile when the live substrate does not match "
            "(fail-closed); None = legacy/undeclared (applies, flagged)."
        ),
    )

    # Promotion lineage
    incumbent_id: str | None = Field(
        default=None,
        description="profile_id of the incumbent this was compared against",
    )
    promotion_gate_report: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Gate evaluation recorded at promotion time (audit trail): which gates "
            "passed, the thresholds used, and the operator confirmations. None until "
            "promoted via the fail-closed CalibrationRegistry.promote()."
        ),
    )

    def binding_status(self) -> BindingStatus:
        """Substrate-declaration status of this profile (see BindingStatus)."""
        if self.backend_binding is None:
            return BindingStatus.LEGACY
        if self.backend_binding.is_pinned():
            return BindingStatus.BOUND
        return BindingStatus.UNPINNED

    def effective_maturity(self) -> ProfileMaturity:
        """Maturity to use, deriving a default for legacy profiles.

        Profiles persisted before the ``maturity`` field existed have it None;
        they were either full production (status PRODUCTION) or not applied.
        """
        if self.maturity is not None:
            return self.maturity
        return (
            ProfileMaturity.PRODUCTION
            if self.status == ProfileStatus.PRODUCTION
            else ProfileMaturity.NONE
        )

    @property
    def ece(self) -> float | None:
        """Expected Calibration Error from recorded metrics (held-out), if any."""
        return self.metrics.get("calibration_error")

    @property
    def brier(self) -> float | None:
        """Brier score from recorded metrics, if any."""
        return self.metrics.get("brier_score")

    @property
    def log_loss(self) -> float | None:
        """Log loss from recorded metrics, if any."""
        return self.metrics.get("log_loss")
