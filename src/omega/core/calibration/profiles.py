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
