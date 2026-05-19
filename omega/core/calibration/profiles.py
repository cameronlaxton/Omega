"""
Calibration profiles — versioned, attributable calibration configurations.

Each profile captures a calibration method + parameters, training provenance,
quality metrics, and lifecycle status. Profiles are league-specific.

Storage and selection are handled by CalibrationRegistry (registry.py).
Fitting is handled by CalibrationFitter (fitter.py).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProfileStatus(str, Enum):
    """Lifecycle status of a calibration profile."""
    CANDIDATE = "candidate"      # Newly fitted, not yet evaluated
    PRODUCTION = "production"    # Active — used for calibration
    ARCHIVED = "archived"        # Replaced by a newer production profile
    REJECTED = "rejected"        # Failed evaluation or manually rejected


# Sentinel ID for the hardcoded static policy (pre-Phase-6c behavior)
STATIC_FALLBACK_ID = "__static_v1__"


class CalibrationProfile(BaseModel):
    """A versioned, attributable calibration configuration.

    Every profile is league-specific and captures enough metadata to
    reproduce, evaluate, and audit its creation.
    """

    # Identity
    profile_id: str = Field(description="Unique ID, e.g. 'iso_nba_v3'")
    schema_version: int = 1
    version: int = Field(ge=1, description="Monotonically increasing per league")
    method: str = Field(description="shrinkage | cap | isotonic | combined | none")
    league: str = Field(description="League code, e.g. 'NBA'")
    status: ProfileStatus = ProfileStatus.CANDIDATE

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
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    promoted_at: str | None = None
    rejected_at: str | None = None
    reject_reason: str | None = None

    # Promotion lineage
    incumbent_id: str | None = Field(
        default=None,
        description="profile_id of the incumbent this was compared against",
    )
