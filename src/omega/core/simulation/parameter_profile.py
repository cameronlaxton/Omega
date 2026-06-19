"""Backend parameter profiles — versioned, attributable, promotable structural
parameters for a deterministic simulation backend.

A ``BackendParameterProfile`` is to the *raw probability generator* what a
``CalibrationProfile`` is to the *post-simulation correction*: a versioned bundle
of the knobs that shape a backend's output, with training provenance, held-out
quality metrics, and a fail-closed promotion lifecycle. It generalizes the
single-parameter ``priors_dixon_coles`` rho-profile pattern (one
``candidate``/``production``/``archived`` row per competition) to the full set of
a backend's structural parameters.

Two kinds of parameter live together, governed as one unit:

* **Structural knobs** (small, finite) — e.g. soccer Dixon-Coles ``rho``,
  ``home_advantage``, the xG->lambda mapping coefficients, ``first_half_share``;
  NFL team-score dispersion ``k`` and score correlation. These live in
  ``params``.
* **Per-entity priors** (large tables: per-player NB ``k``, per-team xG,
  per-player pressure deltas) stay in their existing ``priors_*`` tables. The
  profile *pins* them via ``priors_as_of_date`` — an immutable snapshot pointer —
  so a later refit cannot silently change what a promoted profile reads.
  ``dataset_hash`` makes any mutation of the pinned snapshot detectable.

The profile satisfies the structural ``GateCandidate`` surface
(``profile_id`` / ``sample_size`` / ``metrics``) so the *same* fail-closed
promotion engine that gates calibration profiles
(:mod:`omega.core.governance.promotion_gates`) gates parameter profiles too —
there is no second promotion path. ``metrics`` here are the **raw**
(pre-calibration) held-out ECE/Brier/log-loss: a parameter profile must clear the
quality floor on its *uncalibrated* output, which is the whole point of the rail
(fix the joint/tail in the backend; let calibration correct only the residual).

Persistence (the ``parameter_profiles`` table) and the fail-closed
``promote_parameter_profile`` helper live in :mod:`omega.trace.parameter_profiles`,
mirroring how calibration storage lives in the registry, not on the model.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

UTC = timezone.utc

PARAMETER_PROFILE_SCHEMA_VERSION = 1


class ParameterProfileStatus(str, Enum):
    """Lifecycle status of a backend parameter profile.

    Values match :class:`omega.core.calibration.profiles.ProfileStatus` and the
    ``priors_dixon_coles`` status strings, so the same wire values mean the same
    thing across the calibration plane, the legacy rho table, and this table.
    """

    CANDIDATE = "candidate"  # Fitted, not yet promoted
    PRODUCTION = "production"  # Active — the gatherer injects this profile's params
    ARCHIVED = "archived"  # Replaced by a newer production profile
    REJECTED = "rejected"  # Failed gates or manually rejected


def param_content_hash(params: dict[str, Any]) -> str:
    """Deterministic 12-char hash of a params dict (order-independent).

    Used both to build a reproducible ``profile_id`` and to detect that the
    structural knobs changed between two profiles with the same lineage.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def make_parameter_profile_id(
    backend_name: str, competition_bucket: str, version: int, params: dict[str, Any]
) -> str:
    """Build a reproducible profile_id, e.g.
    ``soccer_bivariate_poisson_dc__FIFA_INTL__v2__a1b2c3d4e5f6``."""
    return f"{backend_name}__{competition_bucket}__v{version}__{param_content_hash(params)}"


class BackendParameterProfile(BaseModel):
    """A versioned, attributable, promotable set of backend structural parameters.

    Exactly one profile per ``(backend_name, competition_bucket)`` is
    ``PRODUCTION`` at a time; that is the one the gatherer reads and injects into
    ``request.prior_payload``. A backend that finds no production profile fails
    closed (``status="skipped"``), the same degraded-but-never-wrong behavior as a
    missing context dict today.
    """

    # Identity + binding
    profile_id: str = Field(description="Reproducible id; see make_parameter_profile_id()")
    schema_version: int = PARAMETER_PROFILE_SCHEMA_VERSION
    version: int = Field(ge=1, description="Monotonically increasing per (backend, bucket)")
    backend_name: str = Field(description="Registered backend, e.g. 'soccer_bivariate_poisson_dc'")
    backend_component_version: str = Field(
        description=(
            "The backend code version these params were fit against, e.g. "
            "'soccer_bvp_dc_v1'. Bound by calibration profiles (P8.3): a "
            "calibration map fit on this version must not serve a different one."
        )
    )
    competition_bucket: str = Field(
        description="Calibration/competition bucket, e.g. 'FIFA_INTL' (resolve_calibration_bucket)"
    )

    # The governed structural parameters + the pinned per-entity priors snapshot
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Structural knobs the backend consumes via prior_payload (rho, hca, ...)",
    )
    priors_as_of_date: str | None = Field(
        default=None,
        description=(
            "Immutable pointer into the per-entity priors_* tables. The gatherer "
            "reads priors rows as-of this date so a later refit cannot change what "
            "a promoted profile sees. None for backends with no per-entity priors."
        ),
    )

    # Training provenance
    dataset_manifest_id: str | None = Field(
        default=None, description="Frozen historical DatasetManifest id used to fit/evaluate"
    )
    dataset_hash: str = Field(description="sha256 of the frozen fit/eval dataset (+ pinned priors)")
    sample_size: int = Field(ge=0, description="Held-out eval sample size (drives SAMPLE_SIZE gate)")

    # RAW (pre-calibration) held-out quality — drives ECE_FLOOR/BRIER/LOG_LOSS gates
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Raw (uncalibrated) brier_score, calibration_error, log_loss, n_eval, cv_*. "
            "dict[str, Any] (not float) so integer-semantic metrics (cv_n_folds, n_eval) "
            "keep their int type instead of being coerced to float."
        ),
    )

    status: ParameterProfileStatus = ParameterProfileStatus.CANDIDATE

    # Lifecycle timestamps
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    promoted_at: str | None = None
    rejected_at: str | None = None
    reject_reason: str | None = None

    # Promotion lineage
    incumbent_id: str | None = Field(
        default=None, description="profile_id of the incumbent this was compared against"
    )
    promotion_gate_report: dict[str, Any] | None = Field(
        default=None,
        description="Gate evaluation recorded at promotion (audit trail); None until promoted",
    )

    def trace_ref(self) -> dict[str, Any]:
        """The provenance reference embedded into a trace (P8.0.3).

        Lets a persisted probability be attributed to the exact parameter set
        that generated it, and lets replay pin parameters instead of re-reading
        live tables.
        """
        return {
            "backend_name": self.backend_name,
            "backend_component_version": self.backend_component_version,
            "param_profile_id": self.profile_id,
            "competition_bucket": self.competition_bucket,
            "priors_as_of_date": self.priors_as_of_date,
            "dataset_hash": self.dataset_hash,
        }
