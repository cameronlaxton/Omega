"""Cross-domain governance primitives shared by the calibration and simulation
planes.

This package owns the *profile-type-agnostic* machinery that more than one domain
needs but neither domain should own:

- ``promotion_gates`` — the fail-closed promotion-gate engine used by BOTH
  calibration-profile promotion (``omega.core.calibration``) and backend
  parameter-profile promotion (``omega.core.simulation``). There is one
  definition of "is this candidate safe to promote", composed by each domain so
  the gates, thresholds, and evidence discipline cannot drift apart.

Nothing here knows about a specific profile type: callers pass any object
satisfying the structural :class:`~omega.core.governance.promotion_gates.GateCandidate`
contract (``profile_id``, ``sample_size``, ``metrics``).
"""

from __future__ import annotations

from omega.core.governance.promotion_gates import (
    DEFAULT_BRIER_IMPROVEMENT,
    DEFAULT_ECE_FLOOR,
    DEFAULT_LOG_LOSS_TOL,
    DEFAULT_MIN_SAMPLES,
    PROMOTION_GATE_REPORT_SCHEMA_VERSION,
    GateCandidate,
    GateReport,
    GateResult,
    PromotionGateError,
    artifact_indicates_pass,
    evaluate_promotion_gates,
)

__all__ = [
    "DEFAULT_BRIER_IMPROVEMENT",
    "DEFAULT_ECE_FLOOR",
    "DEFAULT_LOG_LOSS_TOL",
    "DEFAULT_MIN_SAMPLES",
    "PROMOTION_GATE_REPORT_SCHEMA_VERSION",
    "GateCandidate",
    "GateReport",
    "GateResult",
    "PromotionGateError",
    "artifact_indicates_pass",
    "evaluate_promotion_gates",
]
