"""Honest, multi-factor confidence tiers.

Replaces the dishonest ``tier = "A" if n_iterations >= 1000 else "B"`` rule: a
thousand Monte-Carlo draws say nothing about whether the *probability* is
trustworthy. Confidence is assigned in two stages so each edit stays small:

* **Stage 1** (:func:`assign_confidence`, per edge/side) uses the market +
  profile + simulation signals available where the edge is built: edge/EV, the
  calibration path + profile maturity/metrics, and the iteration count. An ``A``
  here requires a real ``production``-maturity profile with a passing ECE and
  enough samples and iterations — iterations alone can only cap *down*.

* **Stage 2** (:func:`combine_trace_caps`, once per trace) lowers every edge to
  the most restrictive of the trace-level ceilings: the graded
  ``trace_quality.confidence_cap`` (which already encodes the aggregate-quality
  band, zero-evidence/empty-context, static_identity, and QA rules), the
  evidence-metrics gate (bounded_live evidence cannot reach A until its policy
  metrics pass), and imputation discipline.

This module is pure tier algebra + the two policies; the schema-aware wiring
(reading audits off edges, restaking) lives in ``service.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tier ordering, most → least restrictive. ``None`` as a ceiling means "no cap".
TIER_ORDER = ("Pass", "C", "B", "A")
_TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}

# Thresholds an edge must clear to earn an A in Stage 1.
EDGE_MIN_PCT = 3.0
MIN_A_SAMPLES = 100
MAX_A_ECE = 0.05
MIN_A_ITERATIONS = 1000

# Imputation discipline (Stage 2), preserving the historical B2 cap behaviour.
IMPUTATION_PASS_THRESHOLD = 0.4
IMPUTATION_CAP_THRESHOLD = 0.2

# Odds older than this (when known) cannot support an A — the price is stale.
STALE_ODDS_SECONDS = 60 * 60


@dataclass(frozen=True)
class ConfidenceResult:
    """A confidence tier with the binding cap reason and the full reason list."""

    tier: str
    cap_reason: str | None = None
    reasons: list[str] | None = None


def tier_rank(tier: str) -> int:
    """Rank of a tier (higher = less restrictive). Unknown tiers rank as Pass."""
    return _TIER_RANK.get(tier, 0)


def more_restrictive(a: str, b: str) -> str:
    """Return the lower (more restrictive) of two tiers."""
    return a if tier_rank(a) <= tier_rank(b) else b


def cap_tier(tier: str, ceiling: str | None) -> str:
    """Lower ``tier`` to ``ceiling`` if the ceiling is more restrictive."""
    if ceiling is None:
        return tier
    return more_restrictive(tier, ceiling)


def _stage1_ceiling(
    *,
    calibration_path: str | None,
    profile_maturity: str | None,
    profile_sample_size: int | None,
    profile_ece: float | None,
    n_iterations: int,
) -> tuple[str, str | None]:
    """The best tier an edge's own market/profile/sim signals can support.

    Returns ``(tier, reason)`` — ``("A", None)`` only when a fully-trusted
    production profile with a passing ECE and enough samples/iterations applied;
    otherwise ``("B", <first failing reason>)``. Never returns A on iteration
    count alone.
    """
    if calibration_path != "profile":
        return "B", "no_production_profile_calibration"
    if profile_maturity != "production":
        return "B", "profile_maturity_not_production"
    if (profile_sample_size or 0) < MIN_A_SAMPLES:
        return "B", "profile_sample_size_below_floor"
    if profile_ece is None or profile_ece > MAX_A_ECE:
        return "B", "profile_ece_above_floor"
    if n_iterations < MIN_A_ITERATIONS:
        return "B", "insufficient_iterations"
    return "A", None


def assign_confidence(
    *,
    edge_pct: float,
    ev_pct: float | None,
    calibration_path: str | None,
    profile_maturity: str | None,
    profile_sample_size: int | None,
    profile_ece: float | None,
    n_iterations: int,
    edge_threshold: float = EDGE_MIN_PCT,
) -> ConfidenceResult:
    """Stage 1: per-edge confidence from market + profile + simulation signals."""
    if abs(edge_pct) < edge_threshold:
        return ConfidenceResult("Pass", "edge_below_threshold", ["edge_below_threshold"])
    if ev_pct is not None and ev_pct < 0:
        return ConfidenceResult("Pass", "negative_ev", ["negative_ev"])
    tier, reason = _stage1_ceiling(
        calibration_path=calibration_path,
        profile_maturity=profile_maturity,
        profile_sample_size=profile_sample_size,
        profile_ece=profile_ece,
        n_iterations=n_iterations,
    )
    return ConfidenceResult(tier, reason, [reason] if reason else [])


def combine_trace_caps(
    *,
    trace_confidence_cap: str | None,
    evidence_mode: str | None,
    evidence_metrics_passed: bool,
    imputed_fraction: float | None,
    odds_age_seconds: float | None = None,
) -> list[tuple[str, str]]:
    """Stage 2: the trace-level ceilings to apply to every edge.

    Returns a list of ``(ceiling_tier, reason)`` constraints. The caller lowers
    each edge to the most restrictive of these (and the edge's own Stage 1 tier).
    """
    from omega.core.calibration.adjustment_policy import APPLYING_MODES

    caps: list[tuple[str, str]] = []
    if trace_confidence_cap:
        caps.append((trace_confidence_cap, "trace_quality_cap"))
    if evidence_mode in APPLYING_MODES and not evidence_metrics_passed:
        # bounded_live/live evidence is moving the math, but its policy metrics
        # have not passed gates — it must not manufacture an A.
        caps.append(("B", "evidence_metrics_unproven"))
    if imputed_fraction is not None:
        if imputed_fraction > IMPUTATION_PASS_THRESHOLD:
            caps.append(("Pass", "insufficient_real_observations"))
        elif imputed_fraction > IMPUTATION_CAP_THRESHOLD:
            caps.append(("B", "tier_capped_imputation"))
    if odds_age_seconds is not None and odds_age_seconds > STALE_ODDS_SECONDS:
        caps.append(("B", "stale_odds"))
    return caps


def most_restrictive_constraint(caps: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The single binding (lowest-tier) constraint from a cap list, or None."""
    if not caps:
        return None
    return min(caps, key=lambda c: tier_rank(c[0]))
