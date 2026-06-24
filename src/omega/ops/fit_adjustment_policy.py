"""
omega.ops.fit_adjustment_policy â€” fit a CANDIDATE engine adjustment policy.

Closes the structured-reasoning loop: it reads retrospective signal-performance
aggregates (produced by omega-score-evidence-signals) and derives a
``reliability_weight`` per signal type, damping the engine's adjustment for
signals that scored as noise and trusting signals that proved predictive.

The new coefficients are written as a **CANDIDATE** AdjustmentPolicy. This
script never promotes and never changes live behavior â€” promotion is a separate,
gated, operator-driven step (omega-promote-adjustment-policy).

Reliability rule (deterministic):
    reliability_weight = clamp( 2 * (weighted_accuracy - 0.5), 0.0, 1.0 )
A directional signal is a coin flip at 0.50 accuracy -> weight 0.0 (fully
damped). 0.75 -> 0.5. 1.00 -> 1.0. Only signal types with at least
--min-samples scored observations are adjusted; the rest keep full trust.

Usage:
    omega-fit-adjustment-policy
    omega-fit-adjustment-policy --league NBA --min-samples 30
    omega-fit-adjustment-policy --mode live --dry-run

Exit codes:
    0 â€” candidate registered (or dry-run completed)
    1 â€” fatal error or no signal-performance data available
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dataclasses import dataclass, field  # noqa: E402

from omega.core.calibration.adjustment_policy import (  # noqa: E402
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.calibration.profiles import ProfileStatus  # noqa: E402
from omega.strategy.clv_significance import (  # noqa: E402
    DEFAULT_FDR_Q,
    DEFAULT_POWER,
    DEFAULT_POWER_EDGE,
    ProbationStats,
    clv_pvalue_from_stats,
    graduation_mask,
    min_samples_for_power,
    normal_lower_bound,
    pooled_mean_std,
)
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("fit_adjustment_policy")

_DEFAULT_MIN_SAMPLES = 30
# Minimum CLV observations before CLV is used at all (else degrade to direction).
# Distinct from the much stricter power-derived N_min, which is the graduation floor.
_MIN_CLV_SAMPLE = 20
# Cap on reliability_weight for a signal with measured CLV that has NOT cleared the
# statistical bar (fail-closed: trust only proven edge). Misaligned -> 0.
_UNPROVEN_CEILING = 0.25
# CLV share when blending a graduated signal's weight with realized direction.
_GRAD_CLV_WEIGHT = 0.75


def _clamp_weight(x: float) -> float:
    """The legacy reliability shape: coin flip (0.5) -> 0, perfect (1.0) -> 1."""
    return max(0.0, min(1.0, 2.0 * (x - 0.5)))


@dataclass
class _SignalAgg:
    """Per-signal_type rollup across its (source, window, league) rows."""

    dir_n: int = 0
    dir_correct: int = 0
    clv_n: int = 0
    clv_aligned_weighted: float = 0.0  # Σ clv_aligned_i * clv_sample_i
    clv_rows: list[tuple[int, float, float]] = field(default_factory=list)  # (n, mean, std)

    @property
    def direction_accuracy(self) -> float:
        return (self.dir_correct / self.dir_n) if self.dir_n else 0.0

    @property
    def clv_aligned(self) -> float | None:
        return (self.clv_aligned_weighted / self.clv_n) if self.clv_n else None


def _aggregate_by_signal_type(perf_rows: list[dict]) -> dict[str, _SignalAgg]:
    """Roll signal_performance rows up to one aggregate per signal_type.

    Direction counts sum; CLV is pooled via sufficient stats so the fit can apply
    the statistical bar at the signal_type level (the policy coefficient key).
    """
    aggs: dict[str, _SignalAgg] = {}
    for r in perf_rows:
        a = aggs.setdefault(r["signal_type"], _SignalAgg())
        a.dir_n += int(r.get("sample_size") or 0)
        a.dir_correct += int(r.get("direction_correct") or 0)
        clv_n = int(r.get("clv_sample") or 0)
        if clv_n > 0 and r.get("clv_aligned") is not None:
            a.clv_n += clv_n
            a.clv_aligned_weighted += float(r["clv_aligned"]) * clv_n
            a.clv_rows.append(
                (
                    clv_n,
                    float(r.get("clv_cents_when_followed") or 0.0),
                    float(r.get("clv_cents_std") or 0.0),
                )
            )
    return aggs


def _probation_stats(aggs: dict[str, _SignalAgg], n_min: int) -> dict[str, ProbationStats]:
    """Build the per-signal_type statistical-bar inputs from pooled CLV stats."""
    stats: dict[str, ProbationStats] = {}
    for st, a in aggs.items():
        if a.clv_n <= 0:
            continue
        pn, pmean, pstd = pooled_mean_std(a.clv_rows)
        lb = normal_lower_bound(pmean, pstd, pn)
        stats[st] = ProbationStats(
            n=pn,
            n_min=n_min,
            clv_mean=pmean,
            boot_lower_bound=lb,
            pvalue=clv_pvalue_from_stats(pmean, pstd, pn),
            meets_n_min=(pn >= n_min),
            boot_positive=(lb > 0.0),
        )
    return stats


def _reliability_weight(
    agg: _SignalAgg, *, graduated: bool, min_samples: int
) -> float | None:
    """CLV-primary reliability weight in [0, 1], or None to leave base trust intact.

    - No / thin CLV coverage -> graceful degradation to direction accuracy (legacy),
      only if there is enough direction sample; else None (keep base trust).
    - Graduated (cleared the bootstrap + N_min + FDR bar) -> CLV-primary weight,
      confirmed by realized direction once the direction sample is large.
    - Measured CLV but bar NOT cleared -> fail-closed: capped low; a line-restating
      signal (clv_aligned <= 0.5) damps to 0.
    """
    clv_aligned = agg.clv_aligned
    has_clv = clv_aligned is not None and agg.clv_n >= _MIN_CLV_SAMPLE
    if not has_clv:
        if agg.dir_n >= min_samples:
            return round(_clamp_weight(agg.direction_accuracy), 4)
        return None
    clv_w = _clamp_weight(clv_aligned)
    if graduated:
        if agg.dir_n >= min_samples:
            blended = _GRAD_CLV_WEIGHT * clv_w + (1.0 - _GRAD_CLV_WEIGHT) * _clamp_weight(
                agg.direction_accuracy
            )
            return round(blended, 4)
        return round(clv_w, 4)
    return round(min(clv_w, _UNPROVEN_CEILING), 4)


def _next_version(registry: AdjustmentPolicyRegistry) -> int:
    existing = registry.list_policies()
    return max((p.version for p in existing), default=0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit a CANDIDATE adjustment policy from signal performance."
    )
    parser.add_argument("--league", default=None, help="Score one league (default: all)")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=_DEFAULT_MIN_SAMPLES,
        help=f"Min scored observations to adjust a signal type (default: {_DEFAULT_MIN_SAMPLES})",
    )
    parser.add_argument(
        "--mode",
        choices=["shadow", "live"],
        default="shadow",
        help="Rollout mode baked into the candidate (default: shadow)",
    )
    parser.add_argument(
        "--power-edge",
        type=float,
        default=DEFAULT_POWER_EDGE,
        help=f"CLV edge the N_min power analysis must detect (default: {DEFAULT_POWER_EDGE})",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=DEFAULT_POWER,
        help=f"Statistical power for the N_min floor (default: {DEFAULT_POWER})",
    )
    parser.add_argument(
        "--fdr-q",
        type=float,
        default=DEFAULT_FDR_Q,
        help=f"Benjamini-Hochberg target false-discovery rate (default: {DEFAULT_FDR_Q})",
    )
    parser.add_argument("--db", type=str, default=None, help="SQLite path")
    parser.add_argument(
        "--policy-path", type=str, default=None, help="adjustment_policies.json path"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute and report but do not register"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = TraceStore(db_path=args.db)
    perf_rows = store.get_signal_performance(league=args.league, limit=10_000)
    store.close()

    if not perf_rows:
        logger.error("No signal-performance data. Run omega-score-evidence-signals first.")
        return 1

    dataset_hash = str(perf_rows[0].get("dataset_hash") or "")
    aggregates = _aggregate_by_signal_type(perf_rows)

    # The statistical bar (issue #28): bootstrap/normal lower bound + N_min + FDR.
    n_min = min_samples_for_power(edge=args.power_edge, power=args.power)
    stats_by_type = _probation_stats(aggregates, n_min)
    graduated = graduation_mask(stats_by_type, q=args.fdr_q)
    logger.info(
        "Statistical bar: N_min=%d (edge=%.3f, power=%.2f), FDR q=%.2f; "
        "%d signal types carry CLV, %d graduate.",
        n_min,
        args.power_edge,
        args.power,
        args.fdr_q,
        len(stats_by_type),
        sum(1 for v in graduated.values() if v),
    )

    registry = AdjustmentPolicyRegistry(path=args.policy_path)
    base = registry.get_production_policy()
    if base is None:
        logger.error("No production adjustment policy to use as a coefficient base.")
        return 1

    # Deep-copy the base coefficients, then set reliability_weight per signal type.
    new_coeffs = {st: dict(params) for st, params in base.coefficients.items()}
    fitted: list[tuple[str, _SignalAgg, float, bool]] = []
    for signal_type, agg in sorted(aggregates.items()):
        is_grad = graduated.get(signal_type, False)
        weight = _reliability_weight(agg, graduated=is_grad, min_samples=args.min_samples)
        if weight is None:
            logger.info(
                "  skip %-22s dir_n=%-3d clv_n=%-3d — insufficient sample, keeps full trust",
                signal_type,
                agg.dir_n,
                agg.clv_n,
            )
            continue
        params = new_coeffs.setdefault(signal_type, {"cap": 0.10})
        params["reliability_weight"] = weight
        fitted.append((signal_type, agg, weight, is_grad))
        logger.info(
            "  fit  %-22s dir_n=%-3d acc=%.3f | clv_n=%-3d clv_align=%s grad=%s "
            "-> reliability_weight=%.3f",
            signal_type,
            agg.dir_n,
            agg.direction_accuracy,
            agg.clv_n,
            "n/a" if agg.clv_aligned is None else f"{agg.clv_aligned:.3f}",
            "Y" if is_grad else "n",
            weight,
        )

    if not fitted:
        logger.error("No signal type met the sample bar; nothing to fit.")
        return 1

    version = _next_version(registry)
    policy_id = f"adj_v{version}_{dataset_hash[:12] or 'nodata'}"
    mean_weight = sum(w for _, _, w, _ in fitted) / len(fitted)
    n_graduated = sum(1 for _, _, _, g in fitted if g)
    candidate = AdjustmentPolicy(
        policy_id=policy_id,
        version=version,
        status=ProfileStatus.CANDIDATE,
        mode=args.mode,
        coefficients=new_coeffs,
        training_window=datetime.now(UTC).date().isoformat(),
        sample_size=sum(a.dir_n for _, a, _, _ in fitted),
        dataset_hash=dataset_hash,
        metrics={
            "signals_fitted": float(len(fitted)),
            "signals_graduated": float(n_graduated),
            "mean_reliability_weight": round(mean_weight, 4),
            "n_min": float(n_min),
        },
        notes=(
            f"Fitted from signal_performance (dataset {dataset_hash[:12]}); "
            f"{len(fitted)} signal types adjusted, {n_graduated} graduated the CLV bar "
            f"(N_min={n_min}, FDR q={args.fdr_q}). reliability_weight is CLV-primary "
            f"(direction-accuracy fallback where closing-line coverage is thin)."
        ),
        incumbent_id=base.policy_id,
    )

    if args.dry_run:
        logger.info(
            "DRY-RUN â€” candidate %s (version %d, mode=%s, %d signals fitted).",
            policy_id,
            version,
            args.mode,
            len(fitted),
        )
    else:
        registry.register(candidate)
        logger.info(
            "Registered CANDIDATE %s (version %d, mode=%s). "
            "Review and promote with omega-promote-adjustment-policy.",
            policy_id,
            version,
            args.mode,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
