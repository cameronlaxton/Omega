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

from omega.core.calibration.adjustment_policy import (  # noqa: E402
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.calibration.profiles import ProfileStatus  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("fit_adjustment_policy")

_DEFAULT_MIN_SAMPLES = 30


def _reliability_weight(accuracy: float) -> float:
    """Map directional accuracy to a reliability weight in [0, 1]."""
    return max(0.0, min(1.0, 2.0 * (accuracy - 0.5)))


def _aggregate_by_signal_type(
    perf_rows: list[dict],
) -> dict[str, tuple[int, float]]:
    """Collapse signal_performance rows to (total_n, weighted_accuracy) per type."""
    totals: dict[str, list[int]] = {}
    for r in perf_rows:
        st = r["signal_type"]
        n = int(r["sample_size"])
        correct = int(r["direction_correct"])
        bucket = totals.setdefault(st, [0, 0])
        bucket[0] += n
        bucket[1] += correct
    return {
        st: (n, (correct / n) if n else 0.0)
        for st, (n, correct) in totals.items()
    }


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
        logger.error(
            "No signal-performance data. Run omega-score-evidence-signals first."
        )
        return 1

    dataset_hash = str(perf_rows[0].get("dataset_hash") or "")
    aggregates = _aggregate_by_signal_type(perf_rows)

    registry = AdjustmentPolicyRegistry(path=args.policy_path)
    base = registry.get_production_policy()
    if base is None:
        logger.error("No production adjustment policy to use as a coefficient base.")
        return 1

    # Deep-copy the base coefficients, then set reliability_weight per signal type.
    new_coeffs = {st: dict(params) for st, params in base.coefficients.items()}
    fitted: list[tuple[str, int, float, float]] = []
    for signal_type, (total_n, accuracy) in sorted(aggregates.items()):
        if total_n < args.min_samples:
            logger.info(
                "  skip %-22s n=%-3d (< %d) â€” keeps full trust",
                signal_type, total_n, args.min_samples,
            )
            continue
        weight = _reliability_weight(accuracy)
        params = new_coeffs.setdefault(signal_type, {"cap": 0.10})
        params["reliability_weight"] = round(weight, 4)
        fitted.append((signal_type, total_n, accuracy, weight))
        logger.info(
            "  fit  %-22s n=%-3d acc=%.3f -> reliability_weight=%.3f",
            signal_type, total_n, accuracy, weight,
        )

    if not fitted:
        logger.error(
            "No signal type met --min-samples=%d; nothing to fit.", args.min_samples
        )
        return 1

    version = _next_version(registry)
    policy_id = f"adj_v{version}_{dataset_hash[:12] or 'nodata'}"
    mean_weight = sum(w for _, _, _, w in fitted) / len(fitted)
    candidate = AdjustmentPolicy(
        policy_id=policy_id,
        version=version,
        status=ProfileStatus.CANDIDATE,
        mode=args.mode,
        coefficients=new_coeffs,
        training_window=datetime.now(UTC).date().isoformat(),
        sample_size=sum(n for _, n, _, _ in fitted),
        dataset_hash=dataset_hash,
        metrics={
            "signals_fitted": float(len(fitted)),
            "mean_reliability_weight": round(mean_weight, 4),
        },
        notes=(
            f"Fitted from signal_performance (dataset {dataset_hash[:12]}); "
            f"{len(fitted)} signal types adjusted, "
            f"reliability_weight derived from directional accuracy."
        ),
        incumbent_id=base.policy_id,
    )

    if args.dry_run:
        logger.info(
            "DRY-RUN â€” candidate %s (version %d, mode=%s, %d signals fitted).",
            policy_id, version, args.mode, len(fitted),
        )
    else:
        registry.register(candidate)
        logger.info(
            "Registered CANDIDATE %s (version %d, mode=%s). "
            "Review and promote with omega-promote-adjustment-policy.",
            policy_id, version, args.mode,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())




