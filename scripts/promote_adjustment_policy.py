"""
scripts/promote_adjustment_policy.py — promote a CANDIDATE adjustment policy.

Promotion makes a fitted set of evidence coefficients the production policy.
It is gated and operator-driven — never automatic — because promoting (and
especially flipping to ``--go-live``) is the one step that lets structured
evidence change live predictions.

Gates:
    1. SAMPLE_SIZE       — candidate.sample_size >= --min-samples (default 100)
    2. BACKTEST_IMPROVES — shadow-recorded adjustments improve Brier/ECE on a
                           held-out window [REQUIRES DATA] — there is no
                           automated check; confirm with --confirm-backtest
                           after reviewing a backtest run.

``--go-live`` additionally flips the promoted policy to mode='live'. Without it
the promoted policy stays in shadow mode (recorded but not applied), which is
the safe default while coefficients are still being validated.

Usage:
    python scripts/promote_adjustment_policy.py --list-candidates
    python scripts/promote_adjustment_policy.py --candidate-id adj_v2_abc123
    python scripts/promote_adjustment_policy.py --candidate-id <id> --auto --confirm-backtest
    python scripts/promote_adjustment_policy.py --candidate-id <id> --auto \\
        --confirm-backtest --go-live

Exit codes:
    0 — promotion succeeded (or list completed)
    1 — a gate failed and --force was not supplied
    2 — fatal error (unknown candidate, wrong status)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.core.calibration.adjustment_policy import (  # noqa: E402
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.calibration.profiles import ProfileStatus  # noqa: E402

logger = logging.getLogger("promote_adjustment_policy")

_DEFAULT_MIN_SAMPLES = 100


def _evaluate_gates(
    candidate: AdjustmentPolicy,
    min_samples: int,
    confirm_backtest: bool,
) -> list[tuple[str, bool, str]]:
    """Return (gate_name, passed, message) tuples."""
    results: list[tuple[str, bool, str]] = []

    n = candidate.sample_size
    results.append(
        ("SAMPLE_SIZE", n >= min_samples, f"sample_size={n}, required>={min_samples}")
    )
    results.append(
        (
            "BACKTEST_IMPROVES",
            confirm_backtest,
            "operator-confirmed via --confirm-backtest"
            if confirm_backtest
            else "no automated check — confirm with --confirm-backtest after "
            "a backtest shows shadow adjustments improve Brier/ECE",
        )
    )
    return results


def _print_gates(results: list[tuple[str, bool, str]]) -> None:
    for name, passed, msg in results:
        logger.info("  [%s] %-18s %s", "PASS" if passed else "FAIL", name, msg)


def _list_candidates(registry: AdjustmentPolicyRegistry) -> int:
    candidates = registry.list_policies(status=ProfileStatus.CANDIDATE.value)
    if not candidates:
        logger.info("No CANDIDATE adjustment policies.")
        return 0
    logger.info("Found %d CANDIDATE adjustment policy(ies):", len(candidates))
    for p in candidates:
        logger.info(
            "  %s  version=%d  mode=%s  n=%d  signals_fitted=%s",
            p.policy_id,
            p.version,
            p.mode,
            p.sample_size,
            int(p.metrics.get("signals_fitted", 0)),
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote a CANDIDATE adjustment policy to PRODUCTION after gate checks."
    )
    parser.add_argument("--candidate-id", help="policy_id of the CANDIDATE to promote")
    parser.add_argument("--auto", action="store_true", help="Promote iff ALL gates pass")
    parser.add_argument("--force", action="store_true", help="Promote regardless of gates")
    parser.add_argument(
        "--list-candidates", action="store_true", help="List CANDIDATE policies and exit"
    )
    parser.add_argument("--min-samples", type=int, default=_DEFAULT_MIN_SAMPLES)
    parser.add_argument(
        "--policy-path", type=str, default=None, help="adjustment_policies.json path"
    )
    parser.add_argument(
        "--confirm-backtest",
        action="store_true",
        help="Mark BACKTEST_IMPROVES as passed; operator has reviewed a backtest.",
    )
    parser.add_argument(
        "--go-live",
        action="store_true",
        help="After promotion, flip the policy to mode='live' (applies adjustments).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    registry = AdjustmentPolicyRegistry(path=args.policy_path)

    if args.list_candidates:
        return _list_candidates(registry)

    if not args.candidate_id:
        parser.error("--candidate-id is required unless --list-candidates is set")

    candidate = registry.get_policy(args.candidate_id)
    if candidate is None:
        logger.error("Candidate not found: %s", args.candidate_id)
        return 2
    if candidate.status != ProfileStatus.CANDIDATE:
        logger.error(
            "Policy %s has status=%s; only CANDIDATE policies can be promoted",
            candidate.policy_id,
            candidate.status.value,
        )
        return 2

    logger.info("Candidate: %s (version=%d, mode=%s)", candidate.policy_id,
                candidate.version, candidate.mode)
    incumbent = registry.get_production_policy()
    logger.info("Incumbent: %s", incumbent.policy_id if incumbent else "NONE")

    gates = _evaluate_gates(candidate, args.min_samples, args.confirm_backtest)
    logger.info("Gate status:")
    _print_gates(gates)
    all_pass = all(passed for _, passed, _ in gates)

    def _do_promote() -> None:
        registry.promote(candidate.policy_id)
        logger.info("Promoted %s to PRODUCTION.", candidate.policy_id)
        if args.go_live:
            registry.set_mode(candidate.policy_id, "live")
            logger.warning(
                "Policy %s is now mode=LIVE — structured evidence will adjust "
                "live predictions.",
                candidate.policy_id,
            )
        else:
            logger.info(
                "Policy %s promoted in mode=%s. Re-run with --go-live (or edit "
                "the artifact) to apply adjustments to predictions.",
                candidate.policy_id,
                candidate.mode,
            )

    if args.force:
        logger.warning("Promoting with --force; gate failures ignored.")
        _do_promote()
        return 0

    if args.auto:
        if not all_pass:
            logger.error("Not promoting: one or more gates failed under --auto.")
            return 1
        _do_promote()
        return 0

    # Report-only mode.
    if all_pass:
        logger.info("All gates pass. Re-run with --auto to perform the promotion.")
    else:
        logger.info("Not all gates pass. Address failures or re-run with --force.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
