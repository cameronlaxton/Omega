"""
omega-promote-profile — promote a CANDIDATE calibration profile to PRODUCTION.

Promotion is fail-closed: the registry always evaluates the shared promotion
gates (``omega.core.calibration.promotion``) and refuses to promote unless ALL
pass. There is no ``--force`` bypass. This CLI is a thin operator front-end:

  * default (no --auto): dry-run — print the gate status and exit non-zero if
    any gate fails;
  * --auto: perform the promotion through the fail-closed registry path.

Gates:
    SAMPLE_SIZE      candidate.sample_size >= --min-samples (default 100)
    ECE_FLOOR        candidate.calibration_error <= --ece-floor (default 0.05)
    BRIER_IMPROVES   candidate.brier_score improves on incumbent by >= --brier-improvement
    LOG_LOSS_NO_REG  candidate.log_loss does not regress vs incumbent by > --log-loss-tol
    BACKTEST_PARITY  operator-confirmed via --confirm-backtest-parity
    CLV_NON_REG      operator-confirmed via --confirm-clv-non-regression

If no incumbent exists for the (league, market, context_slice), the
improvement/no-regression gates auto-pass; the sample-size, ECE floor, and
operator-confirmation gates still apply.

This script does not fit or re-evaluate. Comparison metrics come from each
profile's stored ``metrics`` dict (filled at fit time by omega-fit-calibration).

Usage:
    omega-promote-profile --candidate-id iso_nba_v3_4f7a9d2b1c3e5f0a
    omega-promote-profile --candidate-id <id> --auto \
        --confirm-backtest-parity --confirm-clv-non-regression
    omega-promote-profile --list-candidates --league NBA

Exit codes:
    0 — promotion succeeded, or dry-run with all gates green, or list completed
    1 — at least one gate failed (dry-run) or promotion was blocked (--auto)
    2 — fatal error (unknown candidate, wrong status, etc.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.calibration.profiles import ProfileStatus  # noqa: E402
from omega.core.calibration.promotion import (  # noqa: E402
    DEFAULT_BRIER_IMPROVEMENT,
    DEFAULT_ECE_FLOOR,
    DEFAULT_LOG_LOSS_TOL,
    DEFAULT_MIN_SAMPLES,
    GateReport,
    PromotionGateError,
    evaluate_promotion_gates,
)
from omega.core.calibration.registry import CalibrationRegistry  # noqa: E402

logger = logging.getLogger("promote_profile")


def _print_gates(report: GateReport) -> None:
    for r in report.results:
        flag = "PASS" if r.passed else "FAIL"
        logger.info("  [%s] %-18s %s", flag, r.name, r.message)


def _list_candidates(registry: CalibrationRegistry, league: str | None) -> int:
    candidates = registry.list_profiles(league=league, status=ProfileStatus.CANDIDATE.value)
    if not candidates:
        logger.info("No CANDIDATE profiles%s.", f" for league={league}" if league else "")
        return 0
    logger.info("Found %d CANDIDATE profile(s):", len(candidates))
    for p in candidates:
        m = p.metrics
        logger.info(
            "  %s  league=%s  market=%s  method=%s  n=%d  brier=%.4f  ece=%.4f  log_loss=%.4f",
            p.profile_id,
            p.league,
            p.market or "game",
            p.method,
            p.sample_size,
            m.get("brier_score", float("nan")),
            m.get("calibration_error", float("nan")),
            m.get("log_loss", float("nan")),
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a CANDIDATE calibration profile to PRODUCTION after fail-closed gate checks."
    )
    parser.add_argument("--candidate-id", help="profile_id of the CANDIDATE to promote")
    parser.add_argument(
        "--auto", action="store_true", help="Perform the promotion (default: dry-run report only)"
    )
    parser.add_argument(
        "--list-candidates", action="store_true", help="List CANDIDATE profiles and exit"
    )
    parser.add_argument("--league", help="Filter for --list-candidates")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--brier-improvement", type=float, default=DEFAULT_BRIER_IMPROVEMENT)
    parser.add_argument("--log-loss-tol", type=float, default=DEFAULT_LOG_LOSS_TOL)
    parser.add_argument("--ece-floor", type=float, default=DEFAULT_ECE_FLOOR)
    parser.add_argument(
        "--confirm-backtest-parity",
        action="store_true",
        help="Mark gate BACKTEST_PARITY as passed; operator has manually verified.",
    )
    parser.add_argument(
        "--confirm-clv-non-regression",
        action="store_true",
        help="Mark gate CLV_NON_REG as passed; operator has manually verified.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    registry = CalibrationRegistry()

    if args.list_candidates:
        return _list_candidates(registry, args.league)

    if not args.candidate_id:
        parser.error("--candidate-id is required unless --list-candidates is set")

    candidate = registry.get_profile(args.candidate_id)
    if candidate is None:
        logger.error("Candidate not found: %s", args.candidate_id)
        return 2
    if candidate.status != ProfileStatus.CANDIDATE:
        logger.error(
            "Profile %s has status=%s; only CANDIDATE profiles can be promoted",
            candidate.profile_id,
            candidate.status.value,
        )
        return 2

    # Gating incumbent: same market, exact slice, else the base profile a sliced
    # candidate would shadow. Resolved by the registry so the CLI and the
    # fail-closed promote() path agree on what the candidate is compared against.
    incumbent = registry.gating_incumbent(candidate)
    logger.info(
        "Candidate: %s (league=%s, market=%s, method=%s)",
        candidate.profile_id,
        candidate.league,
        candidate.market or "game",
        candidate.method,
    )
    if incumbent is not None:
        logger.info(
            "Incumbent: %s (market=%s, method=%s)",
            incumbent.profile_id,
            incumbent.market or "game",
            incumbent.method,
        )
    else:
        logger.info(
            "Incumbent: NONE for league=%s market=%s",
            candidate.league,
            candidate.market or "game",
        )

    report = evaluate_promotion_gates(
        candidate,
        incumbent,
        min_samples=args.min_samples,
        brier_improvement=args.brier_improvement,
        log_loss_tol=args.log_loss_tol,
        ece_floor=args.ece_floor,
        confirm_backtest_parity=args.confirm_backtest_parity,
        confirm_clv_non_regression=args.confirm_clv_non_regression,
    )
    logger.info("Gate status:")
    _print_gates(report)

    if not args.auto:
        if report.passed:
            logger.info("All gates pass. Re-run with --auto to perform the promotion.")
            return 0
        logger.info("Not all gates pass. Address the failures above; there is no --force override.")
        return 1

    # --auto: perform the promotion through the fail-closed registry path. The
    # registry re-evaluates the same gates and is the single source of truth.
    try:
        registry.promote(
            candidate.profile_id,
            confirm_backtest_parity=args.confirm_backtest_parity,
            confirm_clv_non_regression=args.confirm_clv_non_regression,
            min_samples=args.min_samples,
            brier_improvement=args.brier_improvement,
            log_loss_tol=args.log_loss_tol,
            ece_floor=args.ece_floor,
        )
    except PromotionGateError as exc:
        logger.error("Not promoting: %s", exc)
        return 1
    logger.info("Promoted %s to PRODUCTION (--auto, all gates green).", candidate.profile_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
