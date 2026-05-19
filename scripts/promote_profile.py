"""
scripts/promote_profile.py — promote a CANDIDATE calibration profile to PRODUCTION.

Promotion is gated on five checks. ALL gates must pass for --auto promotion;
absent --auto the script reports the gate status and exits non-zero (use --force
to override after manual review).

Gates:
    1. SAMPLE_SIZE      — candidate.sample_size >= --min-samples (default 100)
    2. BRIER_IMPROVES   — candidate.brier_score is at least --brier-improvement
                          (default 0.01) lower than incumbent.brier_score
    3. LOG_LOSS_NO_REG  — candidate.log_loss is not worse than incumbent.log_loss
                          by more than --log-loss-tol (default 0.005)
    4. BACKTEST_PARITY  — backtest-replay ROI parity within ±0.5%
                          [REQUIRES DATA] (skipped until graded bets accumulate; see notes)
    5. CLV_NON_REG      — mean CLV cents does not regress by more than 0.5 cents
                          [REQUIRES DATA] (skipped until closing-line attachment accumulates)

If no incumbent exists for the league, gates 2 and 3 are auto-passed (any first
profile is acceptable as long as sample size is met).

This script's job is gate enforcement only. It does not fit, does not evaluate
on new data. Re-evaluation against a fresh holdout should be done by
`scripts/fit_calibration.py`. Comparison metrics come from each profile's stored
`metrics` dict (filled at fit time).

Usage:
    python scripts/promote_profile.py --candidate-id iso_nba_v3_4f7a9d2b1c3e5f0a
    python scripts/promote_profile.py --candidate-id <id> --auto
    python scripts/promote_profile.py --candidate-id <id> --force
    python scripts/promote_profile.py --list-candidates --league NBA

Exit codes:
    0 — promotion succeeded (or list completed)
    1 — any gate failed and --force was not supplied
    2 — fatal error (unknown candidate, etc.)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus  # noqa: E402
from omega.core.calibration.registry import CalibrationRegistry  # noqa: E402

logger = logging.getLogger("promote_profile")

_DEFAULT_MIN_SAMPLES = 100
_DEFAULT_BRIER_IMPROVEMENT = 0.01
_DEFAULT_LOG_LOSS_TOL = 0.005


def _evaluate_gates(
    candidate: CalibrationProfile,
    incumbent: CalibrationProfile | None,
    min_samples: int,
    brier_improvement: float,
    log_loss_tol: float,
) -> list[tuple[str, bool, str]]:
    """Return a list of (gate_name, passed, message) tuples.

    Gates 4 and 5 are marked SKIPPED (passed=None becomes False in --auto mode
    unless the operator confirms via the corresponding --confirm-* flag).
    """
    results: list[tuple[str, bool, str]] = []

    # Gate 1: sample size
    n = candidate.sample_size
    passed = n >= min_samples
    results.append(
        ("SAMPLE_SIZE", passed, f"candidate.sample_size={n}, required>={min_samples}")
    )

    # Gate 2: Brier improvement
    cand_brier = candidate.metrics.get("brier_score")
    if incumbent is None:
        results.append(("BRIER_IMPROVES", True, "no incumbent — auto-pass"))
    elif cand_brier is None:
        results.append(("BRIER_IMPROVES", False, "candidate has no brier_score metric"))
    else:
        inc_brier = incumbent.metrics.get("brier_score")
        if inc_brier is None:
            results.append(("BRIER_IMPROVES", True, "incumbent has no brier_score — auto-pass"))
        else:
            improvement = inc_brier - cand_brier
            passed = improvement >= brier_improvement
            results.append(
                (
                    "BRIER_IMPROVES",
                    passed,
                    f"improvement={improvement:.4f}, required>={brier_improvement:.4f} "
                    f"(candidate={cand_brier:.4f}, incumbent={inc_brier:.4f})",
                )
            )

    # Gate 3: log-loss no-regression
    cand_log = candidate.metrics.get("log_loss")
    if incumbent is None:
        results.append(("LOG_LOSS_NO_REG", True, "no incumbent — auto-pass"))
    elif cand_log is None:
        results.append(("LOG_LOSS_NO_REG", False, "candidate has no log_loss metric"))
    else:
        inc_log = incumbent.metrics.get("log_loss")
        if inc_log is None:
            results.append(("LOG_LOSS_NO_REG", True, "incumbent has no log_loss — auto-pass"))
        else:
            regression = cand_log - inc_log
            passed = regression <= log_loss_tol
            results.append(
                (
                    "LOG_LOSS_NO_REG",
                    passed,
                    f"delta={regression:+.4f}, tolerated<={log_loss_tol:.4f} "
                    f"(candidate={cand_log:.4f}, incumbent={inc_log:.4f})",
                )
            )

    return results


def _print_gates(results: list[tuple[str, bool, str]]) -> None:
    for name, passed, msg in results:
        flag = "PASS" if passed else "FAIL"
        logger.info("  [%s] %-18s %s", flag, name, msg)


def _list_candidates(registry: CalibrationRegistry, league: str | None) -> int:
    candidates = registry.list_profiles(league=league, status=ProfileStatus.CANDIDATE.value)
    if not candidates:
        logger.info("No CANDIDATE profiles%s.", f" for league={league}" if league else "")
        return 0
    logger.info("Found %d CANDIDATE profile(s):", len(candidates))
    for p in candidates:
        m = p.metrics
        logger.info(
            "  %s  league=%s  method=%s  n=%d  brier=%.4f  ece=%.4f  log_loss=%.4f",
            p.profile_id,
            p.league,
            p.method,
            p.sample_size,
            m.get("brier_score", float("nan")),
            m.get("calibration_error", float("nan")),
            m.get("log_loss", float("nan")),
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a CANDIDATE calibration profile to PRODUCTION after gate checks."
    )
    parser.add_argument("--candidate-id", help="profile_id of the CANDIDATE to promote")
    parser.add_argument("--auto", action="store_true", help="Promote iff ALL gates pass")
    parser.add_argument("--force", action="store_true", help="Promote regardless of gate status")
    parser.add_argument("--list-candidates", action="store_true", help="List CANDIDATE profiles and exit")
    parser.add_argument("--league", help="Filter for --list-candidates")
    parser.add_argument("--min-samples", type=int, default=_DEFAULT_MIN_SAMPLES)
    parser.add_argument("--brier-improvement", type=float, default=_DEFAULT_BRIER_IMPROVEMENT)
    parser.add_argument("--log-loss-tol", type=float, default=_DEFAULT_LOG_LOSS_TOL)
    parser.add_argument(
        "--confirm-backtest-parity",
        action="store_true",
        help="Mark gate 4 (BACKTEST_PARITY) as passed; operator has manually verified.",
    )
    parser.add_argument(
        "--confirm-clv-non-regression",
        action="store_true",
        help="Mark gate 5 (CLV_NON_REG) as passed; operator has manually verified.",
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

    incumbent = registry.get_production(candidate.league)
    logger.info("Candidate: %s (league=%s, method=%s)", candidate.profile_id, candidate.league, candidate.method)
    if incumbent is not None:
        logger.info("Incumbent: %s (method=%s)", incumbent.profile_id, incumbent.method)
    else:
        logger.info("Incumbent: NONE for league=%s", candidate.league)

    gates = _evaluate_gates(
        candidate=candidate,
        incumbent=incumbent,
        min_samples=args.min_samples,
        brier_improvement=args.brier_improvement,
        log_loss_tol=args.log_loss_tol,
    )

    # Gates 4 and 5 require real graded-bet data and a backtest run; until those
    # exist, the operator can confirm them via explicit flags.
    gates.append(
        (
            "BACKTEST_PARITY",
            args.confirm_backtest_parity,
            "operator-confirmed via --confirm-backtest-parity"
            if args.confirm_backtest_parity
            else "no automated check — confirm with --confirm-backtest-parity after backtest review",
        )
    )
    gates.append(
        (
            "CLV_NON_REG",
            args.confirm_clv_non_regression,
            "operator-confirmed via --confirm-clv-non-regression"
            if args.confirm_clv_non_regression
            else "no automated check — confirm with --confirm-clv-non-regression after CLV review",
        )
    )

    logger.info("Gate status:")
    _print_gates(gates)

    all_pass = all(passed for _, passed, _ in gates)

    if args.force:
        logger.warning("Promoting with --force; gate failures ignored.")
        registry.promote(candidate.profile_id)
        logger.info("Promoted %s to PRODUCTION.", candidate.profile_id)
        return 0

    if args.auto:
        if not all_pass:
            logger.error("Not promoting: one or more gates failed under --auto.")
            return 1
        registry.promote(candidate.profile_id)
        logger.info("Promoted %s to PRODUCTION (--auto, all gates green).", candidate.profile_id)
        return 0

    # No --auto, no --force: report only
    if all_pass:
        logger.info("All gates pass. Re-run with --auto to perform the promotion.")
    else:
        logger.info("Not all gates pass. Address failures or re-run with --force after review.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
