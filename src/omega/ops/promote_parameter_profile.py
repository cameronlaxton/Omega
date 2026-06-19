"""
omega-promote-parameter-profile — promote a CANDIDATE backend parameter profile
to PRODUCTION (Phase 8).

The structural-parameter analogue of ``omega-promote-profile``. Promotion is
fail-closed: it always evaluates the SAME shared gates
(``omega.core.governance.promotion_gates``) against the candidate's RAW
(pre-calibration) held-out metrics and refuses to promote unless ALL pass. There
is no ``--force`` bypass. Exactly one profile per ``(backend_name,
competition_bucket)`` is PRODUCTION at a time; promoting archives the incumbent.

  * default (no --auto): dry-run — print the gate status and exit non-zero if any
    gate fails;
  * --auto: perform the promotion through the fail-closed store path.

Comparison metrics come from the candidate's stored ``metrics`` dict, filled at
fit/sweep time (the variant sweep's sealed-holdout raw metrics).

Usage:
    omega-promote-parameter-profile --list-candidates --backend soccer_bivariate_poisson_dc
    omega-promote-parameter-profile --profile-id <id>
    omega-promote-parameter-profile --profile-id <id> --auto \
        --confirm-backtest-parity --parity-report parity.json \
        --confirm-clv-non-regression --clv-report clv.json

Exit codes:
    0 — promotion succeeded, dry-run with all gates green, or list completed
    1 — at least one gate failed (dry-run) or promotion was blocked (--auto)
    2 — fatal error (unknown profile, wrong status, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.governance.promotion_gates import (  # noqa: E402
    DEFAULT_BRIER_IMPROVEMENT,
    DEFAULT_ECE_FLOOR,
    DEFAULT_LOG_LOSS_TOL,
    DEFAULT_MIN_SAMPLES,
    GateReport,
    PromotionGateError,
    evaluate_promotion_gates,
)
from omega.core.simulation.parameter_profile import ParameterProfileStatus  # noqa: E402
from omega.trace.parameter_profiles import (  # noqa: E402
    get_parameter_profile,
    get_production_parameter_profile,
    list_parameter_profiles,
    promote_parameter_profile,
)
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("promote_parameter_profile")


def _load_report(path: str | None) -> dict | None:
    """Load a parity/CLV report JSON artifact (or None). Fails fast on a bad path."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        logger.error("Parity/CLV report not found: %s", path)
        raise SystemExit(2)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.error("Could not parse parity/CLV report %s: %s", path, exc)
        raise SystemExit(2) from exc
    if not isinstance(data, dict):
        logger.error("Parity/CLV report %s must be a JSON object", path)
        raise SystemExit(2)
    return data


def _print_gates(report: GateReport) -> None:
    for r in report.results:
        logger.info("  [%s] %-18s %s", "PASS" if r.passed else "FAIL", r.name, r.message)


def _list_candidates(store: TraceStore, backend: str | None, bucket: str | None) -> int:
    candidates = list_parameter_profiles(
        store, backend_name=backend, competition_bucket=bucket,
        status=ParameterProfileStatus.CANDIDATE,
    )
    if not candidates:
        logger.info("No CANDIDATE parameter profiles for the given filters.")
        return 0
    logger.info("Found %d CANDIDATE parameter profile(s):", len(candidates))
    for p in candidates:
        m = p.metrics
        logger.info(
            "  %s  backend=%s  bucket=%s  n=%d  raw_brier=%.4f  raw_ece=%.4f  raw_log_loss=%.4f",
            p.profile_id,
            p.backend_name,
            p.competition_bucket,
            p.sample_size,
            m.get("brier_score", float("nan")),
            m.get("calibration_error", float("nan")),
            m.get("log_loss", float("nan")),
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a CANDIDATE backend parameter profile to PRODUCTION (fail-closed)."
    )
    parser.add_argument("--profile-id", help="profile_id of the CANDIDATE to promote")
    parser.add_argument("--auto", action="store_true", help="Perform the promotion (default: dry-run)")
    parser.add_argument("--list-candidates", action="store_true", help="List CANDIDATE profiles and exit")
    parser.add_argument("--backend", help="Filter/scope by backend_name (for --list-candidates)")
    parser.add_argument("--bucket", help="Filter by competition_bucket (for --list-candidates)")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--brier-improvement", type=float, default=DEFAULT_BRIER_IMPROVEMENT)
    parser.add_argument("--log-loss-tol", type=float, default=DEFAULT_LOG_LOSS_TOL)
    parser.add_argument("--ece-floor", type=float, default=DEFAULT_ECE_FLOOR)
    parser.add_argument(
        "--confirm-backtest-parity", action="store_true",
        help="Confirm gate BACKTEST_PARITY. REQUIRES --parity-report (pass-indicating).",
    )
    parser.add_argument(
        "--confirm-clv-non-regression", action="store_true",
        help="Confirm gate CLV_NON_REG. REQUIRES --clv-report (pass-indicating).",
    )
    parser.add_argument("--parity-report", default=None, help="Path to a parity report JSON.")
    parser.add_argument("--clv-report", default=None, help="Path to a CLV/non-regression report JSON.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    parity_evidence = _load_report(args.parity_report)
    clv_evidence = _load_report(args.clv_report)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = TraceStore()

    if args.list_candidates:
        return _list_candidates(store, args.backend, args.bucket)

    if not args.profile_id:
        parser.error("--profile-id is required unless --list-candidates is set")

    candidate = get_parameter_profile(store, args.profile_id)
    if candidate is None:
        logger.error("Parameter profile not found: %s", args.profile_id)
        return 2
    if candidate.status != ParameterProfileStatus.CANDIDATE:
        logger.error(
            "Profile %s has status=%s; only CANDIDATE profiles can be promoted",
            candidate.profile_id, candidate.status.value,
        )
        return 2

    incumbent = get_production_parameter_profile(
        store, candidate.backend_name, candidate.competition_bucket
    )
    logger.info(
        "Candidate: %s (backend=%s, bucket=%s)",
        candidate.profile_id, candidate.backend_name, candidate.competition_bucket,
    )
    logger.info("Incumbent: %s", incumbent.profile_id if incumbent else "NONE")

    report = evaluate_promotion_gates(
        candidate, incumbent,
        min_samples=args.min_samples,
        brier_improvement=args.brier_improvement,
        log_loss_tol=args.log_loss_tol,
        ece_floor=args.ece_floor,
        confirm_backtest_parity=args.confirm_backtest_parity,
        confirm_clv_non_regression=args.confirm_clv_non_regression,
        parity_evidence=parity_evidence,
        clv_evidence=clv_evidence,
    )
    logger.info("Gate status:")
    _print_gates(report)

    if not args.auto:
        if report.passed:
            logger.info("All gates pass. Re-run with --auto to perform the promotion.")
            return 0
        logger.info("Not all gates pass. Address the failures above; there is no --force override.")
        return 1

    try:
        promote_parameter_profile(
            store, candidate.profile_id,
            confirm_backtest_parity=args.confirm_backtest_parity,
            confirm_clv_non_regression=args.confirm_clv_non_regression,
            parity_evidence=parity_evidence,
            clv_evidence=clv_evidence,
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
