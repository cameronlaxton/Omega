"""omega-backtest-parity — candidate-vs-incumbent calibration quality on a holdout.

Evaluates both profiles on the same graded historical-replay pairs and recommends
promotion only when the candidate improves Brier without materially regressing ECE
— i.e. it refuses an ECE-only "win". ROI/CLV non-regression is NOT recomputed here
(no duplicate betting math): use the existing walk-forward backtest report
(omega-run-walk-forward-backtest) for ROI/CLV and confirm it at promotion time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from omega.core.calibration.fitter import CalibrationFitter
from omega.core.calibration.registry import CalibrationRegistry
from omega.trace.store import TraceStore

logger = logging.getLogger("omega.ops.backtest_parity")

DEFAULT_BRIER_IMPROVEMENT = 0.01
DEFAULT_ECE_TOL = 0.02


def _extract(fitter: CalibrationFitter, graded: list[dict], plane: str):
    if plane == "prop":
        return fitter.extract_prop_pairs(graded)
    if plane == "draw":
        return fitter.extract_draw_pairs(graded)
    return fitter.extract_pairs(graded)


def evaluate_backtest_parity(
    graded: list[dict],
    candidate_profile,
    incumbent_profile=None,
    *,
    plane: str = "game",
    brier_improvement: float = DEFAULT_BRIER_IMPROVEMENT,
    ece_tol: float = DEFAULT_ECE_TOL,
) -> dict:
    """Compare candidate vs incumbent calibration metrics on the same graded pairs."""
    fitter = CalibrationFitter()
    preds, outs = _extract(fitter, graded, plane)
    n_eval = len(preds)

    recommend = True
    reasons: list[str] = []
    if n_eval == 0:
        return {
            "schema_version": 1,
            "plane": plane,
            "n_eval": 0,
            "candidate": None,
            "incumbent": None,
            "recommend_promotion": False,
            "reasons": ["no eval pairs"],
        }

    candidate = fitter.evaluate(candidate_profile, preds, outs)
    incumbent = fitter.evaluate(incumbent_profile, preds, outs) if incumbent_profile else None

    if incumbent is not None:
        if (incumbent["brier_score"] - candidate["brier_score"]) < brier_improvement:
            recommend = False
            reasons.append("brier_not_improved")
        if (candidate["calibration_error"] - incumbent["calibration_error"]) > ece_tol:
            recommend = False
            reasons.append("ece_regressed")
    else:
        recommend = False
        reasons.append("no_incumbent_baseline")

    return {
        "schema_version": 1,
        "plane": plane,
        "n_eval": n_eval,
        "candidate": candidate,
        "incumbent": incumbent,
        "recommend_promotion": recommend,
        "reasons": reasons,
    }


def _find_profile(registry: CalibrationRegistry, profile_id: str):
    for p in registry.list_profiles():
        if p.profile_id == profile_id:
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Candidate-vs-incumbent calibration parity on graded replay pairs."
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--incumbent-id", default=None, help="Default: registry gating incumbent.")
    parser.add_argument("--league", required=True)
    parser.add_argument("--plane", choices=["game", "prop", "draw"], default="game")
    parser.add_argument("--historical-db", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    registry = CalibrationRegistry()
    candidate = _find_profile(registry, args.candidate_id)
    if candidate is None:
        logger.error("unknown candidate profile_id=%s", args.candidate_id)
        return 2
    if args.incumbent_id:
        incumbent = _find_profile(registry, args.incumbent_id)
        if incumbent is None:
            logger.warning(
                "--incumbent-id=%s not found in registry; proceeding without incumbent baseline.",
                args.incumbent_id,
            )
    else:
        incumbent = registry.gating_incumbent(candidate)

    store = TraceStore(db_path=args.historical_db)
    try:
        graded = store.query_traces(
            league=args.league,
            execution_mode="historical_replay",
            has_outcome=True,
            calibration_eligible_only=True,
            limit=1_000_000,
        )
    finally:
        store.close()

    report = evaluate_backtest_parity(graded, candidate, incumbent, plane=args.plane)
    print(json.dumps(report, indent=2))
    return 0 if report["recommend_promotion"] else 1


if __name__ == "__main__":
    sys.exit(main())
