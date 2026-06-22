"""omega-cv-calibration-diagnostic — cross-validated calibration-quality report.

The production fit (``omega-fit-calibration``) evaluates a candidate's ECE on a
single 80/20 holdout. At a few-hundred-point holdout that estimate carries
±0.005–0.01 noise, so whether a candidate clears the 0.05 ``ECE_FLOOR`` promotion
gate is partly luck-of-the-split. This tool answers the question the single number
cannot: **what is the model's real calibration quality, with a confidence band?**

It runs repeated **stratified k-fold** cross-validation, reusing the *exact* same
calibration primitives the production path uses — ``CalibrationFitter.fit_isotonic``
/ ``fit_shrinkage`` / ``evaluate`` (which calls the same adaptive-ECE estimator the
floor checks against). For every fold it fits on the train folds and scores the
held-out fold, so every sample is scored out-of-sample exactly ``repeats`` times.

For each method it reports across all folds:
  * mean calibrated ECE + 95% CI (normal approx) and the [p2.5, p97.5] band
  * the **sub-floor pass-rate** — fraction of folds whose calibrated ECE ≤ floor
  * mean Brier, and the raw (uncalibrated) ECE for contrast

Reading the result:
  * mean clearly **above** floor, pass-rate low  → genuine residual miscalibration
    (a model/feature problem; more volume will not help).
  * CI **straddles** the floor, pass-rate ≈ 50%  → the model sits *at* the floor and
    single-split pass/fail is noise (a guardrail-design issue, not a bad model).

This script is read-only: it never fits-and-registers or promotes anything.

Usage:
    omega-cv-calibration-diagnostic --league NFL --plane game \
        --historical-only --historical-db var/historical/replay_nfl.db
    omega-cv-calibration-diagnostic --league WORLD_CUP --plane draw \
        --historical-only --historical-db var/historical/replay_world_cup.db \
        --folds 5 --repeats 10

Exit codes:
    0 — diagnostic completed (regardless of whether the model clears the floor)
    1 — fatal error or too few pairs to cross-validate
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.calibration.fitter import (  # noqa: E402
    CalibrationFitter,
    _adaptive_calibration_error,
    stratified_folds,
)
from omega.core.calibration.market import calibration_market_for_plane  # noqa: E402
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("cv_calibration_diagnostic")

_MIN_SAMPLES = int(os.environ.get("OMEGA_MIN_SAMPLES", 30))
_MIN_CV_SAMPLES = (
    _MIN_SAMPLES * 2
)  # need enough that each fold's train split clears the fitter's min samples


@dataclass
class CvResult:
    """Aggregated cross-validated metrics for one method."""

    method: str
    n_pairs: int
    n_folds_total: int
    raw_ece: float
    cal_ece_mean: float
    cal_ece_std: float
    cal_ece_ci95: tuple[float, float]
    cal_ece_band: tuple[float, float]
    brier_mean: float
    pass_rate: float  # fraction of folds with calibrated ECE <= floor


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100]); no numpy dependency needed."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def raw_oos(
    predictions: list[float],
    outcomes: list[int],
    *,
    folds: int,
    repeats: int,
    ece_floor: float,
    base_seed: int,
) -> tuple[float, float]:
    """Out-of-sample raw (uncalibrated) ECE with the *same* per-fold binning the
    calibrated scores use, so raw-vs-calibrated is apples-to-apples. Raw is
    parameter-free, so it cannot overfit — if calibrated ECE is worse than this,
    the calibration step is adding noise, not removing it. Returns (mean, pass_rate)."""
    len(predictions)
    eces: list[float] = []
    for r in range(repeats):
        assignment = stratified_folds(outcomes, folds, base_seed + r)
        for f in range(folds):
            te = assignment[f]
            te_p = [predictions[i] for i in te]
            te_o = [outcomes[i] for i in te]
            if not te_p:
                continue
            eces.append(_adaptive_calibration_error(te_p, te_o))
    if not eces:
        return float("nan"), float("nan")
    mean = sum(eces) / len(eces)
    pass_rate = sum(1 for e in eces if e <= ece_floor) / len(eces)
    return mean, pass_rate


def cross_validate(
    fitter: CalibrationFitter,
    predictions: list[float],
    outcomes: list[int],
    league: str,
    market: str,
    method: str,
    *,
    folds: int,
    repeats: int,
    ece_floor: float,
    base_seed: int,
) -> CvResult:
    """Repeated stratified k-fold CV for one method. Out-of-sample scores only."""
    n = len(predictions)
    fold_eces: list[float] = []
    fold_briers: list[float] = []
    raw_eces: list[float] = []

    for r in range(repeats):
        assignment = stratified_folds(outcomes, folds, base_seed + r)
        for f in range(folds):
            test_idx = set(assignment[f])
            tr_p = [predictions[i] for i in range(n) if i not in test_idx]
            tr_o = [outcomes[i] for i in range(n) if i not in test_idx]
            te_p = [predictions[i] for i in range(n) if i in test_idx]
            te_o = [outcomes[i] for i in range(n) if i in test_idx]
            if len(tr_p) < _MIN_SAMPLES or not te_p:
                continue
            if method == "isotonic":
                profile = fitter.fit_isotonic(tr_p, tr_o, league=league, market=market)
            else:
                profile = fitter.fit_shrinkage(
                    tr_p, tr_o, league=league, market=market, eligible_sample_size=len(tr_p)
                )
            m = fitter.evaluate(profile, te_p, te_o)
            fold_eces.append(m["calibration_error"])
            fold_briers.append(m["brier_score"])
            raw_eces.append(_adaptive_calibration_error(te_p, te_o))

    k_total = len(fold_eces)
    raw_ece = sum(raw_eces) / k_total if k_total else float("nan")
    mean = sum(fold_eces) / k_total if k_total else float("nan")
    var = sum((e - mean) ** 2 for e in fold_eces) / (k_total - 1) if k_total > 1 else 0.0
    std = math.sqrt(var)
    se = std / math.sqrt(k_total) if k_total else float("nan")
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    band = (_percentile(fold_eces, 2.5), _percentile(fold_eces, 97.5))
    brier_mean = sum(fold_briers) / k_total if k_total else float("nan")
    pass_rate = sum(1 for e in fold_eces if e <= ece_floor) / k_total if k_total else float("nan")

    return CvResult(
        method=method,
        n_pairs=n,
        n_folds_total=k_total,
        raw_ece=raw_ece,
        cal_ece_mean=mean,
        cal_ece_std=std,
        cal_ece_ci95=ci,
        cal_ece_band=band,
        brier_mean=brier_mean,
        pass_rate=pass_rate,
    )


def _load_pairs(args: argparse.Namespace) -> tuple[list[float], list[int]]:
    graded: list[dict] = []
    if not args.historical_only:
        store = TraceStore(db_path=args.db, read_only=True)
        log_effective_db(store, logger)
        graded.extend(store.get_graded_traces(league=args.league, limit=100_000))
        store.close()
    if args.historical_only or args.include_historical:
        hstore = TraceStore(db_path=args.historical_db, read_only=True)
        logger.info("historical DB: %s", args.historical_db)
        graded.extend(
            hstore.query_traces(
                league=args.league,
                execution_mode="historical_replay",
                has_outcome=True,
                calibration_eligible_only=True,
                limit=1_000_000,
            )
        )
        hstore.close()

    fitter = CalibrationFitter()
    if args.plane == "draw":
        return fitter.extract_draw_pairs(graded)
    if args.plane == "prop":
        return fitter.extract_prop_pairs(graded)
    return fitter.extract_pairs(graded)


def _verdict(res: CvResult, ece_floor: float, raw_mean: float) -> str:
    """One-line plain-English read of where the model sits vs the floor."""
    lo, hi = res.cal_ece_ci95
    # If the raw probabilities are already under the floor and calibration makes
    # them worse, the fit is overfitting — the right "profile" is no calibration.
    if raw_mean <= ece_floor and res.cal_ece_mean > raw_mean + 0.005:
        return (
            f"OVER-FIT — raw OOS ECE {raw_mean:.4f} already < floor; this method DEGRADES it. "
            "Use method=none (don't calibrate)."
        )
    if hi < ece_floor:
        return "CALIBRATED — mean+CI under floor; promotable on calibration grounds"
    if lo > ece_floor:
        return "MISCALIBRATED — mean-CI above floor; residual is real, volume won't fix it"
    return "AT FLOOR — CI straddles floor; single-split pass/fail is noise (review estimator/floor)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-validated calibration-quality diagnostic (read-only)."
    )
    parser.add_argument("--league", required=True)
    parser.add_argument("--plane", choices=["game", "prop", "draw"], default="game")
    parser.add_argument("--method", choices=["isotonic", "shrinkage", "both"], default="both")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ece-floor", type=float, default=0.05)
    parser.add_argument("--db", default=None, help="Live trace DB (default: production).")
    parser.add_argument("--historical-db", default=None)
    parser.add_argument("--include-historical", action="store_true")
    parser.add_argument("--historical-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.folds <= 0 or args.repeats <= 0:
        raise ValueError("Both --folds and --repeats must be positive integers.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.historical_only and args.include_historical:
        logger.error("--historical-only and --include-historical are mutually exclusive.")
        return 1
    if (args.historical_only or args.include_historical) and not args.historical_db:
        logger.error("--historical-only/--include-historical require --historical-db.")
        return 1

    predictions, outcomes = _load_pairs(args)
    n = len(predictions)
    if n < _MIN_CV_SAMPLES:
        logger.error(
            "Only %d graded %s pairs for %s — need >=%d to cross-validate.",
            n,
            args.plane,
            args.league,
            _MIN_CV_SAMPLES,
        )
        return 1
    if len(set(outcomes)) < 2:
        logger.error("Outcomes are single-class; cannot assess calibration.")
        return 1

    market = calibration_market_for_plane(args.plane)
    # Deterministic base seed from the data + league (repo determinism norm).
    base_seed = int(
        hashlib.sha256(
            (str(sorted(zip([round(p, 6) for p in predictions], outcomes))) + args.league).encode()
        ).hexdigest()[:8],
        16,
    )
    methods = ["isotonic", "shrinkage"] if args.method == "both" else [args.method]
    fitter = CalibrationFitter()

    base_rate = sum(outcomes) / n
    print(
        f"\n{args.league} / {args.plane}  (market={market})  n={n}  "
        f"base_rate={base_rate:.3f}  mean_pred={sum(predictions) / n:.3f}"
    )
    print(
        f"CV: {args.folds}-fold x {args.repeats} repeats, stratified, out-of-sample. floor={args.ece_floor:.3f}"
    )
    print("-" * 104)

    raw_mean, raw_pass = raw_oos(
        predictions,
        outcomes,
        folds=args.folds,
        repeats=args.repeats,
        ece_floor=args.ece_floor,
        base_seed=base_seed,
    )
    print(
        f"{'raw (none)':<10}{'':>9}{raw_mean:>9.4f}{'(no calibration)':>17}{'':>19}{raw_pass:>9.0%}"
    )

    print(
        f"{'method':<10}{'raw_ECE':>9}{'cal_ECE':>9}{'95%CI':>17}{'[p2.5,p97.5]':>19}"
        f"{'<=floor':>9}{'Brier':>9}"
    )
    results: list[CvResult] = []
    for method in methods:
        res = cross_validate(
            fitter,
            predictions,
            outcomes,
            args.league,
            market,
            method,
            folds=args.folds,
            repeats=args.repeats,
            ece_floor=args.ece_floor,
            base_seed=base_seed,
        )
        results.append(res)
        ci_s = f"[{res.cal_ece_ci95[0]:.4f},{res.cal_ece_ci95[1]:.4f}]"
        band_s = f"[{res.cal_ece_band[0]:.4f},{res.cal_ece_band[1]:.4f}]"
        print(
            f"{method:<10}{res.raw_ece:>9.4f}{res.cal_ece_mean:>9.4f}{ci_s:>17}{band_s:>19}"
            f"{res.pass_rate:>9.0%}{res.brier_mean:>9.4f}"
        )
    print("-" * 104)
    for res in results:
        print(f"  {res.method:<10} {_verdict(res, args.ece_floor, raw_mean)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
