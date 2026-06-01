"""
omega-fit-calibration â€” fit calibration profile candidates from graded traces.

Loads graded traces from var/omega_traces.db, splits deterministically into train/holdout,
fits the requested method(s) on the train split, evaluates each on the holdout, and
registers the resulting CalibrationProfile(s) as CANDIDATE in the registry.

This script DOES NOT promote profiles. Use omega-promote-profile for that.

Determinism: the train/holdout split is seeded from the deterministic dataset_hash
(sha256 of sorted prediction+outcome pairs), so re-running on the same DB snapshot
produces the same split and the same metrics. This satisfies the CLAUDE.md required
invariant: "the same frozen quant artifact must always produce the same simulation
seed" â€” extended here to "same data â†’ same calibration fit".

Usage:
    omega-fit-calibration --league NBA
    omega-fit-calibration --league NBA --plane prop
    omega-fit-calibration --league NBA --method isotonic
    omega-fit-calibration --league NBA --method both --min-samples 100
    omega-fit-calibration --league NBA --dry-run

Exit codes:
    0 â€” at least one candidate registered (or dry-run completed)
    1 â€” fatal error or no candidates met --min-samples
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.calibration.fitter import CalibrationFitter  # noqa: E402
from omega.core.calibration.profiles import CalibrationProfile  # noqa: E402
from omega.core.calibration.registry import CalibrationRegistry  # noqa: E402
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("fit_calibration")

_DEFAULT_MIN_SAMPLES = 100
_HOLDOUT_FRAC = 0.20


def _deterministic_split(
    predictions: list[float],
    outcomes: list[int],
    league: str,
) -> tuple[list[float], list[int], list[float], list[int]]:
    """Split (predictions, outcomes) 80/20 with a deterministic seed.

    The seed is sha256(sorted(zip(predictions, outcomes)) + league), so the same
    dataset and league always produce the same split. This is essential for
    reproducibility per CLAUDE.md.
    """
    n = len(predictions)
    pair_hash = hashlib.sha256(
        (str(sorted(zip([round(p, 6) for p in predictions], outcomes))) + league).encode()
    ).hexdigest()
    seed = int(pair_hash[:16], 16)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    holdout_size = max(1, int(n * _HOLDOUT_FRAC))
    holdout_idx = set(indices[:holdout_size])

    train_p, train_o, hold_p, hold_o = [], [], [], []
    for i, (p, o) in enumerate(zip(predictions, outcomes)):
        if i in holdout_idx:
            hold_p.append(p)
            hold_o.append(o)
        else:
            train_p.append(p)
            train_o.append(o)
    return train_p, train_o, hold_p, hold_o


def _next_version(registry: CalibrationRegistry, league: str, method: str, market: str) -> int:
    """Return the next monotonic version number for (league, method, market)."""
    existing = registry.list_profiles(league=league)
    same = [p for p in existing if p.method == method and (p.market or "game") == market]
    return (max((p.version for p in same), default=0)) + 1


def _unique_profile_id(
    method: str, league: str, version: int, dataset_hash: str, market: str = "game"
) -> str:
    """Build a profile_id that will not collide with any prior fit.

    Format: <method-prefix>_<league>[_<market>]_v<version>_<short_hash>
    Example: iso_nba_v3_4f7a9d2b1c3e5f0a, iso_epl_draw_v1_4f7a9d2b1c3e5f0a
    """
    prefix = {"isotonic": "iso", "shrinkage": "shrink"}.get(method, method)
    market_tag = "" if market == "game" else f"{market}_"
    return f"{prefix}_{league.lower()}_{market_tag}v{version}_{dataset_hash[:16]}"


def fit_and_register(
    fitter: CalibrationFitter,
    registry: CalibrationRegistry,
    league: str,
    method: str,
    train_p: list[float],
    train_o: list[int],
    hold_p: list[float],
    hold_o: list[int],
    dry_run: bool,
    market: str = "game",
) -> CalibrationProfile:
    """Fit one method, evaluate on holdout, register as CANDIDATE. Returns profile."""
    if method == "isotonic":
        profile = fitter.fit_isotonic(train_p, train_o, league=league, market=market)
    elif method == "shrinkage":
        profile = fitter.fit_shrinkage(train_p, train_o, league=league, market=market, eligible_sample_size=len(train_p))
    else:
        raise ValueError(f"Unknown method: {method!r}")

    metrics = fitter.evaluate(profile, hold_p, hold_o)
    profile.metrics = metrics

    version = _next_version(registry, league, method, market)
    profile.version = version
    profile.profile_id = _unique_profile_id(
        method=method,
        league=league,
        version=version,
        dataset_hash=profile.dataset_hash,
        market=market,
    )

    if not dry_run:
        registry.register(profile)

    return profile


def _plane_market(plane: str) -> str:
    """Map a fitting plane to its calibration market.

    'draw' is its own market plane (3-way draw probabilities). 'game' and
    'prop' both calibrate on the 'game' market (current behaviour)."""
    return "draw" if plane == "draw" else "game"


def _extract_plane_pairs(
    fitter: CalibrationFitter,
    graded: list[dict[str, object]],
    plane: str,
) -> tuple[list[float], list[int], str]:
    if plane == "prop":
        predictions, outcomes = fitter.extract_prop_pairs(graded)
        return predictions, outcomes, "prop probability/outcome"
    if plane == "draw":
        predictions, outcomes = fitter.extract_draw_pairs(graded)
        return predictions, outcomes, "draw_prob/outcome"
    predictions, outcomes = fitter.extract_pairs(graded)
    return predictions, outcomes, "home_win_prob/outcome"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit calibration profile candidates from graded traces."
    )
    parser.add_argument("--league", required=True, help="League code (e.g. NBA)")
    parser.add_argument(
        "--plane",
        choices=["game", "prop", "draw"],
        default="game",
        help=(
            "Calibration plane to fit: game uses home_win_prob pairs; prop uses "
            "graded prop pairs; draw uses 3-way draw_prob pairs (soccer, hockey "
            "regulation) and registers a market='draw' profile."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["isotonic", "shrinkage", "both"],
        default="both",
        help="Which method(s) to fit (default: both)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=_DEFAULT_MIN_SAMPLES,
        help=f"Refuse to fit with fewer total graded samples (default: {_DEFAULT_MIN_SAMPLES})",
    )
    parser.add_argument("--db", type=str, default=None, help="SQLite path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Fit and evaluate but do not register"
    )
    parser.add_argument(
        "--include-backfilled",
        action="store_true",
        help=(
            "Include traces with backfilled or missing identity in calibration slice fitting. "
            "By default these are excluded because recovered identity fields are derived "
            "metadata, not original request provenance."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = TraceStore(db_path=args.db)
    log_effective_db(store, logger)
    graded = store.get_graded_traces(league=args.league, limit=100_000)
    store.close()

    if not args.include_backfilled:
        pre_filter = len(graded)
        graded = [
            t for t in graded
            if (t.get("trace_quality") or t.get("quality_gate") or {}).get("identity_status")
               not in ("missing", "backfilled")
        ]
        excluded = pre_filter - len(graded)
        if excluded:
            logger.warning(
                "Excluded %d trace(s) with missing/backfilled identity from calibration fitting. "
                "Pass --include-backfilled to include them.",
                excluded,
            )

    if not graded:
        logger.error("No graded traces found for league=%s", args.league)
        return 1

    fitter = CalibrationFitter()
    predictions, outcomes, pair_label = _extract_plane_pairs(fitter, graded, args.plane)

    if len(predictions) < args.min_samples:
        logger.error(
            "Refusing to fit %s plane: only %d graded %s pairs available, "
            "minimum %d required. Run with --min-samples to override.",
            args.plane,
            len(predictions),
            pair_label,
            args.min_samples,
        )
        return 1

    train_p, train_o, hold_p, hold_o = _deterministic_split(predictions, outcomes, args.league)
    logger.info(
        "Loaded %d graded %s pairs for %s plane; split %d train / %d holdout (deterministic).",
        len(predictions),
        pair_label,
        args.plane,
        len(train_p),
        len(hold_p),
    )

    methods = ["isotonic", "shrinkage"] if args.method == "both" else [args.method]
    market = _plane_market(args.plane)
    registry = CalibrationRegistry()
    registered = []

    for method in methods:
        try:
            profile = fit_and_register(
                fitter=fitter,
                registry=registry,
                league=args.league,
                method=method,
                train_p=train_p,
                train_o=train_o,
                hold_p=hold_p,
                hold_o=hold_o,
                dry_run=args.dry_run,
                market=market,
            )
        except ValueError as exc:
            logger.error("Failed to fit %s: %s", method, exc)
            continue

        registered.append(profile)
        logger.info(
            "%s candidate %s: brier=%.4f ece=%.4f log_loss=%.4f n_eval=%d",
            "DRY-RUN" if args.dry_run else "Registered",
            profile.profile_id,
            profile.metrics.get("brier_score", -1),
            profile.metrics.get("calibration_error", -1),
            profile.metrics.get("log_loss", -1),
            profile.metrics.get("n_eval", 0),
        )

    if not registered:
        logger.error("No profiles registered (all fits failed).")
        return 1

    logger.info(
        "Done. %d candidate(s) %s.", len(registered), "evaluated" if args.dry_run else "registered"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())




