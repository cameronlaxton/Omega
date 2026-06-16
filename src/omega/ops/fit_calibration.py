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
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.calibration.context_slices import (  # noqa: E402
    BASE_CONTEXT_SLICE,
    INITIAL_CONTEXT_SLICES,
    context_slice_for_trace,
)
from omega.core.calibration.fitter import CalibrationFitter  # noqa: E402
from omega.core.calibration.market import calibration_market_for_plane  # noqa: E402
from omega.core.calibration.profiles import CalibrationProfile  # noqa: E402
from omega.core.calibration.registry import CalibrationRegistry  # noqa: E402
from omega.core.calibration.sport_family import sport_family_for_league  # noqa: E402
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
    method: str, league: str, version: int, dataset_hash: str, market: str = "game", context_slice: str | None = None
) -> str:
    """Build a profile_id that will not collide with any prior fit.

    Format: <method-prefix>_<league>[_<market>][_<slice>]_v<version>_<short_hash>
    """
    prefix = {"isotonic": "iso", "shrinkage": "shrink"}.get(method, method)
    market_tag = "" if market == "game" else f"{market}_"
    slice_tag = f"{context_slice}_" if context_slice else ""
    return f"{prefix}_{league.lower()}_{market_tag}{slice_tag}v{version}_{dataset_hash[:16]}"


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
    context_slice: str | None = None,
    training_window: str | None = None,
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
    profile.context_slice = context_slice
    if training_window:
        # Date-windowed fit: record the train window for auditability (AGENTS.md:
        # every calibration fit attributable to a specific dataset + window).
        profile.training_window = training_window

    version = _next_version(registry, league, method, market)
    profile.version = version
    profile.profile_id = _unique_profile_id(
        method=method,
        league=league,
        version=version,
        dataset_hash=profile.dataset_hash,
        market=market,
        context_slice=context_slice,
    )

    if not dry_run:
        registry.register(profile)

    return profile


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


def _decision_date(trace: dict[str, Any]) -> str:
    """Event decision date for windowing.

    Historical-replay traces carry the event's ``decision_time`` (the replay run
    timestamp is irrelevant for leakage windows); live traces fall back to the
    analysis ``timestamp`` (taken near game time).
    """
    return str(trace.get("decision_time") or trace.get("timestamp") or "")[:10]


def _in_window(date: str, start: str | None, end: str | None) -> bool:
    if not date:
        return False
    if start and date < start:
        return False
    if end and date > end:
        return False
    return True


def _load_graded_traces(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load graded traces from the live DB, the historical DB, or both.

    Historical traces are selected by ``execution_mode='historical_replay'`` and
    must already be calibration-eligible + graded. Production ``--db`` is read
    only when not ``--historical-only``.
    """
    graded: list[dict[str, Any]] = []
    if not args.historical_only:
        store = TraceStore(db_path=args.db)
        log_effective_db(store, logger)
        graded.extend(store.get_graded_traces(league=args.league, limit=100_000))
        store.close()
    if args.historical_only or args.include_historical:
        hstore = TraceStore(db_path=args.historical_db)
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
    return graded


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--shadow-min-samples", type=int, default=30, help="Minimum samples to fit a shadow profile")
    parser.add_argument("--context-slice", type=str, default=None, help="Fit specific context slice only")
    parser.add_argument("--all-context-slices", action="store_true", help="Fit all canonical slices")
    parser.add_argument("--include-base-with-slices", action="store_true", help="Fit base slice when --all-context-slices is used")
    parser.add_argument("--sport-family", type=str, default=None, help="Override sport family for context resolution")
    parser.add_argument("--db", type=str, default=None, help="SQLite path (live trace DB)")
    parser.add_argument(
        "--historical-db",
        type=str,
        default=None,
        help="Dedicated historical-replay DB to include/fit from (execution_mode=historical_replay).",
    )
    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="Union live (--db) + historical (--historical-db) graded traces.",
    )
    parser.add_argument(
        "--historical-only",
        action="store_true",
        help="Fit ONLY from --historical-db (ignore live --db traces).",
    )
    parser.add_argument("--train-start", default=None, help="Train window start YYYY-MM-DD (incl.)")
    parser.add_argument("--train-end", default=None, help="Train window end YYYY-MM-DD (incl.)")
    parser.add_argument(
        "--holdout-start", default=None, help="Holdout window start YYYY-MM-DD (incl.)"
    )
    parser.add_argument("--holdout-end", default=None, help="Holdout window end YYYY-MM-DD (incl.)")
    parser.add_argument(
        "--allow-same-season-shadow",
        action="store_true",
        help=(
            "Permit overlapping train/holdout windows (shadow diagnostics only). "
            "Such fits must never be promoted."
        ),
    )
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
    args = parser.parse_args(argv)

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

    date_window = any(
        [args.train_start, args.train_end, args.holdout_start, args.holdout_end]
    )
    if date_window:
        if not (args.train_end and args.holdout_start):
            logger.error(
                "date-windowed fit requires at least --train-end and --holdout-start."
            )
            return 1
        if args.holdout_start < args.train_end and not args.allow_same_season_shadow:
            logger.error(
                "holdout-start (%s) precedes train-end (%s): same-season leakage. Pass "
                "--allow-same-season-shadow for shadow diagnostics only (never promotable).",
                args.holdout_start,
                args.train_end,
            )
            return 1

    graded = _load_graded_traces(args)

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

    sport_family = args.sport_family or sport_family_for_league(args.league)

    # Group traces by requested context slice strategy
    groups: dict[str | None, list[dict[str, Any]]] = {}

    if args.all_context_slices:
        for t in graded:
            slice_val = context_slice_for_trace(t, sport_family=sport_family)
            if slice_val is BASE_CONTEXT_SLICE and not args.include_base_with_slices:
                continue
            # If sport is tennis, we have implicit sub-slices like "surface_clay".
            # For simplicity, if we are doing all slices, we only group by the explicit initial slices
            # plus any discovered surface sub-slices.
            if slice_val is not BASE_CONTEXT_SLICE and slice_val not in INITIAL_CONTEXT_SLICES and not slice_val.startswith("surface_"):
                # We can just include all discovered low cardinality slices if they meet the sample threshold later.
                pass
            groups.setdefault(slice_val, []).append(t)
    elif args.context_slice:
        req_slice = args.context_slice
        for t in graded:
            if context_slice_for_trace(t, sport_family=sport_family) == req_slice:
                groups.setdefault(req_slice, []).append(t)
    else:
        # Existing behaviour: base slice only, filtering out early_market natively if not opted in
        groups[BASE_CONTEXT_SLICE] = [
            t for t in graded if context_slice_for_trace(t, sport_family=sport_family) is BASE_CONTEXT_SLICE
        ]

    methods = ["isotonic", "shrinkage"] if args.method == "both" else [args.method]
    market = calibration_market_for_plane(args.plane)
    fitter = CalibrationFitter()
    registry = CalibrationRegistry()
    registered = []

    # Dry run table header
    if args.dry_run:
        print(f"{'League':<10} | {'Family':<15} | {'Market':<10} | {'Slice':<25} | {'N Pairs':<7} | {'Status':<15} | {'Threshold':<9}")
        print("-" * 110)

    for slice_name, slice_traces in groups.items():
        train_window: str | None = None
        if date_window:
            train_traces = [
                t
                for t in slice_traces
                if _in_window(_decision_date(t), args.train_start, args.train_end)
            ]
            hold_traces = [
                t
                for t in slice_traces
                if _in_window(_decision_date(t), args.holdout_start, args.holdout_end)
            ]
            train_p, train_o, pair_label = _extract_plane_pairs(fitter, train_traces, args.plane)
            hold_p, hold_o, _ = _extract_plane_pairs(fitter, hold_traces, args.plane)
            n = len(train_p) + len(hold_p)
        else:
            predictions, outcomes, pair_label = _extract_plane_pairs(fitter, slice_traces, args.plane)
            n = len(predictions)

        status = "eligible"
        threshold = args.min_samples
        if n < args.shadow_min_samples:
            status = "skipped"
            threshold = args.shadow_min_samples
        elif n < args.min_samples:
            status = "shadow"
            threshold = args.min_samples

        if args.dry_run:
            print(f"{args.league:<10} | {sport_family:<15} | {market:<10} | {str(slice_name or 'base'):<25} | {n:<7} | {status:<15} | {threshold:<9}")
            continue

        if n < args.shadow_min_samples:
            logger.warning(
                "Skipping slice %r: %d graded %s pairs available, minimum %d required.",
                slice_name, n, pair_label, args.shadow_min_samples
            )
            continue

        if n < args.min_samples:
            logger.info("Fitting slice %r as shadow profile (%d pairs).", slice_name, n)

        if date_window:
            if not train_p or not hold_p:
                logger.warning(
                    "Skipping slice %r: date-windowed split produced %d train / %d holdout "
                    "pairs (both must be non-empty).",
                    slice_name, len(train_p), len(hold_p)
                )
                continue
            train_window = f"{args.train_start or 'min'}..{args.train_end}"
        else:
            train_p, train_o, hold_p, hold_o = _deterministic_split(
                predictions, outcomes, args.league
            )
        logger.info(
            "Loaded %d graded %s pairs for %s plane (slice=%r); split %d train / %d holdout.",
            n, pair_label, args.plane, slice_name, len(train_p), len(hold_p)
        )

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
                    context_slice=slice_name,
                    training_window=train_window,
                )
                registered.append(profile)
                logger.info(
                    "Registered candidate %s: brier=%.4f ece=%.4f log_loss=%.4f n_eval=%d",
                    profile.profile_id,
                    profile.metrics.get("brier_score", -1),
                    profile.metrics.get("calibration_error", -1),
                    profile.metrics.get("log_loss", -1),
                    profile.metrics.get("n_eval", 0),
                )
            except ValueError as exc:
                logger.error("Failed to fit %s for slice %r: %s", method, slice_name, exc)

    if args.dry_run:
        return 0

    if not registered:
        logger.error("No profiles registered (all fits failed or were skipped).")
        return 1

    logger.info("Done. %d candidate(s) registered.", len(registered))
    return 0


if __name__ == "__main__":
    sys.exit(main())


