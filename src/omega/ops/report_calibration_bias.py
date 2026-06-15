#!/usr/bin/env python3
"""
omega.ops.report_calibration_bias -- Diagnose simulation bias before tuning.

Aggregates graded traces by context slice and reports ECE and log-loss
for the raw simulation probabilities (pre-calibration).
"""

import argparse
import logging
import sys
from collections import defaultdict
from typing import Any

from omega.core.calibration.context_slices import context_slice_for_trace
from omega.core.calibration.fitter import CalibrationFitter
from omega.core.calibration.sport_family import sport_family_for_league
from omega.trace.store import TraceStore, log_effective_db

logger = logging.getLogger("report_calibration_bias")


def main() -> int:
    p = argparse.ArgumentParser(description="Report raw simulation bias by context slice.")
    p.add_argument("--league", type=str, required=True, help="League to report on (e.g., NBA)")
    p.add_argument("--sport-family", type=str, help="Override sport family")
    p.add_argument("--db", type=str, help="SQLite trace database path")
    p.add_argument("--min-samples", type=int, default=50, help="Minimum samples to report a slice")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    store = TraceStore(args.db)
    log_effective_db(store, logger)
    traces = store.get_recent_traces(league=args.league.upper(), status="graded", limit=100000)

    if not traces:
        logger.error("No graded traces found for league=%s", args.league)
        return 1

    sport_family = args.sport_family or sport_family_for_league(args.league)
    fitter = CalibrationFitter()

    # We want to evaluate the raw simulation probs. We can use CalibrationFitter.evaluate
    # with a dummy base profile (method="isotonic", x=[0,1], y=[0,1] i.e. identity mapping).
    from omega.core.calibration.profiles import CalibrationProfile
    identity_profile = CalibrationProfile(
        profile_id="identity",
        method="isotonic",
        league=args.league,
        version=1,
        dataset_hash="none",
        params={},
        metrics={},
        training_window="0d",
        sample_size=0,
    )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        slice_name = context_slice_for_trace(t, sport_family=sport_family)
        groups[str(slice_name or "base")].append(t)

    print(f"\nSimulation Bias Report: {args.league.upper()} (Family: {sport_family})")
    print(f"{'Slice':<25} | {'N':<7} | {'Brier':<7} | {'ECE':<7} | {'LogLoss':<7}")
    print("-" * 63)

    for slice_name, slice_traces in sorted(groups.items()):
        if len(slice_traces) < args.min_samples:
            continue

        # Extract pairs for the moneyline/game market
        predictions, outcomes, _ = fitter.extract_pairs(slice_traces, "game")
        if not predictions:
            continue

        metrics = fitter.evaluate(identity_profile, predictions, outcomes)
        
        brier = metrics.get("brier_score", 0.0)
        ece = metrics.get("calibration_error", 0.0)
        log_loss = metrics.get("log_loss", 0.0)
        n = metrics.get("n_eval", len(predictions))

        print(f"{slice_name:<25} | {n:<7d} | {brier:<7.4f} | {ece:<7.4f} | {log_loss:<7.4f}")

    print("-" * 63)
    return 0


if __name__ == "__main__":
    sys.exit(main())
