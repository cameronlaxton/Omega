#!/usr/bin/env python3
"""
omega.ops.report_input_quality -- Audit and report data gaps before slicing.

Analyzes raw traces to identify missing context labels or market data
that would degrade calibration slice resolution.
"""

import argparse
import logging
import sys
from collections import defaultdict

from omega.trace.store import TraceStore, log_effective_db

logger = logging.getLogger("report_input_quality")


def main() -> int:
    p = argparse.ArgumentParser(description="Report data gaps in input traces")
    p.add_argument("--league", type=str, required=True, help="League to report on")
    p.add_argument("--db", type=str, help="SQLite trace database path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    store = TraceStore(args.db)
    log_effective_db(store, logger)
    traces = store.get_recent_traces(league=args.league.upper(), limit=100000)

    if not traces:
        logger.error("No traces found for league=%s", args.league)
        return 1

    missing_counts = defaultdict(int)
    total = len(traces)

    for t in traces:
        labels = t.get("context_labels")
        if not labels:
            missing_counts["missing_context_labels"] += 1

        market = t.get("market")
        if not market:
            missing_counts["missing_market"] += 1
        
        home_ctx = t.get("home_context")
        away_ctx = t.get("away_context")
        if not home_ctx or not away_ctx:
            missing_counts["missing_team_context"] += 1

    print(f"\nInput Quality Report: {args.league.upper()} (N={total})")
    print("-" * 50)
    for k, v in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (v / total) * 100
        print(f"{k:<30} | {v:>7d} | {pct:>5.1f}%")
    print("-" * 50)
    
    if len(missing_counts) == 0:
        print("No missing key inputs detected.")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
