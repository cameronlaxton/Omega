"""omega-report-replay-leakage — aggregate a replay run's data-quality issues.

Summarizes leakage skip reasons, identity failures, and missing-odds counts from
a replay manifest so unsafe or low-quality rows are visible before trusting a
backtest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from omega.historical.manifests import load_dataset_manifest, load_replay_manifest

logger = logging.getLogger("omega.ops.report_replay_leakage")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate replay leakage/identity/odds issues.")
    parser.add_argument("--replay-id", default=None)
    parser.add_argument(
        "--manifest-id", default=None, help="Derive replay-id from manifest if unset"
    )
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    replay_id = args.replay_id
    if replay_id is None:
        if not args.manifest_id:
            logger.error("provide --replay-id or --manifest-id")
            return 1
        manifest = load_dataset_manifest(args.manifest_id, root=args.root)
        replay_id = f"replay_{manifest.manifest_id}"

    manifest = load_replay_manifest(replay_id, root=args.root)
    records = manifest.records
    n = len(records)

    status_counts: Counter = Counter(r.leakage_status for r in records)
    reason_counts: Counter = Counter()
    for r in records:
        reason_counts.update(r.leakage_reasons)
    identity_failures = sum(1 for r in records if r.identity_status == "missing")
    missing_odds = sum(1 for r in records if r.missing_odds)
    default_context = sum(1 for r in records if r.context_source == "default")
    stale_context = sum(1 for r in records if r.is_stale)

    summary = {
        "replay_id": replay_id,
        "n_events": n,
        "leakage_status_counts": dict(status_counts),
        "leakage_reason_counts": dict(reason_counts),
        "identity_failure_count": identity_failures,
        "missing_odds_count": missing_odds,
        "default_context_count": default_context,
        "stale_context_count": stale_context,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Replay leakage report — {replay_id}")
    print(f"  events: {n}")
    print(f"  leakage status: {dict(status_counts)}")
    if reason_counts:
        print(f"  leakage reasons: {dict(reason_counts)}")
    print(f"  identity failures: {identity_failures}")
    print(f"  missing odds: {missing_odds}")
    print(f"  default context: {default_context}   stale context: {stale_context}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
