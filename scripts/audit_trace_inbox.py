#!/usr/bin/env python
"""
Audit and recover stuck traces in var/inbox/traces/processed/.

Prevents traces from accumulating in subdirectories by:
1. Detecting traces that have been processed but not ingested
2. Moving them back to the root inbox for ingest
3. Reporting stats on recovered traces

Run periodically (e.g., post-session or via cron) to keep the pipeline clean.

Usage:
    python scripts/audit_trace_inbox.py --report
    python scripts/audit_trace_inbox.py --recover
    python scripts/audit_trace_inbox.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("audit_trace_inbox")


def _count_traces_by_league(traces_dir: Path) -> dict[str, int]:
    """Count traces in a directory by league."""
    by_league: dict[str, int] = defaultdict(int)
    for trace_file in traces_dir.glob("*.json"):
        try:
            with open(trace_file) as f:
                data = json.load(f)
            trace = data.get("trace", data)
            league = trace.get("input_snapshot", {}).get("league") or trace.get("league")
            if league:
                by_league[league] += 1
        except (json.JSONDecodeError, OSError):
            pass
    return dict(by_league)


def audit_inbox(inbox_root: Path) -> dict[str, Any]:
    """Audit the trace inbox and subdirectories."""
    traces_dir = inbox_root / "traces"
    root_traces = traces_dir.glob("*.json")
    root_count = len(list(root_traces))

    processed_dir = traces_dir / "processed"
    processed_traces = list(processed_dir.glob("*.json")) if processed_dir.exists() else []

    failed_dir = traces_dir / "failed"
    failed_traces = list(failed_dir.glob("*.json")) if failed_dir.exists() else []

    root_by_league = _count_traces_by_league(traces_dir)
    processed_by_league = _count_traces_by_league(processed_dir) if processed_dir.exists() else {}
    failed_by_league = _count_traces_by_league(failed_dir) if failed_dir.exists() else {}

    return {
        "root_traces_count": root_count,
        "root_traces_by_league": root_by_league,
        "processed_traces_count": len(processed_traces),
        "processed_traces_by_league": processed_by_league,
        "failed_traces_count": len(failed_traces),
        "failed_traces_by_league": failed_by_league,
        "processed_dir": str(processed_dir),
        "failed_dir": str(failed_dir),
        "at_risk": len(processed_traces) > 0 or len(failed_traces) > 0,
    }


def recover_traces(inbox_root: Path, dry_run: bool = False) -> dict[str, Any]:
    """Move stuck traces from processed/ back to root for ingest."""
    traces_dir = inbox_root / "traces"
    processed_dir = traces_dir / "processed"

    if not processed_dir.exists():
        return {"recovered": 0, "status": "processed/ does not exist"}

    processed_files = list(processed_dir.glob("*.json"))
    if not processed_files:
        return {"recovered": 0, "status": "no traces in processed/"}

    if dry_run:
        logger.info(f"DRY-RUN: would recover {len(processed_files)} traces from processed/")
        return {"recovered": len(processed_files), "status": "dry-run", "files": [f.name for f in processed_files]}

    # Move files back to root
    recovered = 0
    for trace_file in processed_files:
        try:
            dest = traces_dir / trace_file.name
            trace_file.rename(dest)
            recovered += 1
        except Exception as e:
            logger.error(f"Failed to recover {trace_file.name}: {e}")

    logger.info(f"Recovered {recovered} traces from processed/")
    return {"recovered": recovered, "status": "success"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and recover stuck traces in var/inbox/traces/processed/."
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=Path("var/inbox"),
        help="Path to inbox directory (default: var/inbox)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print audit report and exit",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Move stuck traces from processed/ back to root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be recovered without making changes",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.inbox.exists():
        logger.error(f"Inbox directory not found: {args.inbox}")
        return 1

    audit_result = audit_inbox(args.inbox)

    if args.report or (not args.recover and not args.dry_run):
        # Print report
        print("\nTRACE INBOX AUDIT")
        print("=" * 70)
        print(f"Root inbox traces:      {audit_result['root_traces_count']}")
        if audit_result["root_traces_by_league"]:
            for league, count in sorted(audit_result["root_traces_by_league"].items()):
                print(f"  {league}: {count}")

        print(f"\nProcessed (stuck):      {audit_result['processed_traces_count']}")
        if audit_result["processed_traces_by_league"]:
            for league, count in sorted(audit_result["processed_traces_by_league"].items()):
                print(f"  {league}: {count}")

        print(f"\nFailed:                 {audit_result['failed_traces_count']}")
        if audit_result["failed_traces_by_league"]:
            for league, count in sorted(audit_result["failed_traces_by_league"].items()):
                print(f"  {league}: {count}")

        print()
        if audit_result["at_risk"]:
            print("[WARNING] AT RISK: Traces stuck in processed/ or failed/")
            if audit_result["processed_traces_count"] > 0:
                print(f"   Run: python scripts/audit_trace_inbox.py --recover")
        else:
            print("[OK] No traces stuck in subdirectories")
        print("=" * 70)

    if args.recover:
        result = recover_traces(args.inbox, dry_run=args.dry_run)
        logger.info(f"Recovery result: {result}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
