"""omega-report-historical-backtest — render a saved backtest report.

Prints the human-readable summary (raw-vs-calibrated probability metrics and a
separate betting-ROI block) or the raw JSON with ``--json``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from omega.historical.manifests import load_backtest_report, load_dataset_manifest
from omega.historical.reports import to_json, to_text

logger = logging.getLogger("omega.ops.report_historical_backtest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a saved historical backtest report.")
    parser.add_argument("--replay-id", default=None, help="Replay run id")
    parser.add_argument(
        "--manifest-id", default=None, help="Derive replay-id from manifest if unset"
    )
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    replay_id = args.replay_id
    if replay_id is None:
        if not args.manifest_id:
            logger.error("provide --replay-id or --manifest-id")
            return 1
        manifest = load_dataset_manifest(args.manifest_id, root=args.root)
        replay_id = f"replay_{manifest.manifest_id}"

    report = load_backtest_report(replay_id, root=args.root)
    print(to_json(report) if args.json else to_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
