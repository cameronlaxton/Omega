"""omega-run-walk-forward-backtest — chronological walk-forward over replay traces.

Reads replayed traces from an isolated ``--backtest-db``, runs walk-forward
calibration (train strictly before each test window), and writes a
:class:`BacktestReport` separating raw-vs-calibrated probability metrics from
betting ROI.
"""

from __future__ import annotations

import argparse
import logging
import sys

from omega.historical.contracts import WalkForwardConfig
from omega.historical.manifests import (
    load_dataset_manifest,
    load_replay_manifest,
    load_selections,
    save_backtest_report,
)
from omega.historical.walk_forward import run_walk_forward
from omega.trace.store import TraceStore, log_effective_db

logger = logging.getLogger("omega.ops.run_walk_forward_backtest")


def _csv_list(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a walk-forward backtest over replay traces.")
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--backtest-db", required=True, help="Isolated backtest DB (NOT production)")
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument("--replay-id", default=None, help="Replay run id (default derived)")
    parser.add_argument("--mode", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--train-window-days", type=int, default=None)
    parser.add_argument("--test-window-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--min-train-samples", type=int, default=50)
    parser.add_argument("--min-slice-samples", type=int, default=30)
    parser.add_argument("--markets", default="game", help="Comma list, e.g. game,draw")
    parser.add_argument("--slices", default="", help="Comma list of context slices to fit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = load_dataset_manifest(args.manifest_id, root=args.root)
    replay_id = args.replay_id or f"replay_{manifest.manifest_id}"

    try:
        replay_manifest = load_replay_manifest(replay_id, root=args.root)
        replay_records = replay_manifest.records
    except FileNotFoundError:
        replay_records = []
    selections = load_selections(replay_id, root=args.root)

    config = WalkForwardConfig(
        mode=args.mode,
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        min_train_samples=args.min_train_samples,
        min_slice_samples=args.min_slice_samples,
        markets=_csv_list(args.markets) or ["game"],
        slices=_csv_list(args.slices),
    )

    store = TraceStore(db_path=args.backtest_db)
    try:
        log_effective_db(store, logger)
        report = run_walk_forward(
            store,
            config=config,
            league=manifest.league,
            replay_id=replay_id,
            dataset_manifest_id=manifest.manifest_id,
            selections=selections,
            replay_records=replay_records,
        )
    finally:
        store.close()

    out = save_backtest_report(report, root=args.root)
    logger.info(
        "Walk-forward complete: %d folds, %d markets aggregated.",
        len(report.folds),
        len(report.aggregate_metrics_by_market),
    )
    print(str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
