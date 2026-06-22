"""omega-replay-historical-slate — replay a dataset into a backtest TraceStore.

Loads the normalized dataset, runs the deterministic :class:`ReplayEngine`
against an **isolated** backtest DB (``--backtest-db``, never the production DB),
and saves the replay manifest + candidate selections. Outcomes and closing lines
are attached through the existing TraceStore mechanisms; staking is optional.
"""

from __future__ import annotations

import argparse
import logging
import sys

from omega.historical.contracts import ReplayConfig
from omega.historical.manifests import (
    load_dataset_manifest,
    load_normalized_dataset,
    save_replay_manifest,
    save_selections,
)
from omega.historical.replay import ReplayDataset, ReplayEngine
from omega.trace.store import TraceStore, log_effective_db

logger = logging.getLogger("omega.ops.replay_historical_slate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a historical dataset into a backtest DB.")
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument(
        "--backtest-db",
        required=True,
        help="Isolated sqlite path for replayed traces (NOT the production DB).",
    )
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument("--replay-id", default=None, help="Replay run id (default derived)")
    parser.add_argument("--session-id", default="historical-replay")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--n-iterations", type=int, default=1000)
    parser.add_argument("--simulation-backend", default="fast_score")
    parser.add_argument("--enable-staking", action="store_true")
    parser.add_argument("--leakage-policy", choices=["skip", "fail"], default="skip")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = load_dataset_manifest(args.manifest_id, root=args.root)
    ds_parts = load_normalized_dataset(args.manifest_id, root=args.root)
    dataset = ReplayDataset(
        events=ds_parts["events"],
        outcomes=ds_parts["outcomes"],
        odds=ds_parts["odds"],
        extra_context=ds_parts["extra_context"],
        history_override=ds_parts["history_override"],
        prop_markets=ds_parts.get("prop_markets", {}),
        prop_context=ds_parts.get("prop_context", {}),
    )

    config = ReplayConfig(
        dataset_manifest_id=manifest.manifest_id,
        backtest_db_path=args.backtest_db,
        session_id=args.session_id,
        bankroll=args.bankroll,
        n_iterations=args.n_iterations,
        simulation_backend=args.simulation_backend,
        enable_staking=args.enable_staking,
        leakage_policy=args.leakage_policy,
    )
    replay_id = args.replay_id or f"replay_{manifest.manifest_id}"

    store = TraceStore(db_path=args.backtest_db)
    try:
        log_effective_db(store, logger)
        engine = ReplayEngine(store, config)
        result = engine.run(dataset, replay_id=replay_id, league=manifest.league)
    finally:
        store.close()

    save_replay_manifest(result.manifest, root=args.root)
    save_selections(replay_id, result.selections, root=args.root)

    logger.info(
        "Replay %s: persisted=%d skipped=%d selections=%d",
        replay_id,
        result.n_persisted,
        result.n_skipped,
        len(result.selections),
    )
    print(replay_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
