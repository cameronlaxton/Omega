"""omega-build-prop-context - build as-of player-prop context artifacts.

Inputs are local-only: an ingested normalized historical dataset plus a supplied
player-stat CSV. The command writes replay-ready ``prop_context.json`` beside
the dataset and a separate ``prop_context_audit.json`` for coverage checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omega.historical.adapters.csv_player_stats import CsvPlayerStatsAdapter
from omega.historical.manifests import (
    datasets_dir,
    load_dataset_manifest,
    load_normalized_dataset,
)
from omega.historical.prop_context import (
    PropContextBuildConfig,
    build_prop_context,
    targets_from_prop_markets,
)

logger = logging.getLogger("omega.ops.build_prop_context")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build as-of historical player-prop context for replay."
    )
    parser.add_argument("--manifest-id", required=True, help="Ingested dataset manifest id")
    parser.add_argument("--player-stats", required=True, help="Historical player-stat CSV")
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument(
        "--output",
        default=None,
        help="prop_context JSON path (default: dataset/prop_context.json)",
    )
    parser.add_argument(
        "--audit-output",
        default=None,
        help="audit JSON path (default: dataset/prop_context_audit.json)",
    )
    parser.add_argument("--lookback-games", type=int, default=10)
    parser.add_argument("--min-history-games", type=int, default=5)
    parser.add_argument("--stale-days", type=int, default=120)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stats_path = Path(args.player_stats)
    if not stats_path.exists():
        logger.error("player-stats file not found: %s", stats_path)
        return 1

    try:
        manifest = load_dataset_manifest(args.manifest_id, root=args.root)
        ds_parts = load_normalized_dataset(args.manifest_id, root=args.root)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    adapter = CsvPlayerStatsAdapter(manifest.league)
    observations = adapter.read_stat_observations(stats_path)
    targets = targets_from_prop_markets(ds_parts["events"], ds_parts.get("prop_markets", {}))
    if not targets:
        logger.warning(
            "No prop-market targets found for manifest %s; writing empty context.",
            manifest.manifest_id,
        )

    config = PropContextBuildConfig(
        lookback_games=args.lookback_games,
        min_history_games=args.min_history_games,
        stale_days=args.stale_days,
    )
    result = build_prop_context(
        manifest_id=manifest.manifest_id,
        league=manifest.league,
        targets=targets,
        observations=observations,
        config=config,
    )

    dataset_dir = datasets_dir(args.root) / manifest.manifest_id
    context_path = Path(args.output) if args.output else dataset_dir / "prop_context.json"
    audit_path = (
        Path(args.audit_output) if args.audit_output else dataset_dir / "prop_context_audit.json"
    )
    _write_json(context_path, result.context)
    _write_json(audit_path, result.audit.model_dump(mode="json"))

    logger.info(
        "Built prop context for %s: targets=%d missing_rate=%.3f stale_rate=%.3f",
        manifest.manifest_id,
        result.audit.target_count,
        result.audit.missing_context_rate,
        result.audit.stale_context_rate,
    )
    print(str(context_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
