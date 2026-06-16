"""omega-validate-historical-dataset — verify a pinned dataset against disk.

Re-hashes the manifest's files and fails closed on any drift unless ``--refresh``
is given. Also reports row counts, date range, and normalized event/outcome/odds
counts so an operator can confirm the dataset before replay.
"""

from __future__ import annotations

import argparse
import logging
import sys

from omega.historical.dataset_manifest import DatasetHashDriftError, verify_manifest
from omega.historical.manifests import (
    load_dataset_manifest,
    load_normalized_dataset,
    save_dataset_manifest,
)

logger = logging.getLogger("omega.ops.validate_historical_dataset")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a pinned historical dataset.")
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-pin file hashes when they drifted (writes a new manifest).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = load_dataset_manifest(args.manifest_id, root=args.root)

    try:
        verified = verify_manifest(manifest, refresh=args.refresh)
    except DatasetHashDriftError as exc:
        logger.error("%s", exc)
        logger.error("Re-run with --refresh to re-pin the dataset intentionally.")
        return 1

    if verified.manifest_id != manifest.manifest_id:
        save_dataset_manifest(verified, root=args.root)
        logger.warning("Dataset re-pinned: new manifest_id=%s", verified.manifest_id)

    try:
        ds = load_normalized_dataset(verified.manifest_id, root=args.root)
        n_events = len(ds["events"])
        n_outcomes = len(ds["outcomes"])
        n_odds = sum(len(v) for v in ds["odds"].values())
    except FileNotFoundError:
        n_events = n_outcomes = n_odds = 0

    logger.info("manifest_id=%s source=%s league=%s", verified.manifest_id, verified.source_name, verified.league)
    logger.info("files=%d total_rows=%d", len(verified.files), verified.total_rows)
    logger.info("date_range=%s..%s", verified.date_range_start, verified.date_range_end)
    logger.info("normalized: events=%d outcomes=%d odds=%d", n_events, n_outcomes, n_odds)
    if verified.limitations:
        logger.info("limitations: %s", "; ".join(verified.limitations))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
