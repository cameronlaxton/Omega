"""
scripts/refresh_wehoop.py — load wehoop WNBA history into backtest artifacts.

Weekly refresh of the wehoop (sportsdataverse) WNBA team-box export into frozen
backtest artifacts used for WNBA replay determinism and calibration. This is the
historical/backtest path; the live in-season fetch path is espn_wnba.py.

Pipeline (all via omega/integrations/wehoop.py on the shared ETL harness):
  1. Fetch the season team-box Parquet (cached under data/cache/wehoop/).
  2. Validate every row (fail loud on schema drift).
  3. Resolve team names through data/aliases/WNBA.json.
  4. Build deterministic FrozenArtifacts (off/def rating + pace + final score).
  5. Write one JSON per artifact to the output directory, versioned by
     FrozenArtifact.schema_version.

Replay-safe: with OMEGA_REPLAY_MODE=1 a cached pull is served, but a cold pull
(network) is blocked by the ETL guard.

Usage:
    python scripts/refresh_wehoop.py --season 2025
    python scripts/refresh_wehoop.py --season 2025 --out data/backtest_artifacts/WNBA
    python scripts/refresh_wehoop.py --season 2025 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.integrations._etl import load_alias_table  # noqa: E402
from omega.integrations.wehoop import load_wnba_artifacts  # noqa: E402

logger = logging.getLogger("refresh_wehoop")

_DEFAULT_OUT = _REPO_ROOT / "data" / "backtest_artifacts" / "WNBA"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh WNBA backtest artifacts from wehoop")
    parser.add_argument("--season", type=int, required=True, help="WNBA season year, e.g. 2025")
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Output directory for FrozenArtifact JSON files",
    )
    parser.add_argument("--cache-root", default=None, help="Override ETL cache root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    alias_table = load_alias_table("WNBA")
    try:
        artifacts, skipped = load_wnba_artifacts(
            args.season, cache_root=args.cache_root, alias_table=alias_table
        )
    except Exception as exc:  # noqa: BLE001 - surface ETL failures loudly
        logger.error("wehoop load failed for season %s: %s", args.season, exc)
        return 1

    logger.info(
        "Built %d WNBA artifacts for season %s (skipped %d).",
        len(artifacts),
        args.season,
        len(skipped),
    )
    for line in skipped:
        logger.warning("  skipped %s", line)

    if args.dry_run:
        logger.info("Dry run — not writing artifacts.")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for artifact in artifacts:
        path = out_dir / f"{artifact.artifact_id}.json"
        path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        written += 1

    logger.info("Wrote %d artifact JSON files to %s", written, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
