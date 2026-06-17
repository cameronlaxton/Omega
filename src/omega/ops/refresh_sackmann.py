"""
omega.ops.refresh_sackmann — rebuild surface-segmented tennis rate priors.

Weekly refresh (Phase 7 M3): reads the Sackmann ATP/WTA match CSVs —
local-first from ``data/tennis/``, otherwise via the ETL cache — computes
12-month half-life SPW%/RPW% per (player, surface), resolves player names
through ``data/aliases/TENNIS.json``, and upserts ``priors_tennis`` rows.

Replay-safe: local/cached files are served under OMEGA_REPLAY_MODE=1; cold
network pulls are blocked by the ETL guard.

Usage:
    omega-refresh-sackmann --tour atp --years 2024,2025,2026
    omega-refresh-sackmann --tour wta --years 2024,2025,2026 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.integrations._etl import load_alias_table  # noqa: E402
from omega.integrations.tennis_sackmann import load_tennis_priors  # noqa: E402

logger = logging.getLogger("refresh_sackmann")


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh priors_tennis from Sackmann CSVs")
    parser.add_argument("--tour", choices=("atp", "wta"), required=True)
    parser.add_argument(
        "--years",
        required=True,
        help="Comma-separated season years, e.g. 2024,2025,2026",
    )
    parser.add_argument("--as-of", default=None, help="as_of_date stamp (default: today)")
    parser.add_argument("--min-matches", type=int, default=3)
    parser.add_argument("--local-root", default=None, help="Override local CSV dir (data/tennis)")
    parser.add_argument("--cache-root", default=None, help="Override ETL cache root")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    years = [int(y.strip()) for y in args.years.split(",")]
    as_of = args.as_of or date.today().isoformat()

    try:
        priors, unresolved = load_tennis_priors(
            args.tour,
            years,
            as_of_date=as_of,
            local_root=args.local_root,
            cache_root=args.cache_root,
            alias_table=load_alias_table("TENNIS"),
            min_matches=args.min_matches,
        )
    except Exception as exc:  # noqa: BLE001 - surface ETL failures loudly
        logger.error("sackmann load failed for %s %s: %s", args.tour, years, exc)
        return 1

    logger.info(
        "%s %s: %d (player, surface) rate rows (%d unresolved player(s) excluded)",
        args.tour.upper(),
        years,
        len(priors),
        len(unresolved),
    )

    if args.dry_run:
        for prior in priors[:10]:
            logger.info(
                "  %-28s %-6s spw=%.3f rpw=%.3f (n=%d)",
                prior.player,
                prior.surface,
                prior.spw_pct,
                prior.rpw_pct,
                prior.n_matches,
            )
        logger.info("Dry run — not writing priors_tennis.")
        return 0

    from omega.trace.priors import upsert_tennis_prior
    from omega.trace.store import TraceStore

    store = TraceStore(db_path=args.db) if args.db else TraceStore()
    try:
        for prior in priors:
            upsert_tennis_prior(store, prior)
    finally:
        store.close()
    logger.info("Wrote %d priors_tennis rows.", len(priors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
