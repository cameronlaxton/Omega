"""
omega.ops.fetch_outcomes_all â€” attach outcomes for all supported leagues.

Dispatches fetch_outcomes_nba.py, fetch_outcomes_wnba.py, fetch_outcomes_mlb.py,
fetch_outcomes_soccer.py, and fetch_outcomes_props.py in sequence. All runs are
idempotent.

Usage:
    omega-fetch-outcomes-all
    omega-fetch-outcomes-all --since 2026-05-10 --until 2026-05-14
    omega-fetch-outcomes-all --dry-run
    omega-fetch-outcomes-all --leagues nba props

Exit codes:
    0 â€” all dispatched successfully
    1 â€” at least one sub-script failed
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logger = logging.getLogger("fetch_outcomes_all")

# Invoke the canonical module entrypoints (python -m omega.ops.fetch_outcomes_*).
# The legacy top-level scripts/ directory was removed; pointing at it made this
# dispatcher a silent no-op (every league "Script not found, skipping").
_MODULES: dict[str, str] = {
    "nba": "omega.ops.fetch_outcomes_nba",
    "wnba": "omega.ops.fetch_outcomes_wnba",
    "mlb": "omega.ops.fetch_outcomes_mlb",
    "soccer": "omega.ops.fetch_outcomes_soccer",
    "props": "omega.ops.fetch_outcomes_props",
}

_DEFAULT_LEAGUES = ("nba", "wnba", "mlb", "soccer", "props")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach outcomes for all leagues. Idempotent â€” safe to re-run."
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=sorted(_MODULES),
        default=list(_DEFAULT_LEAGUES),
        help="Which leagues to process (default: all)",
    )
    parser.add_argument("--db", default=None, help="SQLite path (passed through)")
    parser.add_argument("--since", default=None, help="Start date YYYY-MM-DD (passed through)")
    parser.add_argument("--until", default=None, help="End date YYYY-MM-DD (passed through)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    failures = 0
    for league in args.leagues:
        module = _MODULES[league]
        cmd = [sys.executable, "-m", module]
        if args.db:
            cmd += ["--db", args.db]
        if args.since:
            cmd += ["--since", args.since]
        if args.until:
            cmd += ["--until", args.until]
        if args.dry_run:
            cmd.append("--dry-run")

        logger.info("Running %s: %s", league, " ".join(cmd))
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        if result.returncode != 0:
            failures += 1
            logger.error("%s FAILED (exit=%d)", league, result.returncode)
        else:
            logger.info("%s OK", league)

    if failures:
        logger.error("%d sub-script(s) failed.", failures)
        return 1
    logger.info("All outcome scripts completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())




