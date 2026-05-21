"""
scripts/fetch_outcomes_all.py — attach outcomes for all supported leagues.

Dispatches fetch_outcomes_nba.py, fetch_outcomes_mlb.py, and
fetch_outcomes_props.py in sequence. All runs are idempotent.

Usage:
    python scripts/fetch_outcomes_all.py
    python scripts/fetch_outcomes_all.py --since 2026-05-10 --until 2026-05-14
    python scripts/fetch_outcomes_all.py --dry-run
    python scripts/fetch_outcomes_all.py --leagues nba props

Exit codes:
    0 — all dispatched successfully
    1 — at least one sub-script failed
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("fetch_outcomes_all")

_SCRIPTS: dict[str, Path] = {
    "nba": _REPO_ROOT / "scripts" / "fetch_outcomes_nba.py",
    "mlb": _REPO_ROOT / "scripts" / "fetch_outcomes_mlb.py",
    "props": _REPO_ROOT / "scripts" / "fetch_outcomes_props.py",
}

_DEFAULT_LEAGUES = ("nba", "mlb", "props")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach outcomes for all leagues. Idempotent — safe to re-run."
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=sorted(_SCRIPTS),
        default=list(_DEFAULT_LEAGUES),
        help="Which leagues to process (default: all)",
    )
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
        script = _SCRIPTS[league]
        if not script.exists():
            logger.warning("Script not found, skipping: %s", script)
            continue

        cmd = [sys.executable, str(script)]
        if args.since:
            cmd += ["--since", args.since]
        if args.until:
            cmd += ["--until", args.until]
        if args.dry_run:
            cmd.append("--dry-run")

        logger.info("Running %s: %s", league, " ".join(cmd))
        if args.dry_run and not script.exists():
            continue

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
