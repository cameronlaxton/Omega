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
    "nhl": "omega.ops.fetch_outcomes_nhl",
    "props": "omega.ops.fetch_outcomes_props",
}

_DEFAULT_LEAGUES = ("nba", "wnba", "mlb", "soccer", "nhl", "props")
_FETCH_OUTCOMES_TIMEOUT_SECONDS = 20 * 60


def _build_cmd(
    league: str,
    *,
    db: str | None,
    since: str | None,
    until: str | None,
    dry_run: bool,
) -> list[str]:
    cmd = [sys.executable, "-m", _MODULES[league]]
    if db:
        cmd += ["--db", db]
    if since:
        cmd += ["--since", since]
    if until:
        cmd += ["--until", until]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_fetch_outcomes(
    *,
    leagues: list[str] | tuple[str, ...] | None = None,
    db: str | None = None,
    since: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    capture_output: bool = False,
    timeout_seconds: int = _FETCH_OUTCOMES_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Dispatch per-league outcome fetchers and return a structured result.

    Shared entrypoint for both the CLI ``main()`` and the ``omega_fetch_outcomes``
    MCP tool. Each league is an idempotent subprocess; ``capture_output=True``
    captures stdout/stderr (tail only) per league instead of streaming, which the
    MCP tool needs to return a JSON-friendly payload.

    To exclude soccer (future-dated fixtures), pass an explicit ``leagues`` list
    without ``"soccer"``.
    """
    selected = list(leagues) if leagues is not None else list(_DEFAULT_LEAGUES)
    unknown = [lg for lg in selected if lg not in _MODULES]
    if unknown:
        raise ValueError(
            f"Unknown league(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(sorted(_MODULES))}"
        )

    results: list[dict[str, object]] = []
    for league in selected:
        cmd = _build_cmd(league, db=db, since=since, until=until, dry_run=dry_run)
        logger.info("Running %s: %s", league, " ".join(cmd))
        try:
            if capture_output:
                proc = subprocess.run(
                    cmd,
                    cwd=_REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
                tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
            else:
                proc = subprocess.run(cmd, cwd=_REPO_ROOT, timeout=timeout_seconds)
                tail = ""
        except subprocess.TimeoutExpired as exc:
            output = exc.output.decode(errors="replace") if isinstance(exc.output, bytes) else (exc.output or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            tail = (output or "")[-2000:] + (stderr or "")[-2000:]
            results.append(
                {
                    "league": league,
                    "returncode": None,
                    "ok": False,
                    "timed_out": True,
                    "timeout_seconds": timeout_seconds,
                    "output_tail": tail,
                }
            )
            logger.error("%s FAILED (timeout=%ss)", league, timeout_seconds)
            continue
        ok = proc.returncode == 0
        results.append(
            {
                "league": league,
                "returncode": proc.returncode,
                "ok": ok,
                "timed_out": False,
                "output_tail": tail,
            }
        )
        logger.info("%s %s", league, "OK" if ok else f"FAILED (exit={proc.returncode})")

    failures = sum(1 for r in results if not r["ok"])
    return {
        "ok": failures == 0,
        "failures": failures,
        "dry_run": dry_run,
        "leagues": selected,
        "results": results,
    }


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

    outcome = run_fetch_outcomes(
        leagues=args.leagues,
        db=args.db,
        since=args.since,
        until=args.until,
        dry_run=args.dry_run,
    )

    if not outcome["ok"]:
        logger.error("%d sub-script(s) failed.", outcome["failures"])
        return 1
    logger.info("All outcome scripts completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())



