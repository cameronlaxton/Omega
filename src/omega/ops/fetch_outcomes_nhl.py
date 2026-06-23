"""
omega.ops.fetch_outcomes_nhl — attach NHL final scores from ESPN to ungraded traces.

Workflow:
    1. Determine the date range (default: yesterday in ET).
    2. For each date, fetch the ESPN scoreboard once.
    3. Query the trace store for ungraded NHL traces in that range.
    4. Match each ungraded trace to a FinalGame by (home_team, away_team) — both
       resolved through the alias table. If both directions match (trace home ==
       ESPN home AND trace away == ESPN away), attach the outcome.
    5. Report unmatched traces so unmapped aliases can be added.

Usage:
    omega-fetch-outcomes-nhl
    omega-fetch-outcomes-nhl --since 2026-05-10 --until 2026-05-14
    omega-fetch-outcomes-nhl --since yesterday
    omega-fetch-outcomes-nhl --dry-run

Exit codes:
    0 — completed (may have unmatched; see log)
    1 — fatal error (e.g., ESPN unreachable)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.integrations.espn_nhl import FinalGame, canonical_team, fetch_scoreboard  # noqa: E402
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("fetch_outcomes_nhl")


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def _parse_date_arg(s: str) -> date:
    """Accept YYYY-MM-DD, 'today', 'yesterday'."""
    s = s.strip().lower()
    if s == "today":
        return datetime.now(UTC).date()
    if s == "yesterday":
        return datetime.now(UTC).date() - timedelta(days=1)
    return date.fromisoformat(s)


def _iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _trace_matchup(trace: dict) -> tuple[str, str] | None:
    """Pull (home_canonical, away_canonical) from a trace dict, or None if unresolvable.

    Looks first at trace['input_snapshot'] (the canonical engine input), then
    falls back to a "{away} @ {home}" matchup string.
    """
    snap = trace.get("input_snapshot") or {}
    home = snap.get("home_team")
    away = snap.get("away_team")
    if home and away:
        c_home = canonical_team(home)
        c_away = canonical_team(away)
        if c_home and c_away:
            return (c_home, c_away)

    matchup = trace.get("matchup") or ""
    if " @ " in matchup:
        away_str, home_str = matchup.split(" @ ", 1)
        c_home = canonical_team(home_str)
        c_away = canonical_team(away_str)
        if c_home and c_away:
            return (c_home, c_away)
    return None


def _match_trace_to_game(
    trace: dict,
    games_by_pair: dict[tuple[str, str], FinalGame],
) -> FinalGame | None:
    pair = _trace_matchup(trace)
    if pair is None:
        return None
    return games_by_pair.get(pair)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    *,
    scoreboard_fetcher=None,
) -> int:
    """CLI entry point.

    ``scoreboard_fetcher`` exists for tests to bypass HTTP: pass a callable
    taking an ISO date string and returning a list of FinalGame. Production
    leaves it None and the live ESPN NHL fetcher is used.
    """
    parser = argparse.ArgumentParser(description="Attach ESPN final scores to NHL traces")
    parser.add_argument(
        "--since", default="yesterday", help="Start date (YYYY-MM-DD | today | yesterday)"
    )
    parser.add_argument("--until", default=None, help="End date inclusive (default = since)")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        start = _parse_date_arg(args.since)
        end = _parse_date_arg(args.until) if args.until else start
    except ValueError as exc:
        logger.error("Bad date arg: %s", exc)
        return 1
    if end < start:
        logger.error("--until (%s) is before --since (%s)", end, start)
        return 1

    sb_fetch = scoreboard_fetcher or fetch_scoreboard

    store = TraceStore(db_path=args.db)
    log_effective_db(store, logger)

    attached = 0
    unmatched: list[str] = []
    skipped_already_graded = 0
    # Track trace_ids matched in any date iteration so prior-day window duplicates
    # are not double-reported in the unmatched list (BUG-DRY-1).
    matched_trace_ids: set[str] = set()

    for d in _iter_dates(start, end):
        logger.info("Fetching ESPN scoreboard for %s", d)
        try:
            games = sb_fetch(d.isoformat())
        except Exception as exc:  # noqa: BLE001
            logger.error("ESPN fetch failed for %s: %s", d, exc)
            return 1

        finals = [g for g in games if g.status == "final"]
        if not finals:
            logger.info("No final games on %s (%d events total)", d, len(games))
            continue

        games_by_pair: dict[tuple[str, str], FinalGame] = {
            (g.home_team, g.away_team): g for g in finals
        }

        # Pull ungraded NHL traces in this date window
        traces = store.query_traces(
            league="NHL",
            start=f"{d.isoformat()}T00:00:00Z",
            end=f"{d.isoformat()}T23:59:59Z",
            has_outcome=False,
            limit=500,
        )

        # Also pull ungraded traces from the prior day
        prior = d - timedelta(days=1)
        traces.extend(
            store.query_traces(
                league="NHL",
                start=f"{prior.isoformat()}T00:00:00Z",
                end=f"{prior.isoformat()}T23:59:59Z",
                has_outcome=False,
                limit=500,
            )
        )

        for trace in traces:
            if trace.get("kind") == "prop":
                continue

            tid = trace.get("trace_id", "?")

            game = _match_trace_to_game(trace, games_by_pair)
            if game is None:
                if tid not in matched_trace_ids:
                    pair = _trace_matchup(trace)
                    pair_str = f"{pair[1]} @ {pair[0]}" if pair else "<unresolved>"
                    unmatched.append(f"{tid} ({pair_str})")
                continue

            if args.dry_run:
                logger.info(
                    "DRY %s -> %s @ %s (%d-%d)",
                    tid,
                    game.away_team,
                    game.home_team,
                    game.away_score,
                    game.home_score,
                )
                matched_trace_ids.add(tid)
                attached += 1
                continue

            try:
                store.attach_outcome(
                    trace_id=tid,
                    home_score=game.home_score,
                    away_score=game.away_score,
                    source="api:espn",
                )
                matched_trace_ids.add(tid)
                attached += 1
                logger.info(
                    "ATTACHED %s -> %s %d, %s %d",
                    tid,
                    game.home_team,
                    game.home_score,
                    game.away_team,
                    game.away_score,
                )
            except ValueError as exc:
                logger.warning("Skipped %s: %s", tid, exc)
                skipped_already_graded += 1

    logger.info(
        "Done. attached=%d unmatched=%d skipped=%d",
        attached,
        len(unmatched),
        skipped_already_graded,
    )
    if unmatched:
        logger.warning("Unmatched traces (consider extending NHL_TEAMS aliases):")
        for line in unmatched:
            logger.warning("  - %s", line)

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
