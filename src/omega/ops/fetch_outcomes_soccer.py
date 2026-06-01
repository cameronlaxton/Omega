"""
omega.ops.fetch_outcomes_soccer â€” attach soccer final scores from ESPN to traces.

Game-plane outcome grader for the soccer leagues Omega simulates (MLS, EPL,
LA_LIGA, BUNDESLIGA, SERIE_A, LIGUE_1, CHAMPIONS_LEAGUE, LIGA_MX). Soccer is a
3-way result; draw grading is handled by ``TraceStore.attach_outcome`` (equal
scores â†’ ``result == "draw"``). This script never touches prop traces.

Unlike the basketball/baseball scripts, ESPN's soccer scoreboard is
per-competition, so each (date, league) pair is fetched separately.

Workflow (per date, per soccer league):
    1. Fetch that league's ESPN scoreboard for the date.
    2. Query ungraded game traces for that league code in the date window
       (game date and the prior day).
    3. Match each trace to a FinalGame by canonical (home, away).
    4. Attach via store.attach_outcome(..., source="api:espn").
    5. Report unmatched traces so aliases can be added to
       omega/integrations/espn_soccer.py::SOCCER_TEAM_ALIASES.

Usage:
    omega-fetch-outcomes-soccer
    omega-fetch-outcomes-soccer --since 2026-05-17 --until 2026-05-17
    omega-fetch-outcomes-soccer --leagues EPL LA_LIGA
    omega-fetch-outcomes-soccer --dry-run

Exit codes:
    0 â€” completed (may have unmatched; see log)
    1 â€” fatal error (bad args, ESPN unreachable)
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.integrations.espn_soccer import (  # noqa: E402
    SOCCER_LEAGUE_SLUGS,
    FinalGame,
    canonical_team,
    fetch_scoreboard,
)
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("fetch_outcomes_soccer")

_SOCCER_LEAGUES = tuple(SOCCER_LEAGUE_SLUGS.keys())


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def _parse_date_arg(s: str) -> date:
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
    """Pull (home_canonical, away_canonical) from a trace, or None if unresolvable."""
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
    scoreboard_fetcher: Callable[[str, str], Any] | None = None,
) -> int:
    """CLI entry point.

    ``scoreboard_fetcher`` exists for tests to bypass HTTP: pass a callable
    taking (iso_date, league) and returning a list of FinalGame. Production
    leaves it None and the live ESPN soccer fetcher is used.
    """
    parser = argparse.ArgumentParser(
        description="Attach ESPN final scores to soccer game traces"
    )
    parser.add_argument(
        "--since", default="yesterday", help="Start date (YYYY-MM-DD | today | yesterday)"
    )
    parser.add_argument("--until", default=None, help="End date inclusive (default = since)")
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=sorted(_SOCCER_LEAGUES),
        default=list(_SOCCER_LEAGUES),
        help="Which soccer leagues to process (default: all)",
    )
    parser.add_argument(
        "--db", default=None, help="SQLite path (default: var/omega_traces.db)"
    )
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
    matched_trace_ids: set[str] = set()

    for d in _iter_dates(start, end):
        for league in args.leagues:
            logger.info("Fetching ESPN soccer scoreboard for %s on %s", league, d)
            try:
                games = sb_fetch(d.isoformat(), league)
            except Exception as exc:  # noqa: BLE001
                logger.error("ESPN soccer fetch failed for %s %s: %s", league, d, exc)
                return 1

            finals = [g for g in games if g.status == "final"]
            if not finals:
                logger.info("No final %s matches on %s (%d events)", league, d, len(games))
                continue

            games_by_pair: dict[tuple[str, str], FinalGame] = {
                (g.home_team, g.away_team): g for g in finals
            }

            traces = store.query_traces(
                league=league,
                start=f"{d.isoformat()}T00:00:00Z",
                end=f"{d.isoformat()}T23:59:59Z",
                has_outcome=False,
                limit=500,
            )
            prior = d - timedelta(days=1)
            traces.extend(
                store.query_traces(
                    league=league,
                    start=f"{prior.isoformat()}T00:00:00Z",
                    end=f"{prior.isoformat()}T23:59:59Z",
                    has_outcome=False,
                    limit=500,
                )
            )

            for trace in traces:
                # Prop traces grade against player stats â€” fetch_outcomes_props owns them.
                if trace.get("kind") == "prop":
                    continue

                tid = trace.get("trace_id", "?")
                game = _match_trace_to_game(trace, games_by_pair)
                if game is None:
                    if tid not in matched_trace_ids:
                        pair = _trace_matchup(trace)
                        pair_str = f"{pair[1]} @ {pair[0]}" if pair else "<unresolved>"
                        unmatched.append(f"{tid} [{league}] ({pair_str})")
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
        logger.warning("Unmatched traces (consider extending SOCCER_TEAM_ALIASES):")
        for line in unmatched:
            logger.warning("  - %s", line)

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())





