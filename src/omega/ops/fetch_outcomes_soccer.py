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
from datetime import timedelta
from pathlib import Path
from typing import Any

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
from omega.ops._date_args import iter_dates as _iter_dates  # noqa: E402
from omega.ops._date_args import parse_date_arg as _parse_date_arg  # noqa: E402
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("fetch_outcomes_soccer")

_SOCCER_LEAGUES = tuple(SOCCER_LEAGUE_SLUGS.keys())

# Competitions whose knockout matches are SINGLE-LEG: a match that reaches extra
# time or penalties was level at 90', so the 3-way (1X2) "draw" market settles as
# a draw regardless of the ET/shootout winner. Two-legged ties (e.g. UCL knockout
# rounds) are deliberately excluded — a second leg reaching ET reflects the
# AGGREGATE, not that leg's 90' result, so the same inference does not hold.
_SINGLE_LEG_KNOCKOUT_LEAGUES = frozenset(
    {"WORLD_CUP", "FIFA_WORLD_CUP_2026", "FIFA_INTL", "EURO", "COPA_AMERICA", "NATIONS_LEAGUE"}
)


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
) -> tuple[FinalGame, bool] | None:
    pair = _trace_matchup(trace)
    if pair is None:
        return None
    # 1. Try exact match
    if pair in games_by_pair:
        return games_by_pair[pair], False
    # 2. Try reversed match (neutral site games can nominalize home/away differently)
    reversed_pair = (pair[1], pair[0])
    if reversed_pair in games_by_pair:
        return games_by_pair[reversed_pair], True
    return None


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
    parser = argparse.ArgumentParser(description="Attach ESPN final scores to soccer game traces")
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
    all_queried_traces: dict[str, tuple[str, str]] = {}
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

            # Query traces in a generous window around date d (d - 30 days to d + 2 days)
            # to handle pre-placed bets (e.g. World Cup tournament) and timezone offsets.
            window_start = (d - timedelta(days=30)).isoformat() + "T00:00:00Z"
            window_end = (d + timedelta(days=2)).isoformat() + "T23:59:59Z"
            traces = store.query_traces(
                league=league,
                start=window_start,
                end=window_end,
                has_outcome=False,
                limit=1000,
            )

            for trace in traces:
                # Prop traces grade against player stats — fetch_outcomes_props owns them.
                if trace.get("kind") == "prop":
                    continue

                tid = trace.get("trace_id", "?")
                pair = _trace_matchup(trace)
                pair_str = f"{pair[1]} @ {pair[0]}" if pair else "<unresolved>"
                all_queried_traces[tid] = (league, pair_str)

                matched = _match_trace_to_game(trace, games_by_pair)
                if matched is None:
                    continue
                game, is_reversed = matched

                home_score = game.away_score if is_reversed else game.home_score
                away_score = game.home_score if is_reversed else game.away_score

                # Single-leg knockout decided after regulation -> the 1X2 result
                # is a regulation draw (sides were level at 90'), even if the ESPN
                # score shows the ET/shootout winner. Scores are still stored as
                # provenance; the override only sets the graded 3-way result.
                result_override = None
                if game.decided_after_regulation and league.upper() in _SINGLE_LEG_KNOCKOUT_LEAGUES:
                    result_override = "draw"

                h_team = pair[0] if pair else "Home"
                a_team = pair[1] if pair else "Away"

                if args.dry_run:
                    logger.info(
                        "DRY %s -> %s @ %s (%d-%d)",
                        tid,
                        a_team,
                        h_team,
                        away_score,
                        home_score,
                    )
                    matched_trace_ids.add(tid)
                    attached += 1
                    continue

                try:
                    store.attach_outcome(
                        trace_id=tid,
                        home_score=home_score,
                        away_score=away_score,
                        source="api:espn:aet" if result_override else "api:espn",
                        result_override=result_override,
                    )
                    matched_trace_ids.add(tid)
                    attached += 1
                    logger.info(
                        "ATTACHED %s -> %s %d, %s %d",
                        tid,
                        h_team,
                        home_score,
                        a_team,
                        away_score,
                    )
                except ValueError as exc:
                    logger.warning("Skipped %s: %s", tid, exc)
                    skipped_already_graded += 1

    unmatched = []
    for tid, (league, pair_str) in all_queried_traces.items():
        if tid not in matched_trace_ids:
            unmatched.append(f"{tid} [{league}] ({pair_str})")

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
