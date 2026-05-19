"""
scripts/backfill_outcomes_manual.py — interactive ESPN-based grading for the
backlog of ungraded game/prop traces.

Use case: traces from before the producer fix carried no game identity for
props, or game traces whose teams aliased badly. This script walks every
ungraded trace tied to a given (league, date) and offers attach / skip /
quit prompts. Outcomes are written with
``source="manual:espn_boxscore_<YYYYMMDD>"`` matching the convention used
for this weekend's first manual run.

Two modes:

1. **Date-window mode** (default):
       python scripts/backfill_outcomes_manual.py --date 2026-05-17 --league NBA

   Fetches the ESPN scoreboard + per-game box scores once, then walks each
   ungraded trace whose game_date (or trace timestamp, for legacy game-only
   traces) matches.

2. **Single-trace mode** (for legacy props lacking game identity):
       python scripts/backfill_outcomes_manual.py \\
           --trace-id sandbox-abcd1234 \\
           --league NBA --game-date 2026-05-17 \\
           --home "Miami Heat" --away "Boston Celtics"

   Lets the operator manually point a single trace at a specific game.

Never mutates the trace blob itself — outcomes always land in the
outcomes / prop_outcomes tables, preserving Phase 6's "outcome attachment
must not mutate source records" invariant.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.integrations import espn_mlb, espn_nba  # noqa: E402
from omega.integrations.espn_boxscore import (  # noqa: E402
    fetch_box_score,
    normalize_player_name,
    parse_box_score,
    supported_prop_type,
)
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("backfill_outcomes_manual")

_SUPPORTED_LEAGUES = ("NBA", "MLB")

# Confirmation callback: takes a description and returns "y" | "n" | "q".
ConfirmFn = Callable[[str], str]


# ---------------------------------------------------------------------------
# Helpers shared with fetch_outcomes_props
# ---------------------------------------------------------------------------

def _interactive_confirm(description: str) -> str:
    """Prompt the operator. Returns 'y' (attach), 'n' (skip), or 'q' (quit)."""
    while True:
        answer = input(f"{description} [y/n/q]: ").strip().lower()
        if answer in ("y", "n", "q"):
            return answer
        print("Please answer y, n, or q.")


def _canonical_pair(league: str, home: str, away: str) -> tuple[str, str] | None:
    if league == "NBA":
        c_home = espn_nba.canonical_team(home)
        c_away = espn_nba.canonical_team(away)
    elif league == "MLB":
        c_home = espn_mlb.canonical_team(home)
        c_away = espn_mlb.canonical_team(away)
    else:
        return None
    if not (c_home and c_away):
        return None
    return (c_home, c_away)


def _scoreboard_for(league: str, d: date):
    if league == "NBA":
        return espn_nba.fetch_scoreboard(d.isoformat())
    if league == "MLB":
        return espn_mlb.fetch_scoreboard(d.isoformat())
    return []


def _source_label(d: date) -> str:
    return f"manual:espn_boxscore_{d.strftime('%Y%m%d')}"


# ---------------------------------------------------------------------------
# Trace classification
# ---------------------------------------------------------------------------

def _trace_game_pair(
    trace: dict[str, Any],
    league: str,
) -> tuple[str, str] | None:
    """Resolve a trace to a canonical (home, away) pair, or None."""
    snap = trace.get("input_snapshot") or {}
    home = snap.get("home_team")
    away = snap.get("away_team")
    if home and away:
        return _canonical_pair(league, home, away)
    matchup = trace.get("matchup") or ""
    if " @ " in matchup:
        away_s, home_s = matchup.split(" @ ", 1)
        return _canonical_pair(league, home_s, away_s)
    return None


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def backfill_date(
    store: TraceStore,
    league: str,
    d: date,
    *,
    confirm: ConfirmFn = _interactive_confirm,
    scoreboard_fetcher: Callable[[str, date], Any] = _scoreboard_for,
    box_score_fetcher: Callable[[str, str], dict[str, Any]] = fetch_box_score,
    dry_run: bool = False,
) -> dict[str, int]:
    """Walk every ungraded trace tied to `(league, d)` with attach/skip prompts.

    Returns counts: {'game_attached', 'prop_attached', 'skipped', 'quit'}.
    """
    counts = {
        "game_attached": 0,
        "prop_attached": 0,
        "skipped": 0,
        "quit": 0,
        "unmatched": 0,
    }

    window_start = (d - timedelta(days=1)).isoformat() + "T00:00:00Z"
    window_end = (d + timedelta(days=1)).isoformat() + "T23:59:59Z"
    traces = store.query_traces(
        league=league, start=window_start, end=window_end,
        has_outcome=False, limit=1000,
    )

    if not traces:
        logger.info("No ungraded %s traces in window [%s, %s]", league, window_start, window_end)
        return counts

    games = scoreboard_fetcher(league, d)
    games_by_pair = {(g.home_team, g.away_team): g for g in games}

    # Cache box scores so we fetch each event once per session
    box_score_cache: dict[str, dict[str, Any]] = {}
    source = _source_label(d)

    for trace in traces:
        tid = trace.get("trace_id", "?")
        kind = trace.get("kind") or "unknown"
        snap = trace.get("input_snapshot") or {}

        if kind == "prop":
            game_date = snap.get("game_date")
            if game_date and game_date != d.isoformat():
                continue
            pair = _trace_game_pair(trace, league)
            if pair is None:
                logger.warning("UNMATCHED prop %s (no team identity)", tid)
                counts["unmatched"] += 1
                continue
            game = games_by_pair.get(pair)
            if game is None or game.status != "final":
                logger.warning("UNMATCHED prop %s (no final game for %s @ %s on %s)",
                               tid, pair[1], pair[0], d)
                counts["unmatched"] += 1
                continue
            player_name = snap.get("player_name")
            prop_type = (snap.get("prop_type") or "").lower()
            line = snap.get("line")
            if not (player_name and prop_type and line is not None):
                logger.warning("UNMATCHED prop %s (missing player/prop_type/line)", tid)
                counts["unmatched"] += 1
                continue
            if not supported_prop_type(league, prop_type):
                logger.warning("UNSUPPORTED prop_type %r on %s (grade by hand)", prop_type, tid)
                counts["unmatched"] += 1
                continue

            payload = box_score_cache.get(game.event_id)
            if payload is None:
                try:
                    payload = box_score_fetcher(league, game.event_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Box score fetch failed for %s: %s", game.event_id, exc)
                    counts["unmatched"] += 1
                    continue
                box_score_cache[game.event_id] = payload
            stats = parse_box_score(payload, league)
            player_stats = stats.get(normalize_player_name(player_name))
            if not player_stats or prop_type not in player_stats:
                logger.warning("UNMATCHED prop %s (player %r %r not in box score)",
                               tid, player_name, prop_type)
                counts["unmatched"] += 1
                continue
            stat_value = player_stats[prop_type]

            result = trace.get("result") or {}
            side = (result.get("recommendation") or "").lower()
            if side not in ("over", "under"):
                side = "over" if (result.get("over_prob") or 0) >= (result.get("under_prob") or 0) else "under"

            desc = (
                f"PROP {tid}: {player_name} {prop_type} {line} ({side}) "
                f"=> actual={stat_value} [{pair[1]} @ {pair[0]} {d}]"
            )
            answer = confirm(desc)
            if answer == "q":
                counts["quit"] += 1
                return counts
            if answer == "n":
                counts["skipped"] += 1
                continue
            if dry_run:
                logger.info("DRY %s", desc)
                counts["prop_attached"] += 1
                continue
            store.attach_prop_outcome(
                trace_id=tid,
                player_name=player_name,
                stat_type=prop_type,
                stat_value=float(stat_value),
                line=float(line),
                side=side,
                source=source,
            )
            counts["prop_attached"] += 1
            logger.info("ATTACHED %s", desc)

        else:
            # BUG-3 guard: only true game traces get game-shaped outcomes.
            # Previously the `else` branch swept up "research / analysis-only
            # / unknown" kinds, which let placeholder 1-0 scores land on prop
            # traces missing a `kind` field. Skip anything that isn't
            # explicitly kind='game'.
            if kind != "game":
                logger.warning(
                    "SKIP %s: trace.kind=%r is not 'game'; refusing to attach "
                    "a game-shaped outcome. Grade props via the prop branch or "
                    "fix the trace's kind field.",
                    tid, kind,
                )
                counts["skipped"] += 1
                continue
            pair = _trace_game_pair(trace, league)
            if pair is None:
                logger.warning("UNMATCHED game-kind trace %s (no team identity)", tid)
                counts["unmatched"] += 1
                continue
            game = games_by_pair.get(pair)
            if game is None or game.status != "final":
                logger.warning("UNMATCHED game-kind trace %s (no final game for %s @ %s)",
                               tid, pair[1], pair[0])
                counts["unmatched"] += 1
                continue
            desc = (
                f"GAME {tid} [{kind}]: {pair[1]} @ {pair[0]} {d} "
                f"=> {game.away_team} {game.away_score}, {game.home_team} {game.home_score}"
            )
            answer = confirm(desc)
            if answer == "q":
                counts["quit"] += 1
                return counts
            if answer == "n":
                counts["skipped"] += 1
                continue
            if dry_run:
                logger.info("DRY %s", desc)
                counts["game_attached"] += 1
                continue
            store.attach_outcome(
                trace_id=tid,
                home_score=game.home_score,
                away_score=game.away_score,
                source=source,
            )
            counts["game_attached"] += 1
            logger.info("ATTACHED %s", desc)

    return counts


def backfill_single_trace(
    store: TraceStore,
    trace_id: str,
    league: str,
    game_date: date,
    home_team: str,
    away_team: str,
    *,
    confirm: ConfirmFn = _interactive_confirm,
    scoreboard_fetcher: Callable[[str, date], Any] = _scoreboard_for,
    box_score_fetcher: Callable[[str, str], dict[str, Any]] = fetch_box_score,
    dry_run: bool = False,
) -> dict[str, int]:
    """Pin a single legacy trace to an operator-supplied game and grade it."""
    counts = {"game_attached": 0, "prop_attached": 0, "skipped": 0, "quit": 0, "unmatched": 0}

    trace = store.get_trace(trace_id)
    if trace is None:
        logger.error("Trace not found: %s", trace_id)
        counts["unmatched"] += 1
        return counts

    pair = _canonical_pair(league, home_team, away_team)
    if pair is None:
        logger.error("Could not canonicalize teams: %s / %s", home_team, away_team)
        counts["unmatched"] += 1
        return counts

    games = scoreboard_fetcher(league, game_date)
    game = next((g for g in games if (g.home_team, g.away_team) == pair and g.status == "final"), None)
    if game is None:
        logger.error("No final game for %s @ %s on %s", pair[1], pair[0], game_date)
        counts["unmatched"] += 1
        return counts

    source = _source_label(game_date)
    kind = trace.get("kind") or "unknown"
    snap = trace.get("input_snapshot") or {}

    if kind == "prop":
        player_name = snap.get("player_name")
        prop_type = (snap.get("prop_type") or "").lower()
        line = snap.get("line")
        if not (player_name and prop_type and line is not None):
            logger.error("Prop trace %s missing player_name/prop_type/line", trace_id)
            counts["unmatched"] += 1
            return counts
        if not supported_prop_type(league, prop_type):
            logger.error("Unsupported prop_type %r for league %s", prop_type, league)
            counts["unmatched"] += 1
            return counts
        try:
            payload = box_score_fetcher(league, game.event_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Box score fetch failed: %s", exc)
            counts["unmatched"] += 1
            return counts
        stats = parse_box_score(payload, league)
        player_stats = stats.get(normalize_player_name(player_name))
        if not player_stats or prop_type not in player_stats:
            logger.error("Player %r %r not in box score", player_name, prop_type)
            counts["unmatched"] += 1
            return counts
        stat_value = player_stats[prop_type]
        result = trace.get("result") or {}
        side = (result.get("recommendation") or "").lower()
        if side not in ("over", "under"):
            side = "over" if (result.get("over_prob") or 0) >= (result.get("under_prob") or 0) else "under"
        desc = (
            f"PROP {trace_id}: {player_name} {prop_type} {line} ({side}) "
            f"=> actual={stat_value} [{pair[1]} @ {pair[0]} {game_date}]"
        )
        answer = confirm(desc)
        if answer == "q":
            counts["quit"] += 1
            return counts
        if answer == "n":
            counts["skipped"] += 1
            return counts
        if not dry_run:
            store.attach_prop_outcome(
                trace_id=trace_id,
                player_name=player_name,
                stat_type=prop_type,
                stat_value=float(stat_value),
                line=float(line),
                side=side,
                source=source,
            )
        counts["prop_attached"] += 1
        logger.info("ATTACHED %s", desc)
    else:
        # BUG-3 guard: see backfill_date(). Only explicit game traces get a
        # game-shaped outcome attached.
        if kind != "game":
            logger.warning(
                "SKIP %s: trace.kind=%r is not 'game'; refusing to attach a "
                "game-shaped outcome.", trace_id, kind,
            )
            counts["skipped"] += 1
            return counts
        desc = (
            f"GAME {trace_id} [{kind}]: {pair[1]} @ {pair[0]} {game_date} "
            f"=> {game.away_team} {game.away_score}, {game.home_team} {game.home_score}"
        )
        answer = confirm(desc)
        if answer == "q":
            counts["quit"] += 1
            return counts
        if answer == "n":
            counts["skipped"] += 1
            return counts
        if not dry_run:
            store.attach_outcome(
                trace_id=trace_id,
                home_score=game.home_score,
                away_score=game.away_score,
                source=source,
            )
        counts["game_attached"] += 1
        logger.info("ATTACHED %s", desc)

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date_arg(s: str) -> date:
    s = s.strip().lower()
    if s == "today":
        return datetime.now(UTC).date()
    if s == "yesterday":
        return datetime.now(UTC).date() - timedelta(days=1)
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive ESPN-based grading for the ungraded trace backlog"
    )
    parser.add_argument("--league", required=True, choices=_SUPPORTED_LEAGUES)
    parser.add_argument("--date", help="Walk all ungraded traces on this date (YYYY-MM-DD | today | yesterday)")
    parser.add_argument("--trace-id", help="Single-trace mode: pin this trace to the supplied game")
    parser.add_argument("--game-date", help="(single-trace mode) Game date YYYY-MM-DD")
    parser.add_argument("--home", help="(single-trace mode) Home team string")
    parser.add_argument("--away", help="(single-trace mode) Away team string")
    parser.add_argument("--db", default=None, help="SQLite path (default: repo-root omega_traces.db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would attach but don't write")
    parser.add_argument("--yes", action="store_true",
                        help="Auto-confirm every prompt (non-interactive batch mode)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    confirm: ConfirmFn = (lambda _desc: "y") if args.yes else _interactive_confirm

    store = TraceStore(db_path=args.db)
    try:
        if args.trace_id:
            if not (args.game_date and args.home and args.away):
                logger.error("--trace-id mode requires --game-date, --home, and --away")
                return 1
            counts = backfill_single_trace(
                store,
                trace_id=args.trace_id,
                league=args.league,
                game_date=_parse_date_arg(args.game_date),
                home_team=args.home,
                away_team=args.away,
                confirm=confirm,
                dry_run=args.dry_run,
            )
        elif args.date:
            counts = backfill_date(
                store,
                league=args.league,
                d=_parse_date_arg(args.date),
                confirm=confirm,
                dry_run=args.dry_run,
            )
        else:
            logger.error("Specify either --date <YYYY-MM-DD> or --trace-id <id>")
            return 1
    finally:
        store.close()

    logger.info(
        "Done. game_attached=%d prop_attached=%d skipped=%d unmatched=%d quit=%d",
        counts["game_attached"], counts["prop_attached"],
        counts["skipped"], counts["unmatched"], counts["quit"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
