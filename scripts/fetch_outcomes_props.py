"""
scripts/fetch_outcomes_props.py — attach ESPN box-score stats to ungraded prop traces.

Workflow:
    1. Determine the date range (default: yesterday in UTC).
    2. For each date, query ungraded prop traces in the date window for each
       requested league.
    3. Group traces by (game_date, home_team, away_team) and fetch the ESPN
       scoreboard once per league/date pair to resolve game pair → event_id.
    4. For each game, fetch the box score once and parse player stats.
    5. For each prop trace, resolve the player + stat and call
       store.attach_prop_outcome(..., source="api:espn_boxscore").
    6. Report unmatched traces (missing game fields, unknown player, unmapped
       prop_type) so they can be triaged manually via
       scripts/backfill_outcomes_manual.py.

Usage:
    python scripts/fetch_outcomes_props.py
    python scripts/fetch_outcomes_props.py --since 2026-05-17 --until 2026-05-17
    python scripts/fetch_outcomes_props.py --league NBA --dry-run

Exit codes:
    0 — completed (may have unmatched; see log)
    1 — fatal error (bad args, ESPN unreachable on first fetch)
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc

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

logger = logging.getLogger("fetch_outcomes_props")

_SUPPORTED_LEAGUES = ("NBA", "MLB")


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
# Trace extraction
# ---------------------------------------------------------------------------


def _prop_fields(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the prop-grading fields out of a trace dict. Returns None when
    the trace is not a prop or required fields are missing."""
    if trace.get("kind") != "prop":
        return None
    snap = trace.get("input_snapshot") or {}
    player_name = snap.get("player_name")
    prop_type = snap.get("prop_type")
    line = snap.get("line")
    home_team = snap.get("home_team")
    away_team = snap.get("away_team")
    game_date = snap.get("game_date")
    if not (
        player_name and prop_type and line is not None and home_team and away_team and game_date
    ):
        return None
    # Determine grading side: prefer the trace's own recommendation when present
    result = trace.get("result") or {}
    side = (result.get("recommendation") or "").lower()
    if side not in ("over", "under"):
        # Fall back to the higher-probability side, then over.
        over_p = result.get("over_prob")
        under_p = result.get("under_prob")
        if over_p is not None and under_p is not None:
            side = "over" if over_p >= under_p else "under"
        else:
            side = "over"
    return {
        "player_name": player_name,
        "prop_type": str(prop_type).lower(),
        "line": float(line),
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date,
        "side": side,
    }


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


# ---------------------------------------------------------------------------
# Per-league scoreboard accessor (event_id resolution)
# ---------------------------------------------------------------------------


def _scoreboard_for(league: str, d: date):
    if league == "NBA":
        return espn_nba.fetch_scoreboard(d.isoformat())
    if league == "MLB":
        return espn_mlb.fetch_scoreboard(d.isoformat())
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    *,
    scoreboard_fetcher: Callable[[str, date], Any] | None = None,
    box_score_fetcher: Callable[[str, str], dict[str, Any]] | None = None,
) -> int:
    """CLI entry point.

    The two ``*_fetcher`` keyword args exist for tests to bypass HTTP: pass
    callables returning fixture data. Production calls leave them None and
    the live ESPN fetchers are used.
    """
    parser = argparse.ArgumentParser(
        description="Attach ESPN box-score player stats to NBA/MLB prop traces"
    )
    parser.add_argument(
        "--since", default="yesterday", help="Start date (YYYY-MM-DD | today | yesterday)"
    )
    parser.add_argument("--until", default=None, help="End date inclusive (default = since)")
    parser.add_argument(
        "--league",
        default=None,
        choices=_SUPPORTED_LEAGUES,
        help="Restrict to one league (default: all supported)",
    )
    parser.add_argument(
        "--db", default=None, help="SQLite path (default: repo-root omega_traces.db)"
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

    leagues = [args.league] if args.league else list(_SUPPORTED_LEAGUES)
    sb_fetch = scoreboard_fetcher or _scoreboard_for
    bs_fetch = box_score_fetcher or fetch_box_score

    store = TraceStore(db_path=args.db)

    attached = 0
    unmatched: list[str] = []
    unsupported_prop: list[str] = []
    skipped_missing_fields: list[str] = []
    # Dedup sets — the same trace_id can appear across multiple date windows
    # (BUG-PROP-1: duplicate warning lines per trace).
    _unsupported_seen: set[str] = set()
    _missing_fields_seen: set[str] = set()

    for league in leagues:
        for d in _iter_dates(start, end):
            # Pull ungraded prop traces in a generous time window (game date
            # ± 1 day). Then we filter to those whose game_date matches d.
            window_start = (d - timedelta(days=1)).isoformat() + "T00:00:00Z"
            window_end = (d + timedelta(days=1)).isoformat() + "T23:59:59Z"
            traces = store.query_traces(
                league=league,
                start=window_start,
                end=window_end,
                has_outcome=False,
                limit=1000,
            )

            # Defense-in-depth for BUG-2: when the agent minted a separate
            # bet-confirmation trace, the bet's trace_id is disjoint from the
            # analysis trace's id. Sweep pending prop bet_records so we grade
            # via the bet's trace_id (which the report tooling joins on).
            # Idempotent merge — skip ids already in `traces` to avoid
            # double-processing.
            seen_ids = {t.get("trace_id") for t in traces}
            for bt in store.query_ungraded_prop_bet_traces(
                league=league,
                start=window_start,
                end=window_end,
                limit=1000,
            ):
                if bt.get("trace_id") not in seen_ids:
                    traces.append(bt)
                    seen_ids.add(bt.get("trace_id"))

            # Group prop traces for this game date by canonical (home, away)
            grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = (
                defaultdict(list)
            )
            for trace in traces:
                fields = _prop_fields(trace)
                if fields is None:
                    if trace.get("kind") == "prop":
                        tid = trace.get("trace_id", "?")
                        # BUG-PROP-2: check prop_type against supported list first.
                        # An unsupported prop_type (e.g. first_basket) with all game
                        # identity fields present was misclassified as missing fields.
                        snap = trace.get("input_snapshot") or {}
                        pt = str(snap.get("prop_type") or "").lower()
                        if pt and not supported_prop_type(league, pt):
                            if tid not in _unsupported_seen:
                                _unsupported_seen.add(tid)
                                unsupported_prop.append(f"{tid} ({league} {pt})")
                        elif tid not in _missing_fields_seen:
                            _missing_fields_seen.add(tid)
                            skipped_missing_fields.append(
                                f"{tid} (missing game_date/home/away)"
                            )
                    continue
                if fields["game_date"] != d.isoformat():
                    continue
                if not supported_prop_type(league, fields["prop_type"]):
                    tid = trace.get("trace_id", "?")
                    if tid not in _unsupported_seen:
                        _unsupported_seen.add(tid)
                        unsupported_prop.append(f"{tid} ({league} {fields['prop_type']})")
                    continue
                pair = _canonical_pair(league, fields["home_team"], fields["away_team"])
                if pair is None:
                    unmatched.append(
                        f"{trace.get('trace_id', '?')} (unmapped team: "
                        f"{fields['away_team']} @ {fields['home_team']})"
                    )
                    continue
                grouped[pair].append((trace, fields))

            if not grouped:
                continue

            # Fetch scoreboard once for this league/date to resolve pair → event_id
            try:
                games = sb_fetch(league, d)
            except Exception as exc:  # noqa: BLE001
                logger.error("ESPN scoreboard fetch failed for %s %s: %s", league, d, exc)
                return 1

            games_by_pair = {(g.home_team, g.away_team): g for g in games}

            for pair, items in grouped.items():
                game = games_by_pair.get(pair)
                if game is None or game.status != "final":
                    for trace, fields in items:
                        unmatched.append(
                            f"{trace.get('trace_id', '?')} (no final game for "
                            f"{fields['away_team']} @ {fields['home_team']} on {d})"
                        )
                    continue

                # Fetch the box score once per game
                try:
                    payload = bs_fetch(league, game.event_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Box score fetch failed for event %s: %s", game.event_id, exc)
                    for trace, _fields in items:
                        unmatched.append(f"{trace.get('trace_id', '?')} (box score fetch failed)")
                    continue
                stats_by_player = parse_box_score(payload, league)

                for trace, fields in items:
                    player_key = normalize_player_name(fields["player_name"])
                    player_stats = stats_by_player.get(player_key)
                    if not player_stats:
                        unmatched.append(
                            f"{trace.get('trace_id', '?')} (player "
                            f"{fields['player_name']!r} not in box score)"
                        )
                        continue
                    if fields["prop_type"] not in player_stats:
                        unmatched.append(
                            f"{trace.get('trace_id', '?')} ({fields['prop_type']} "
                            f"not available for {fields['player_name']})"
                        )
                        continue
                    stat_value = player_stats[fields["prop_type"]]

                    if args.dry_run:
                        logger.info(
                            "DRY %s -> %s %s=%s vs %s (%s)",
                            trace["trace_id"],
                            fields["player_name"],
                            fields["prop_type"],
                            stat_value,
                            fields["line"],
                            fields["side"],
                        )
                        attached += 1
                        continue

                    try:
                        store.attach_prop_outcome(
                            trace_id=trace["trace_id"],
                            player_name=fields["player_name"],
                            stat_type=fields["prop_type"],
                            stat_value=float(stat_value),
                            line=float(fields["line"]),
                            side=fields["side"],
                            source="api:espn_boxscore",
                        )
                        attached += 1
                        logger.info(
                            "ATTACHED %s -> %s %s=%s vs %s (%s)",
                            trace["trace_id"],
                            fields["player_name"],
                            fields["prop_type"],
                            stat_value,
                            fields["line"],
                            fields["side"],
                        )
                    except ValueError as exc:
                        logger.warning("Skipped %s: %s", trace.get("trace_id"), exc)

    logger.info(
        "Done. attached=%d unmatched=%d unsupported=%d missing_fields=%d",
        attached,
        len(unmatched),
        len(unsupported_prop),
        len(skipped_missing_fields),
    )
    if unmatched:
        logger.warning("Unmatched prop traces:")
        for line in unmatched:
            logger.warning("  - %s", line)
    if unsupported_prop:
        logger.warning("Prop types not yet supported by box-score parser (grade manually):")
        for line in unsupported_prop:
            logger.warning("  - %s", line)
    if skipped_missing_fields:
        logger.warning("Prop traces missing game identity (grade manually):")
        for line in skipped_missing_fields:
            logger.warning("  - %s", line)

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
