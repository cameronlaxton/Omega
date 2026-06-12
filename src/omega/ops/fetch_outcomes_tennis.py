"""
omega.ops.fetch_outcomes_tennis — attach tennis match outcomes to traces.

Tennis has no ESPN scoreboard module, so grading is dual-source (Phase 7 M3):

* ``--source odds_api`` (primary, near-real-time): The Odds API ``/scores``
  endpoint per active tennis tournament key (keys are per-tournament — see
  ``resolve_tennis_sport_keys``). The provider's scores array carries sets won
  per player.
* ``--source sackmann`` (authoritative backfill, ~weekly lag): the Sackmann
  match CSVs already used for priors. Sets won are parsed from the ``score``
  string ("7-6(5) 6-4" -> 2-0). Retirements still grade when the winner leads
  in completed sets; walkovers never grade.

Outcomes attach as ``home_score``/``away_score`` = sets won (the tennis
SimulationResult convention), so moneyline settlement grades the match winner
and a set-spread bet grades against the sets margin. Player matching uses
``normalize_player_name`` + the TENNIS alias table on both sides.

Usage:
    omega-fetch-outcomes-tennis --source sackmann --since 2026-06-20 --until 2026-06-29
    omega-fetch-outcomes-tennis --source odds_api --leagues ATP
    omega-fetch-outcomes-tennis --source sackmann --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.integrations._etl import load_alias_table, resolve_entity  # noqa: E402
from omega.integrations.espn_boxscore import normalize_player_name  # noqa: E402
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("fetch_outcomes_tennis")

_TENNIS_LEAGUES = ("ATP", "WTA", "GRAND_SLAM")
_SET_TOKEN_RE = re.compile(r"^(\d+)-(\d+)(?:\(\d+\))?$")
# Tournament start dates lag actual match dates by up to two weeks (slams).
_TOURNEY_DATE_LOOKBACK_DAYS = 21


def parse_sets_from_score(score: str | None) -> tuple[int, int] | None:
    """Parse (winner_sets, loser_sets) from a Sackmann score string.

    Returns None for walkovers/defaults and for anomalies where the winner
    does not lead in completed sets (mid-set retirements) — those must not be
    graded as a sets outcome.
    """
    if not score:
        return None
    upper = score.upper()
    if "W/O" in upper or "WEA" in upper or "DEF" in upper:
        return None
    winner_sets = loser_sets = 0
    for token in score.split():
        m = _SET_TOKEN_RE.match(token)
        if m is None:
            continue  # "RET" markers and bracket junk
        winner_games, loser_games = int(m.group(1)), int(m.group(2))
        # Only completed sets count: 6+ with a 2-game margin, or 7-6/7-5
        # (super-tiebreak scores like 10-8 pass the first condition). A
        # mid-set retirement score like 4-1 is not a won set.
        hi, lo = max(winner_games, loser_games), min(winner_games, loser_games)
        if not ((hi >= 6 and hi - lo >= 2) or (hi == 7 and lo in (5, 6))):
            continue
        if winner_games > loser_games:
            winner_sets += 1
        else:
            loser_sets += 1
    if winner_sets <= loser_sets:
        return None
    return winner_sets, loser_sets


def _canonical_player(name: str, alias_table: dict) -> str:
    """Canonical match key for a player name: alias table, else normalized."""
    return resolve_entity(name, alias_table) or normalize_player_name(name)


def _parse_date_arg(s: str) -> date:
    s = s.strip().lower()
    if s == "today":
        return datetime.now(UTC).date()
    if s == "yesterday":
        return datetime.now(UTC).date() - timedelta(days=1)
    return date.fromisoformat(s)


def _trace_players(trace: dict) -> tuple[str, str] | None:
    snap = trace.get("input_snapshot") or {}
    home = snap.get("home_team")
    away = snap.get("away_team")
    if home and away:
        return str(home), str(away)
    matchup = trace.get("matchup") or ""
    if " @ " in matchup:
        away_str, home_str = matchup.split(" @ ", 1)
        return home_str.strip(), away_str.strip()
    return None


# ---------------------------------------------------------------------------
# Sources -> {(frozenset({playerA, playerB}), played_date): (player_sets map)}
# ---------------------------------------------------------------------------


MatchKey = tuple[frozenset[str], date]


def collect_sackmann_results(
    tours: list[str],
    start: date,
    end: date,
    alias_table: dict,
    *,
    local_root: str | None = None,
    cache_root: str | None = None,
) -> dict[MatchKey, dict[str, int]]:
    """Completed-match sets-won maps from the Sackmann CSVs in a date window."""
    from omega.integrations.tennis_sackmann import fetch_matches_csv, parse_matches

    lookback = start - timedelta(days=_TOURNEY_DATE_LOOKBACK_DAYS)
    years = sorted({start.year, end.year, lookback.year})
    results: dict[MatchKey, dict[str, int]] = {}
    for tour in tours:
        for year in years:
            try:
                csv_text = fetch_matches_csv(
                    tour, year, local_root=local_root, cache_root=cache_root
                )
            except Exception as exc:  # noqa: BLE001 - a missing season is non-fatal
                logger.warning("no %s %s match file: %s", tour, year, exc)
                continue
            for row in parse_matches(csv_text):
                played = datetime.strptime(str(row.tourney_date), "%Y%m%d").date()
                if not (lookback <= played <= end):
                    continue
                sets = parse_sets_from_score(row.score)
                if sets is None:
                    continue
                winner = _canonical_player(row.winner_name, alias_table)
                loser = _canonical_player(row.loser_name, alias_table)
                results[(frozenset((winner, loser)), played)] = {
                    winner: sets[0],
                    loser: sets[1],
                }
    return results


def collect_odds_api_results(
    tours: list[str],
    alias_table: dict,
    *,
    days_from: int = 3,
    client=None,
) -> dict[MatchKey, dict[str, int]]:
    """Completed-match sets-won maps from The Odds API /scores endpoint."""
    from omega.integrations.odds_api import OddsApiClient, resolve_tennis_sport_keys

    client = client or OddsApiClient()
    results: dict[MatchKey, dict[str, int]] = {}
    for tour in tours:
        try:
            sport_keys = resolve_tennis_sport_keys(client, tour)
        except Exception as exc:  # noqa: BLE001
            logger.error("could not resolve %s tournament keys: %s", tour, exc)
            continue
        for sport_key in sport_keys:
            try:
                events = client.fetch_scores(tour, days_from, sport_key=sport_key)
            except Exception as exc:  # noqa: BLE001 - one dead tournament is non-fatal
                logger.warning("scores fetch failed for %s: %s", sport_key, exc)
                continue
            for event in events:
                if not event.completed or len(event.scores) < 2:
                    continue
                try:
                    sets_by_player = {
                        _canonical_player(name, alias_table): int(score)
                        for name, score in event.scores
                    }
                except ValueError:
                    continue  # non-numeric score payload
                if len(sets_by_player) == 2:
                    played = datetime.fromisoformat(
                        event.commence_time.replace("Z", "+00:00")
                    ).date()
                    results[(frozenset(sets_by_player), played)] = sets_by_player
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attach tennis match outcomes (sets won) to game traces"
    )
    parser.add_argument("--source", choices=("sackmann", "odds_api"), default="sackmann")
    parser.add_argument(
        "--since", default="yesterday", help="Start date (YYYY-MM-DD | today | yesterday)"
    )
    parser.add_argument("--until", default=None, help="End date inclusive (default = since)")
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=_TENNIS_LEAGUES,
        default=list(_TENNIS_LEAGUES),
    )
    parser.add_argument("--local-root", default=None, help="Override data/tennis dir")
    parser.add_argument("--cache-root", default=None, help="Override ETL cache root")
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

    # GRAND_SLAM traces may belong to either tour; query both sources for them.
    tours = sorted(
        {"atp", "wta"}
        if "GRAND_SLAM" in args.leagues
        else {lg.lower() for lg in args.leagues}
    )
    alias_table = load_alias_table("TENNIS")

    if args.source == "sackmann":
        results = collect_sackmann_results(
            tours, start, end, alias_table,
            local_root=args.local_root, cache_root=args.cache_root,
        )
        source_tag = "api:sackmann"
    else:
        results = collect_odds_api_results(
            [t.upper() for t in tours], alias_table
        )
        source_tag = "api:odds_api"
    logger.info("%s: %d completed matches in window", args.source, len(results))

    store = TraceStore(db_path=args.db)
    log_effective_db(store, logger)

    attached = 0
    unmatched: list[str] = []
    skipped = 0
    for d in (start + timedelta(days=i) for i in range((end - start).days + 1)):
        for league in args.leagues:
            traces = store.query_traces(
                league=league,
                start=f"{d.isoformat()}T00:00:00Z",
                end=f"{d.isoformat()}T23:59:59Z",
                has_outcome=False,
                limit=500,
            )
            for trace in traces:
                if trace.get("kind") == "prop":
                    continue
                tid = trace.get("trace_id", "?")
                players = _trace_players(trace)
                if players is None:
                    unmatched.append(f"{tid} [{league}] (<unresolved players>)")
                    continue
                home_key = _canonical_player(players[0], alias_table)
                away_key = _canonical_player(players[1], alias_table)
                match = results.get((frozenset((home_key, away_key)), d))
                if match is None:
                    unmatched.append(f"{tid} [{league}] ({players[1]} @ {players[0]})")
                    continue

                home_sets, away_sets = match[home_key], match[away_key]
                if args.dry_run:
                    logger.info("DRY %s -> %s %d, %s %d", tid, players[0], home_sets, players[1], away_sets)
                    attached += 1
                    continue
                try:
                    store.attach_outcome(
                        trace_id=tid,
                        home_score=home_sets,
                        away_score=away_sets,
                        source=source_tag,
                    )
                    attached += 1
                    logger.info(
                        "ATTACHED %s -> %s %d, %s %d", tid, players[0], home_sets, players[1], away_sets
                    )
                except ValueError as exc:
                    logger.warning("Skipped %s: %s", tid, exc)
                    skipped += 1

    logger.info("Done. attached=%d unmatched=%d skipped=%d", attached, len(unmatched), skipped)
    if unmatched:
        logger.warning("Unmatched traces (consider extending data/aliases/TENNIS.json):")
        for line in unmatched:
            logger.warning("  - %s", line)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
