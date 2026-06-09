"""
omega.ops.backfill_closing_lines â€” backfill missed closing-line windows from the
paid Odds API historical endpoint.

Use this when the T-30min live capture (fetch_closing_lines.py) was not run for
a past bet. The historical endpoint returns the closest snapshot equal to or
earlier than the requested ``date`` timestamp.

Cost note: historical requests are charged per region+market combination, the
same as live requests. Check omega_odds_api_budget.json before running a wide
date range.

Usage:
    # Backfill all pending bets for NBA from last week, closing at 21:00 ET
    omega-backfill-closing-lines --league NBA --since 2026-05-10 --until 2026-05-14 --close-time 01:00:00Z

    # Backfill one specific trace at an exact timestamp
    omega-backfill-closing-lines --trace-id sandbox-abc123 --at 2026-05-14T21:30:00Z

    # Dry run (no DB writes, still hits the API)
    omega-backfill-closing-lines --league NBA --since 2026-05-10 --dry-run

Exit codes:
    0 â€” completed (may have unmatched; see log)
    1 â€” fatal error (API key missing, bad args)
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.integrations.odds_api import (  # noqa: E402
    EventOdds,
    OddsApiBudgetExceeded,
    OddsApiClient,
    OddsApiKeyMissing,
    sport_key_for,
)
from omega.integrations.odds_resolver import provider_market_for_prop  # noqa: E402
from omega.trace.db import require_sqlite_backend  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("backfill_closing_lines")

_DEFAULT_CLOSE_OFFSET_HOURS = 6  # added to decision_timestamp when --close-time is not given

# ---------------------------------------------------------------------------
# Helpers shared with fetch_closing_lines (duplicated to keep scripts
# self-contained; extract to omega.integrations.match if they grow)
# ---------------------------------------------------------------------------


def _identity(name: str) -> str | None:
    return name or None


def _load_canonicalizer(league: str) -> Callable[[str], str | None]:
    # Soccer leagues (MLS, EPL, CHAMPIONS_LEAGUE, WORLD_CUP, LIGA_MX, ...) all
    # share one canonicalizer in espn_soccer rather than a per-league
    # espn_<league> module. Resolve them there first; otherwise they fall through
    # to _identity and team names are never normalized against the Odds API
    # snapshot (e.g. "USA" vs "United States"), so no soccer event ever matches.
    try:
        from omega.integrations.espn_soccer import SOCCER_LEAGUE_SLUGS, canonical_team

        if league.upper() in SOCCER_LEAGUE_SLUGS:
            return canonical_team
    except ImportError:
        pass

    league_lc = league.lower()
    for mod_path in (
        f"omega.integrations.espn_{league_lc}",
        f"omega.integrations.team_aliases_{league_lc}",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        fn = getattr(mod, "canonical_team", None)
        if callable(fn):
            return fn
    return _identity


_MARKET_MAP = {"moneyline": "h2h", "spread": "spreads", "total": "totals"}


def _is_supported_market(bet_market: str) -> bool:
    return bet_market in _MARKET_MAP or bet_market.startswith("player_prop:")


def _match_outcome(
    bet_market: str,
    descriptor: str,
    home: str,
    away: str,
    books,
    book_preference: str | None,
):
    odds_market = _MARKET_MAP[bet_market]
    candidates = [b for b in books if b.market == odds_market]
    if book_preference:
        prefs = [b for b in candidates if b.bookmaker == book_preference.lower()]
        if prefs:
            candidates = prefs

    desc = descriptor.lower()
    if bet_market == "moneyline":
        target = home if "home" in desc else (away if "away" in desc else None)
        if target is None:
            return None
        for b in candidates:
            if b.selection.lower() == target.lower():
                return b
        return None

    if bet_market == "spread":
        target = home if "home" in desc else (away if "away" in desc else None)
        if target is None:
            return None
        for b in candidates:
            if b.selection.lower() == target.lower():
                return b
        return None

    if bet_market == "total":
        label = "Over" if "over" in desc else ("Under" if "under" in desc else None)
        if label is None:
            return None
        for b in candidates:
            if b.selection == label:
                return b
        return None

    return None


def _match_prop_outcome(bet: dict, books):
    stat_key = bet["market"].split(":", 1)[1]
    provider_market = provider_market_for_prop(bet["league"], stat_key)
    if not provider_market:
        return None
    desc = f"{bet.get('selection_descriptor', '')} {bet.get('selection', '')}".lower()
    side = "over" if "over" in desc else ("under" if "under" in desc else None)
    if side is None:
        return None
    selection = bet.get("selection") or ""
    player_hint = selection.split(" Over ", 1)[0].split(" Under ", 1)[0]
    for book in books:
        if book.market != provider_market:
            continue
        if book.selection.lower() != side:
            continue
        if bet.get("line_taken") is not None and book.point != float(bet["line_taken"]):
            continue
        if player_hint and book.description and player_hint.lower() not in book.description.lower():
            continue
        return book
    return None


# ---------------------------------------------------------------------------
# Bet selection
# ---------------------------------------------------------------------------


def _pending_bets_needing_close(
    store: TraceStore,
    league_filter: str | None,
    since: str | None,
    until: str | None,
    trace_id: str | None,
) -> list[dict]:
    sql = (
        "SELECT b.ledger_id AS bet_id, b.trace_id, b.bookmaker AS book, b.market,"
        "       b.selection, b.selection_descriptor, b.line AS line_taken,"
        "       b.odds AS odds_taken, b.decision_timestamp, b.status,"
        "       t.league, t.full_trace"
        " FROM bet_ledger b"
        " JOIN traces t ON t.trace_id = b.trace_id"
        " LEFT JOIN closing_lines c"
        "   ON c.trace_id = b.trace_id"
        "  AND c.market = b.market"
        "  AND c.selection_descriptor = b.selection_descriptor"
        " WHERE b.provenance IN ('user_confirmed', 'engine_auto', 'backfill')"
        "   AND c.closing_id IS NULL"
    )
    params: list = []
    if league_filter:
        sql += " AND t.league = ?"
        params.append(league_filter.upper())
    if since:
        sql += " AND b.decision_timestamp >= ?"
        params.append(since)
    if until:
        sql += " AND b.decision_timestamp <= ?"
        params.append(until)
    if trace_id:
        sql += " AND b.trace_id = ?"
        params.append(trace_id)
    rows = store.conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Historical timestamp derivation
# ---------------------------------------------------------------------------


def _close_timestamp_for_bet(bet: dict, explicit_at: str | None, close_offset_hours: int) -> str:
    """Return an ISO-8601 UTC timestamp to pass to the historical endpoint.

    If ``explicit_at`` is provided it is used for all bets. Otherwise we add
    ``close_offset_hours`` to the bet's decision_timestamp as a proxy for when
    the line closed (typically 4â€“8 h after the decision is logged).
    """
    if explicit_at:
        return explicit_at
    decision = bet.get("decision_timestamp") or ""
    try:
        dt = datetime.fromisoformat(decision.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        dt = datetime.now(UTC)
    return (dt + timedelta(hours=close_offset_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Per-league processing
# ---------------------------------------------------------------------------


def _process_league(
    league: str,
    bets: list[dict],
    client: OddsApiClient,
    store: TraceStore,
    explicit_at: str | None,
    close_offset_hours: int,
    dry_run: bool,
) -> tuple[int, list[str]]:
    sport_key = sport_key_for(league)
    if sport_key is None:
        return 0, [f"{b['bet_id']} (league {league!r} has no sport_key mapping)" for b in bets]

    canonicalize = _load_canonicalizer(league)
    attached = 0
    skipped: list[str] = []

    # Group bets by the historical timestamp we'll use for them so we issue
    # one API request per timestamp bucket rather than one per bet.
    by_ts: dict[str, list[dict]] = {}
    for bet in bets:
        ts = _close_timestamp_for_bet(bet, explicit_at, close_offset_hours)
        by_ts.setdefault(ts, []).append(bet)

    for ts, ts_bets in sorted(by_ts.items()):
        logger.info("[%s] fetching historical snapshot at %s (%d bet(s))", league, ts, len(ts_bets))
        try:
            snapshot = client.fetch_historical_odds(league, date=ts)
        except OddsApiKeyMissing:
            logger.error("OMEGA_ODDS_API_KEY not set.")
            return attached, skipped + [f"{b['bet_id']} (api key missing)" for b in ts_bets]
        except OddsApiBudgetExceeded as exc:
            logger.error("Budget exceeded: %s", exc)
            return attached, skipped + [f"{b['bet_id']} (budget exceeded)" for b in ts_bets]
        except Exception as exc:  # noqa: BLE001
            logger.error("Historical fetch failed for %s at %s: %s", league, ts, exc)
            skipped.extend(f"{b['bet_id']} (api fetch failed: {exc})" for b in ts_bets)
            continue

        actual_ts = snapshot.timestamp or ts
        logger.info(
            "[%s] snapshot timestamp=%s, events=%d, remaining budget=%d",
            league,
            actual_ts,
            len(snapshot.events),
            client.remaining_budget(),
        )

        events_by_pair: dict[tuple[str, str], EventOdds] = {
            (canonicalize(e.home_team) or e.home_team, canonicalize(e.away_team) or e.away_team): e
            for e in snapshot.events
        }

        for bet in ts_bets:
            if not _is_supported_market(bet["market"]):
                skipped.append(f"{bet['bet_id']} (unsupported market: {bet['market']})")
                continue

            import json as _json

            try:
                input_snap = _json.loads(bet["full_trace"]).get("input_snapshot") or {}
            except Exception:  # noqa: BLE001
                input_snap = {}
            home = canonicalize(input_snap.get("home_team", ""))
            away = canonicalize(input_snap.get("away_team", ""))
            if not home or not away:
                skipped.append(f"{bet['bet_id']} (cannot resolve event teams from trace)")
                continue

            event = events_by_pair.get((home, away))
            if event is None:
                skipped.append(f"{bet['bet_id']} ({away} @ {home}: not in snapshot at {actual_ts})")
                continue

            if bet["market"].startswith("player_prop:"):
                stat_key = bet["market"].split(":", 1)[1]
                provider_market = provider_market_for_prop(league, stat_key)
                if not provider_market:
                    skipped.append(f"{bet['bet_id']} (no provider prop mapping for {stat_key})")
                    continue
                try:
                    prop_snapshot = client.fetch_historical_event_odds(
                        league,
                        event_id=event.event_id,
                        date=ts,
                        markets=provider_market,
                        bookmakers=str(bet["book"]).lower(),
                    )
                    event = prop_snapshot.events[0] if prop_snapshot.events else None
                    if event is None:
                        skipped.append(
                            f"{bet['bet_id']} (historical event odds returned no events)"
                        )
                        continue
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{bet['bet_id']} (prop historical fetch failed: {exc})")
                    continue
                bet = {**bet, "league": league}
                outcome = _match_prop_outcome(bet, event.books)
            else:
                outcome = _match_outcome(
                    bet_market=bet["market"],
                    descriptor=bet["selection_descriptor"],
                    home=event.home_team,
                    away=event.away_team,
                    books=event.books,
                    book_preference=bet["book"],
                )

            if outcome is None:
                skipped.append(
                    f"{bet['bet_id']} (no book matched market+descriptor in historical snapshot)"
                )
                continue

            if dry_run:
                logger.info(
                    "DRY [%s] %s -> %s %s @ %s pt=%s (snapshot=%s)",
                    league,
                    bet["bet_id"],
                    outcome.bookmaker,
                    outcome.market,
                    outcome.price,
                    outcome.point,
                    actual_ts,
                )
                attached += 1
                continue

            store.attach_closing_line(
                trace_id=bet["trace_id"],
                market=bet["market"],
                selection_descriptor=bet["selection_descriptor"],
                closing_odds=outcome.price,
                closing_line=outcome.point,
                closing_timestamp=actual_ts,
                source=f"the-odds-api-historical:{outcome.bookmaker}",
            )
            attached += 1
            logger.info(
                "ATTACHED [%s] %s -> %s %s pt=%s (snapshot=%s book=%s)",
                league,
                bet["bet_id"],
                outcome.price,
                bet["market"],
                outcome.point,
                actual_ts,
                outcome.bookmaker,
            )

    return attached, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missed closing-line windows via the paid Odds API historical endpoint"
    )
    parser.add_argument("--league", default=None, help="Restrict to one league (NBA, MLB, NFL, â€¦)")
    parser.add_argument(
        "--since",
        default=None,
        help="Earliest decision_timestamp to consider (ISO-8601 or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Latest decision_timestamp to consider (ISO-8601 or YYYY-MM-DD)",
    )
    parser.add_argument("--trace-id", dest="trace_id", default=None, help="Backfill a single trace")
    parser.add_argument(
        "--at",
        default=None,
        help="Exact ISO-8601 UTC timestamp to pass to the historical endpoint for all matched bets. "
        "If omitted, defaults to decision_timestamp + --close-offset-hours.",
    )
    parser.add_argument(
        "--close-offset-hours",
        dest="close_offset_hours",
        type=int,
        default=_DEFAULT_CLOSE_OFFSET_HOURS,
        help=f"Hours to add to decision_timestamp to infer the close time (default: {_DEFAULT_CLOSE_OFFSET_HOURS}). "
        "Ignored when --at is provided.",
    )
    parser.add_argument(
        "--db", default=None, help="SQLite path (default: var/omega_traces.db)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    require_sqlite_backend("backfill_closing_lines.py")

    store = TraceStore(db_path=args.db)
    pending = _pending_bets_needing_close(
        store,
        league_filter=args.league,
        since=args.since,
        until=args.until,
        trace_id=args.trace_id,
    )

    if not pending:
        logger.info("No pending bets needing closing-line backfill.")
        store.close()
        return 0

    by_league: dict[str, list[dict]] = {}
    for bet in pending:
        by_league.setdefault((bet["league"] or "").upper(), []).append(bet)

    logger.info(
        "Found %d pending bet(s) across %d league(s): %s",
        len(pending),
        len(by_league),
        sorted(by_league),
    )

    try:
        client = OddsApiClient()
    except OddsApiKeyMissing:
        logger.error("OMEGA_ODDS_API_KEY is not set. Export it or add it to .env before running.")
        store.close()
        return 1

    total_attached = 0
    total_skipped: list[str] = []

    for league in sorted(by_league):
        if not league:
            total_skipped.extend(f"{b['bet_id']} (no league on trace)" for b in by_league[league])
            continue
        attached, skipped = _process_league(
            league=league,
            bets=by_league[league],
            client=client,
            store=store,
            explicit_at=args.at,
            close_offset_hours=args.close_offset_hours,
            dry_run=args.dry_run,
        )
        total_attached += attached
        total_skipped.extend(skipped)

    logger.info("Done. attached=%d skipped=%d", total_attached, len(total_skipped))
    if total_skipped:
        logger.warning("Skipped bets:")
        for line in total_skipped:
            logger.warning("  - %s", line)

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())




