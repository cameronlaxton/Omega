"""
scripts/fetch_closing_lines_nba.py — capture NBA market close for bets needing CLV resolution.

Workflow:
    1. Pull bet_records (status=pending) whose trace has no matching
       closing_line row. Limit to NBA traces.
    2. Group by event (home_team, away_team) so we minimize API calls.
    3. Fetch the current NBA odds snapshot via OddsApiClient (one request).
    4. For each bet, locate the matching event + bookmaker + market + selection
       in the snapshot. Persist via TraceStore.attach_closing_line().

This script is intended to run shortly before tip-off — the "closing line" is
the live snapshot taken inside a small window before the game starts. Schedule
it accordingly (the free tier of the-odds-api does not include historical
endpoints).

Usage:
    OMEGA_ODDS_API_KEY=xxxxxxxx python scripts/fetch_closing_lines_nba.py
    python scripts/fetch_closing_lines_nba.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.integrations.espn_nba import canonical_team  # noqa: E402
from omega.integrations.odds_api import (  # noqa: E402
    BookOdds,
    EventOdds,
    OddsApiBudgetExceeded,
    OddsApiClient,
    OddsApiKeyMissing,
)
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("fetch_closing_lines_nba")


# ---------------------------------------------------------------------------
# Descriptor parsing
# ---------------------------------------------------------------------------

_MARKET_MAP = {
    "moneyline": "h2h",
    "spread": "spreads",
    "total": "totals",
}


def _is_supported_market(bet_market: str) -> bool:
    return bet_market in _MARKET_MAP


def _match_outcome(
    bet_market: str,
    descriptor: str,
    home: str,
    away: str,
    books: List[BookOdds],
    book_preference: Optional[str],
) -> Optional[BookOdds]:
    """Find the BookOdds row matching the descriptor.

    Descriptor conventions (from system_prompt.txt §11):
      moneyline:  home_moneyline | away_moneyline
      spread:     home_spread_<line> | away_spread_<line>
      total:      total_over_<line> | total_under_<line>

    `book_preference` (e.g. "draftkings") narrows to a single book if available.
    Otherwise the first matching book wins.
    """
    odds_market = _MARKET_MAP[bet_market]
    candidates = [b for b in books if b.market == odds_market]
    if book_preference:
        prefs = [b for b in candidates if b.bookmaker == book_preference.lower()]
        if prefs:
            candidates = prefs

    desc = descriptor.lower()
    if bet_market == "moneyline":
        if "home" in desc:
            target = home
        elif "away" in desc:
            target = away
        else:
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
        if "over" in desc:
            label = "Over"
        elif "under" in desc:
            label = "Under"
        else:
            return None
        for b in candidates:
            if b.selection == label:
                return b
        return None

    return None


# ---------------------------------------------------------------------------
# Bet selection
# ---------------------------------------------------------------------------

def _pending_bets_needing_close(store: TraceStore) -> List[Dict]:
    """Return rows joining bet_records, traces, closing_lines — only pending bets
    with no closing-line attached yet."""
    rows = store.conn.execute(
        """SELECT b.bet_id, b.trace_id, b.book, b.market, b.selection,
                  b.selection_descriptor, b.line_taken, b.odds_taken,
                  b.decision_timestamp, b.status,
                  t.league, t.full_trace
           FROM bet_records b
           JOIN traces t ON t.trace_id = b.trace_id
           LEFT JOIN closing_lines c
             ON c.trace_id = b.trace_id
            AND c.market = b.market
            AND c.selection_descriptor = b.selection_descriptor
           WHERE t.league = 'NBA'
             AND b.status = 'pending'
             AND c.closing_id IS NULL"""
    ).fetchall()
    return [dict(r) for r in rows]


def _event_key(home: str, away: str) -> Tuple[str, str]:
    return (canonical_team(home) or home, canonical_team(away) or away)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Capture NBA closing lines for pending bets")
    parser.add_argument("--db", default=None, help="SQLite path (default: repo-root omega_traces.db)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = TraceStore(db_path=args.db)
    pending = _pending_bets_needing_close(store)
    if not pending:
        logger.info("No pending bets needing closing-line capture.")
        store.close()
        return 0

    logger.info("Found %d pending bet(s) needing closing-line capture.", len(pending))

    # Fetch one snapshot of NBA odds — covers all pending events.
    client = OddsApiClient()
    try:
        events = client.fetch_nba_odds()
    except OddsApiKeyMissing:
        logger.error("OMEGA_ODDS_API_KEY not set; cannot fetch closing lines.")
        store.close()
        return 1
    except OddsApiBudgetExceeded as exc:
        logger.error("the-odds-api budget exceeded: %s", exc)
        store.close()
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Odds API fetch failed: %s", exc)
        store.close()
        return 1

    logger.info("Fetched %d NBA events; remaining budget=%d", len(events), client.remaining_budget())

    events_by_pair: Dict[Tuple[str, str], EventOdds] = {
        _event_key(e.home_team, e.away_team): e for e in events
    }

    attached = 0
    skipped: List[str] = []

    for bet in pending:
        if not _is_supported_market(bet["market"]):
            skipped.append(f"{bet['bet_id']} (unsupported market: {bet['market']})")
            continue

        # Pull home/away from the trace's input_snapshot
        full_trace = bet["full_trace"]
        try:
            import json as _json
            input_snap = (_json.loads(full_trace).get("input_snapshot") or {})
        except Exception:  # noqa: BLE001
            input_snap = {}
        home = canonical_team(input_snap.get("home_team", ""))
        away = canonical_team(input_snap.get("away_team", ""))
        if not home or not away:
            skipped.append(f"{bet['bet_id']} (cannot resolve event teams from trace)")
            continue

        event = events_by_pair.get((home, away))
        if event is None:
            skipped.append(f"{bet['bet_id']} ({away} @ {home}: not in current odds snapshot)")
            continue

        outcome = _match_outcome(
            bet_market=bet["market"],
            descriptor=bet["selection_descriptor"],
            home=event.home_team,
            away=event.away_team,
            books=event.books,
            book_preference=bet["book"],
        )
        if outcome is None:
            skipped.append(f"{bet['bet_id']} (no book matched market+descriptor)")
            continue

        if args.dry_run:
            logger.info("DRY %s -> %s %s @ %s pt=%s",
                        bet["bet_id"], outcome.bookmaker, outcome.market, outcome.price, outcome.point)
            attached += 1
            continue

        store.attach_closing_line(
            trace_id=bet["trace_id"],
            market=bet["market"],
            selection_descriptor=bet["selection_descriptor"],
            closing_odds=outcome.price,
            closing_line=outcome.point,
            closing_timestamp=outcome.last_update or datetime.now(timezone.utc).isoformat(),
            source=f"the-odds-api:{outcome.bookmaker}",
        )
        attached += 1
        logger.info("ATTACHED %s -> %s %s pt=%s (book=%s)",
                    bet["bet_id"], outcome.price, bet["market"], outcome.point, outcome.bookmaker)

    logger.info("Done. attached=%d skipped=%d", attached, len(skipped))
    if skipped:
        logger.warning("Skipped bets:")
        for line in skipped:
            logger.warning("  - %s", line)

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
