"""
scripts/fetch_closing_lines.py — capture market close for bets needing CLV resolution.

This is the live post-decision Odds API capture path. The canonical write path is
still `scripts/ingest_closing_lines.py` for agent-emitted or manually reviewed
snapshots, but Cowork automation may use this script when a current close is
needed before tip-off. Paid historical backfill should use the historical
methods on `omega.integrations.odds_api`.

POST-DECISION ONLY. This is the **only** consumer of omega.integrations.odds_api.
Pre-decision line sourcing happens inside the LLM via WebFetch on direct
sportsbook pages (see prompts/system_prompt.txt §6.1.5). Do not extend this
script to fetch lines before a decision is made.

Sport-agnostic. Iterates the leagues found in pending bet_records and fetches
one snapshot per league from the-odds-api. Add new leagues by extending
``SPORT_KEY_MAP`` in omega/integrations/odds_api.py.

Workflow:
    1. Pull bet_records (status=pending) whose trace has no matching
       closing_line row.
    2. Group by trace.league. Skip leagues without a sport_key mapping.
    3. For each league, fetch the current odds snapshot (one API request).
    4. For each bet, locate the matching event + bookmaker + market + selection
       in the snapshot. Persist via TraceStore.attach_closing_line().

Markets captured: h2h, spreads, totals, and mapped event-level player props.

Schedule this shortly before tip-off for each event so the snapshot truly
represents the closing line. Paid historical endpoints can backfill missed
windows when coverage exists.

Phase 6g note: paid historical endpoints are now available through
`OddsApiClient`; use them for missed-window backfill and reproducible replay
artifacts instead of treating this live script as the only CLV source.

Usage:
    OMEGA_ODDS_API_KEY=xxxxxxxx python scripts/fetch_closing_lines.py
    python scripts/fetch_closing_lines.py --dry-run
    python scripts/fetch_closing_lines.py --league NBA
    python scripts/fetch_closing_lines.py --league NFL --dry-run
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.integrations.odds_api import (  # noqa: E402
    BookOdds,
    EventOdds,
    OddsApiBudgetExceeded,
    OddsApiClient,
    OddsApiKeyMissing,
    sport_key_for,
)
from omega.trace.store import TraceStore  # noqa: E402
from scripts.resolve_odds import provider_market_for_prop  # noqa: E402

logger = logging.getLogger("fetch_closing_lines")


# ---------------------------------------------------------------------------
# Team canonicalization registry
# ---------------------------------------------------------------------------

def _identity(name: str) -> Optional[str]:
    return name or None


def _load_canonicalizer(league: str) -> Callable[[str], Optional[str]]:
    """Per-league team-name canonicalizer. Falls back to identity when no
    omega.integrations.espn_{league} module ships a ``canonical_team``."""
    league_lc = league.lower()
    candidates = [
        f"omega.integrations.espn_{league_lc}",
        f"omega.integrations.team_aliases_{league_lc}",
    ]
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        fn = getattr(mod, "canonical_team", None)
        if callable(fn):
            return fn
    return _identity


# ---------------------------------------------------------------------------
# Descriptor parsing
# ---------------------------------------------------------------------------

_MARKET_MAP = {
    "moneyline": "h2h",
    "spread": "spreads",
    "total": "totals",
}


def _is_supported_market(bet_market: str) -> bool:
    return bet_market in _MARKET_MAP or bet_market.startswith("player_prop:")


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


def _match_prop_outcome(bet: Dict, books: List[BookOdds]) -> Optional[BookOdds]:
    """Find the provider row for an exact player-prop close."""
    stat_key = bet["market"].split(":", 1)[1]
    provider_market = provider_market_for_prop(bet["league"], stat_key)
    if not provider_market:
        return None
    desc = f"{bet.get('selection_descriptor', '')} {bet.get('selection', '')}".lower()
    if "over" in desc:
        side = "over"
    elif "under" in desc:
        side = "under"
    else:
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
    store: TraceStore, league_filter: Optional[str] = None,
) -> List[Dict]:
    """Return rows joining bet_records, traces, closing_lines — only pending bets
    with no closing-line attached yet. Optionally filter to a single league."""
    sql = (
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
           WHERE b.status = 'pending'
             AND c.closing_id IS NULL"""
    )
    params: tuple = ()
    if league_filter:
        sql += " AND t.league = ?"
        params = (league_filter.upper(),)
    rows = store.conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _event_key(home: str, away: str, canonicalize: Callable[[str], Optional[str]]) -> Tuple[str, str]:
    return (canonicalize(home) or home, canonicalize(away) or away)


# ---------------------------------------------------------------------------
# Per-league processing
# ---------------------------------------------------------------------------

def _process_league(
    league: str,
    bets: List[Dict],
    client: OddsApiClient,
    store: TraceStore,
    dry_run: bool,
) -> Tuple[int, List[str]]:
    """Fetch one snapshot for `league`, attach closing lines for `bets`.
    Returns ``(attached_count, skipped_messages)``."""
    sport_key = sport_key_for(league)
    if sport_key is None:
        return 0, [f"{b['bet_id']} (league {league!r} has no sport_key mapping)" for b in bets]

    try:
        events = client.fetch_event_odds(league)
    except OddsApiKeyMissing:
        logger.error("OMEGA_ODDS_API_KEY not set; cannot fetch %s closing lines.", league)
        return 0, [f"{b['bet_id']} (api key missing)" for b in bets]
    except OddsApiBudgetExceeded as exc:
        logger.error("the-odds-api budget exceeded for %s: %s", league, exc)
        return 0, [f"{b['bet_id']} (budget exceeded)" for b in bets]
    except Exception as exc:  # noqa: BLE001
        logger.error("Odds API fetch failed for %s: %s", league, exc)
        return 0, [f"{b['bet_id']} (api fetch failed: {exc})" for b in bets]

    logger.info(
        "[%s] fetched %d events; remaining budget=%d",
        league, len(events), client.remaining_budget(),
    )

    canonicalize = _load_canonicalizer(league)
    events_by_pair: Dict[Tuple[str, str], EventOdds] = {
        _event_key(e.home_team, e.away_team, canonicalize): e for e in events
    }

    attached = 0
    skipped: List[str] = []

    for bet in bets:
        if not _is_supported_market(bet["market"]):
            skipped.append(f"{bet['bet_id']} (unsupported market: {bet['market']})")
            continue

        full_trace = bet["full_trace"]
        try:
            import json as _json
            input_snap = (_json.loads(full_trace).get("input_snapshot") or {})
        except Exception:  # noqa: BLE001
            input_snap = {}
        home = canonicalize(input_snap.get("home_team", ""))
        away = canonicalize(input_snap.get("away_team", ""))
        if not home or not away:
            skipped.append(f"{bet['bet_id']} (cannot resolve event teams from trace)")
            continue

        event = events_by_pair.get((home, away))
        if event is None:
            skipped.append(f"{bet['bet_id']} ({away} @ {home}: not in {league} snapshot)")
            continue

        if bet["market"].startswith("player_prop:"):
            stat_key = bet["market"].split(":", 1)[1]
            provider_market = provider_market_for_prop(league, stat_key)
            if not provider_market:
                skipped.append(f"{bet['bet_id']} (no provider prop mapping for {stat_key})")
                continue
            try:
                event = client.fetch_current_event_odds(
                    league,
                    event.event_id,
                    markets=provider_market,
                    bookmakers=str(bet["book"]).lower(),
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{bet['bet_id']} (prop event odds fetch failed: {exc})")
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
            skipped.append(f"{bet['bet_id']} (no book matched market+descriptor)")
            continue

        if dry_run:
            logger.info("DRY [%s] %s -> %s %s @ %s pt=%s",
                        league, bet["bet_id"], outcome.bookmaker, outcome.market,
                        outcome.price, outcome.point)
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
        logger.info("ATTACHED [%s] %s -> %s %s pt=%s (book=%s)",
                    league, bet["bet_id"], outcome.price, bet["market"],
                    outcome.point, outcome.bookmaker)

    return attached, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Capture closing lines for pending bets across mapped leagues")
    parser.add_argument("--db", default=None, help="SQLite path (default: repo-root omega_traces.db)")
    parser.add_argument(
        "--league", default=None,
        help="Restrict to a single league code (e.g. NBA, NFL, MLB). Default: process all leagues found in pending bets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = TraceStore(db_path=args.db)
    pending = _pending_bets_needing_close(store, league_filter=args.league)
    if not pending:
        logger.info("No pending bets needing closing-line capture.")
        store.close()
        return 0

    # Group by league
    by_league: Dict[str, List[Dict]] = {}
    for bet in pending:
        by_league.setdefault((bet["league"] or "").upper(), []).append(bet)

    logger.info(
        "Found %d pending bet(s) across %d league(s): %s",
        len(pending), len(by_league), sorted(by_league),
    )

    client = OddsApiClient()
    total_attached = 0
    total_skipped: List[str] = []

    for league in sorted(by_league):
        if not league:
            total_skipped.extend(f"{b['bet_id']} (no league on trace)" for b in by_league[league])
            continue
        attached, skipped = _process_league(
            league=league, bets=by_league[league],
            client=client, store=store, dry_run=args.dry_run,
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
