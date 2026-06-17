"""
omega.ops.capture_early_lines â€” capture low-liquidity early lines (Phase 7).

Some leagues (WNBA today, future low-liquidity sports) open with inefficient,
violently-moving lines. Locking in size early can be valuable, but those early
prices do NOT reflect closing probability. Blending them into the canonical
``closing_lines`` table would destroy CLV as a metric and bias calibration
toward phantom edges (Phase 7 red-team finding 4).

This cron shim writes early captures to the dedicated ``early_market_snapshots``
table ONLY â€” never ``closing_lines``. The canonical CLV computation reads only
``closing_lines`` and never joins this table, so isolation is structural. The
calibration fitter excludes early-market-tagged traces by default.

Only leagues whose ``leagues.py`` config carries ``liquidity_profile == "low"``
are eligible; passing a normal-liquidity league is a no-op with a warning, so an
operator cannot accidentally pollute the early table with mainstream markets.

Markets captured: h2h (moneyline), spreads, totals.

Usage:
    OMEGA_ODDS_API_KEY=xxxx omega-capture-early-lines --leagues WNBA
    omega-capture-early-lines --leagues WNBA --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.config.leagues import get_league_config  # noqa: E402
from omega.integrations.odds_api import (  # noqa: E402
    OddsApiBudgetExceeded,
    OddsApiClient,
    OddsApiKeyMissing,
    sport_key_for,
)
from omega.trace.market_snapshot import EarlyMarketSnapshot  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("capture_early_lines")

# Provider market -> (omega market label, descriptor side builder).
_MARKET_LABELS = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "total",
}


def _descriptor(market_label: str, selection: str, point: float | None) -> str:
    """Stable snake-ish selection descriptor for an early line."""
    sel = selection.strip().lower().replace(" ", "_")
    if point is None:
        return f"{market_label}:{sel}"
    return f"{market_label}:{sel}:{point}"


def _capture_league(
    league: str,
    client: OddsApiClient,
    store: TraceStore,
    dry_run: bool,
) -> tuple[int, list[str]]:
    """Fetch one snapshot for *league* and record early-market rows.

    Returns ``(recorded_count, skipped_messages)``.
    """
    config = get_league_config(league)
    liquidity_profile = config.get("liquidity_profile")
    if liquidity_profile != "low":
        return 0, [
            f"{league}: liquidity_profile={liquidity_profile!r} (early capture is "
            "for low-liquidity leagues only) â€” skipped"
        ]

    if sport_key_for(league) is None:
        return 0, [f"{league}: no sport_key mapping in odds_api.py â€” skipped"]

    try:
        events = client.fetch_event_odds(league)
    except OddsApiKeyMissing:
        logger.error("OMEGA_ODDS_API_KEY not set; cannot fetch %s early lines.", league)
        return 0, [f"{league}: api key missing"]
    except OddsApiBudgetExceeded as exc:
        logger.error("the-odds-api budget exceeded for %s: %s", league, exc)
        return 0, [f"{league}: budget exceeded ({exc})"]
    except Exception as exc:  # noqa: BLE001
        logger.error("Odds API fetch failed for %s: %s", league, exc)
        return 0, [f"{league}: api fetch failed ({exc})"]

    captured_at = datetime.now(UTC).isoformat()
    recorded = 0
    skipped: list[str] = []

    for event in events:
        # Pre-decision captures are not yet tied to a decision trace_id; key them
        # to the provider event so rows are stable and idempotent.
        event_ref = f"event:{event.event_id}"
        for book in event.books:
            market_label = _MARKET_LABELS.get(book.market)
            if market_label is None:
                continue
            descriptor = _descriptor(market_label, book.selection, book.point)
            snapshot = EarlyMarketSnapshot(
                trace_id=event_ref,
                league=league.upper(),
                market=market_label,
                selection_descriptor=descriptor,
                early_line=book.point,
                early_odds=book.price,
                liquidity_profile=liquidity_profile,
                captured_at=captured_at,
                source=f"the-odds-api:{book.bookmaker}",
            )
            if dry_run:
                logger.info(
                    "DRY [%s] %s %s @ %s pt=%s (book=%s)",
                    league,
                    event_ref,
                    descriptor,
                    book.price,
                    book.point,
                    book.bookmaker,
                )
                recorded += 1
                continue
            store.record_early_market_snapshot(snapshot)
            recorded += 1

    logger.info("[%s] recorded %d early-market rows from %d events", league, recorded, len(events))
    return recorded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture low-liquidity early lines into early_market_snapshots"
    )
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument(
        "--leagues",
        required=True,
        help="Comma-separated low-liquidity league codes (e.g. WNBA).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    leagues = [lg.strip().upper() for lg in args.leagues.split(",") if lg.strip()]
    if not leagues:
        logger.error("No leagues provided.")
        return 2

    store = TraceStore(db_path=args.db)
    client = OddsApiClient()
    total_recorded = 0
    total_skipped: list[str] = []

    for league in leagues:
        recorded, skipped = _capture_league(league, client, store, args.dry_run)
        total_recorded += recorded
        total_skipped.extend(skipped)

    logger.info("Done. recorded=%d skipped=%d", total_recorded, len(total_skipped))
    for line in total_skipped:
        logger.warning("  - %s", line)

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())





