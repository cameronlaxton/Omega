"""
omega.ops.settle_bets - settle pending bet_ledger rows with attached outcomes.

Usage:
    python -m omega.ops.settle_bets
    python -m omega.ops.settle_bets --apply
    python -m omega.ops.settle_bets --league NBA --provenance all --apply
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.ledger_settlement import (  # noqa: E402
    SettlementSummary,
    auto_void_aged_pending,
    settle_pending_ledger,
)
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("settle_bets")


def _print_summary(summary: SettlementSummary, *, apply: bool, provenance: str | None) -> None:
    staked = summary.total_staked
    net = summary.total_net
    roi = (net / staked * 100.0) if staked else 0.0
    settled_total = sum(summary.settled.values())

    logger.info("Bet-ledger settlement summary")
    logger.info("-----------------------------")
    logger.info("Mode:                         %s", "APPLY" if apply else "DRY-RUN")
    logger.info("Provenance:                   %s", provenance or "all")
    logger.info("Pending rows scanned:         %d", summary.pending_scanned)
    logger.info("Settled rows:                 %d", settled_total)
    logger.info(
        "  won/lost/push/void:         %d / %d / %d / %d",
        summary.settled.get("won", 0),
        summary.settled.get("lost", 0),
        summary.settled.get("push", 0),
        summary.settled.get("void", 0),
    )
    logger.info("Ungradeable/no outcome:       %d", summary.ungradeable)
    logger.info("Total staked (settled):       $%.2f", staked)
    logger.info("Total net PnL (settled):      $%.2f", net)
    logger.info("ROI:                          %.1f%%", roi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Settle pending bet_ledger rows from attached outcomes."
    )
    parser.add_argument("--db", type=str, default=None, help="SQLite path")
    parser.add_argument("--league", type=str, default=None, help="Limit to one league")
    parser.add_argument("--sport", type=str, default=None, help="Limit to one sport")
    parser.add_argument(
        "--provenance",
        type=str,
        default="user_confirmed",
        choices=["user_confirmed", "engine_auto", "backfill", "all"],
        help="Ledger provenance to settle (default: user_confirmed)",
    )
    parser.add_argument("--start", type=str, default=None, help="Min decision timestamp")
    parser.add_argument("--end", type=str, default=None, help="Max decision timestamp")
    parser.add_argument("--limit", type=int, default=100000, help="Max pending rows to scan")
    parser.add_argument(
        "--auto-void-older-than-days",
        type=int,
        default=None,
        help=(
            "After the normal settle pass, VOID any still-pending rows with no "
            "attached outcome that are older than N days (event has almost "
            "certainly happened; the gap is missing outcome data). Rows that "
            "ARE gradeable are settled normally above, never voided."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Write settled rows")
    parser.add_argument("--dry-run", action="store_true", help="Scan only (default)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    apply = bool(args.apply) and not args.dry_run
    provenance = None if args.provenance == "all" else args.provenance

    try:
        store = TraceStore(db_path=args.db)
        log_effective_db(store, logger)
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot open trace store: %s", exc)
        return 1

    try:
        summary = settle_pending_ledger(
            store,
            apply=apply,
            league=args.league,
            sport=args.sport,
            provenance=provenance,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
        _print_summary(summary, apply=apply, provenance=provenance)

        if args.auto_void_older_than_days is not None:
            void_summary = auto_void_aged_pending(
                store,
                older_than_days=args.auto_void_older_than_days,
                apply=apply,
                league=args.league,
                sport=args.sport,
                provenance=provenance,
                limit=args.limit,
            )
            logger.info("")
            logger.info(
                "Auto-void pass (pending, no outcome, older than %dd)",
                args.auto_void_older_than_days,
            )
            logger.info("-----------------------------------------------------")
            logger.info("Mode:                         %s", "APPLY" if apply else "DRY-RUN")
            logger.info("Voided rows:                  %d", void_summary.settled.get("void", 0))
            logger.info("Total staked (voided):        $%.2f", void_summary.total_staked)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
