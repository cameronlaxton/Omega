"""
omega.ops.backfill_bets — log historical traces into the dollar bet_ledger.

Iterates persisted traces in var/omega_traces.db and, for each trace that does
not already have a ledger row, extracts the engine's single recommended
selection (game best edge or prop over/under) and logs it as a flat-stake bet
(default $25 on a $1000 bankroll) with provenance='backfill'. Where the trace
already has an attached outcome (outcomes / prop_outcomes), the bet is graded to
won/lost/push and its dollar PnL computed; otherwise it lands 'pending'.

This is one-bet-per-trace by design. Traces that recommended a pass, have no
actionable edge, carry unparseable odds, or are slates are skipped and counted.

SAFETY: writes ONLY to bet_ledger. The bet_records table (units-based CLV
substrate that drives CLV + the live prop-outcome sweep) is never touched, so
no phantom wagers leak into those pipelines.

Usage:
    python -m omega.ops.backfill_bets                 # dry-run (default)
    python -m omega.ops.backfill_bets --apply         # writes
    python -m omega.ops.backfill_bets --league NBA --start 2026-01-01 --apply

Exit codes:
    0 — scan completed (dry-run or apply)
    1 — fatal error (DB missing/unreadable)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.bet_settlement import (  # noqa: E402
    ExtractResult,
    extract_recommended_bet,
)
from omega.trace.ledger_bet import (  # noqa: E402
    DEFAULT_BANKROLL,
    DEFAULT_STAKE_AMOUNT,
    BetProvenance,
    LedgerBet,
    LedgerStatus,
)
from omega.trace.ledger_settlement import (  # noqa: E402
    grade_ledger_bet,
    grade_ledger_fields,
    settle_pending_ledger,
)
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("backfill_bets")


@dataclass
class BackfillSummary:
    traces_scanned: int = 0
    already_present: int = 0
    eligible: int = 0
    inserted: int = 0
    skipped: Counter = field(default_factory=Counter)  # keyed by extract reason
    graded: Counter = field(default_factory=Counter)  # newly-inserted, graded on insert
    regraded: Counter = field(default_factory=Counter)  # previously-pending rows graded now
    pending: int = 0
    total_staked: float = 0.0
    total_net: float = 0.0


def _iter_trace_rows(store: TraceStore, *, league, start, end, limit):
    clauses: list[str] = []
    params: list = []
    if league:
        clauses.append("league = ?")
        params.append(league)
    if start:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end:
        clauses.append("timestamp <= ?")
        params.append(end)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT trace_id, full_trace FROM traces {where} ORDER BY timestamp"
    if limit:
        sql += f" LIMIT {int(limit)}"
    for row in store.conn.execute(sql, params).fetchall():
        try:
            trace = json.loads(row["full_trace"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("trace %s has unreadable full_trace; skipping", row["trace_id"])
            continue
        yield row["trace_id"], trace


def _ledger_exists(store: TraceStore, bet: LedgerBet) -> bool:
    row = store.conn.execute(
        "SELECT 1 FROM bet_ledger WHERE trace_id = ? AND market = ? "
        "AND selection_descriptor = ? LIMIT 1",
        (bet.trace_id, bet.market, bet.selection_descriptor),
    ).fetchone()
    return row is not None


def _grade_fields(
    store: TraceStore,
    *,
    trace_id: str,
    market: str,
    selection_descriptor: str,
    line: float | None,
    odds: float,
    stake: float,
) -> tuple[LedgerStatus, float | None, float | None] | None:
    """Grade a bet against an attached outcome. None => leave pending."""
    return grade_ledger_fields(
        store,
        trace_id=trace_id,
        market=market,
        selection_descriptor=selection_descriptor,
        line=line,
        odds=odds,
        stake=stake,
    )


def _grade(store: TraceStore, bet: LedgerBet) -> tuple[LedgerStatus, float | None, float | None] | None:
    return grade_ledger_bet(store, bet)


def regrade_pending(store: TraceStore, *, apply: bool, summary: BackfillSummary) -> None:
    """Grade ledger rows still 'pending' that now have an attached outcome.

    This is what lets a later run pick up bets logged earlier: log first
    (pending), fetch outcomes, then re-run — these rows get settled in place.
    Idempotent: a row that stays ungradeable is left pending.
    """
    settled = settle_pending_ledger(store, apply=apply, provenance=None)
    summary.regraded.update(settled.settled)
    summary.total_staked += settled.total_staked
    summary.total_net += settled.total_net


def run_backfill(
    store: TraceStore,
    *,
    apply: bool = False,
    league: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    stake: float = DEFAULT_STAKE_AMOUNT,
    bankroll: float = DEFAULT_BANKROLL,
) -> BackfillSummary:
    summary = BackfillSummary()

    for tid, trace in _iter_trace_rows(
        store, league=league, start=start, end=end, limit=limit
    ):
        summary.traces_scanned += 1
        result: ExtractResult = extract_recommended_bet(
            trace,
            provenance=BetProvenance.BACKFILL,
            stake_amount=stake,
            bankroll=bankroll,
        )
        if result.bet is None:
            summary.skipped[result.reason] += 1
            continue

        bet = result.bet
        if _ledger_exists(store, bet):
            summary.already_present += 1
            continue

        summary.eligible += 1

        grade = _grade(store, bet)
        if grade is not None:
            status, payout, net = grade
            bet.status = status
            bet.payout_amount = payout
            bet.net_pnl = net
            summary.graded[status.value] += 1
            summary.total_staked += bet.stake_amount
            if net is not None:
                summary.total_net += net
        else:
            summary.pending += 1

        if apply:
            store.record_ledger_bet(bet)
            if grade is not None:
                store.grade_ledger_bet(
                    bet.ledger_id, bet.status, bet.payout_amount, bet.net_pnl
                )
            logger.debug(
                "logged %s %s @ %s [%s]",
                bet.market,
                bet.selection_descriptor,
                bet.odds,
                bet.status.value,
            )

    # Second pass: settle any ledger rows still pending that now have an outcome
    # (covers bets logged on an earlier run, before outcomes were fetched).
    regrade_pending(store, apply=apply, summary=summary)

    return summary


def _print_summary(summary: BackfillSummary, *, apply: bool) -> None:
    staked = summary.total_staked
    net = summary.total_net
    roi = (net / staked * 100.0) if staked else 0.0
    wins = summary.graded.get("won", 0) + summary.regraded.get("won", 0)
    losses = summary.graded.get("lost", 0) + summary.regraded.get("lost", 0)
    decided = wins + losses
    win_pct = (wins / decided * 100.0) if decided else 0.0
    regraded_total = sum(summary.regraded.values())

    logger.info("Bet-ledger backfill summary")
    logger.info("---------------------------")
    logger.info("Mode:                          %s", "APPLY" if apply else "DRY-RUN")
    logger.info("Traces scanned:                %d", summary.traces_scanned)
    logger.info("Already in ledger:             %d", summary.already_present)
    logger.info("Eligible (new bets):           %d", summary.eligible)
    logger.info("  graded won/lost/push/void:   %d / %d / %d / %d",
                summary.graded.get("won", 0), summary.graded.get("lost", 0),
                summary.graded.get("push", 0), summary.graded.get("void", 0))
    logger.info("  pending (no outcome):        %d", summary.pending)
    if regraded_total:
        logger.info("Re-graded prior pending rows:  %d (won %d / lost %d / push %d)",
                    regraded_total, summary.regraded.get("won", 0),
                    summary.regraded.get("lost", 0), summary.regraded.get("push", 0))
    for reason, n in sorted(summary.skipped.items()):
        logger.info("Skipped (%s):%s%d", reason, " " * max(1, 18 - len(reason)), n)
    logger.info("Total staked (graded):         $%.2f", staked)
    logger.info("Total net PnL (graded):        $%.2f", net)
    logger.info("ROI:                           %.1f%%", roi)
    logger.info("Win%% (excl. push/void):        %.1f%% (%d/%d)", win_pct, wins, decided)
    if apply:
        logger.info("Bets written:                  %d", summary.eligible)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill historical traces into the dollar bet_ledger (flat stake)."
    )
    parser.add_argument("--db", type=str, default=None, help="SQLite path")
    parser.add_argument("--league", type=str, default=None, help="Limit to one league")
    parser.add_argument("--start", type=str, default=None, help="Min trace timestamp (ISO)")
    parser.add_argument("--end", type=str, default=None, help="Max trace timestamp (ISO)")
    parser.add_argument("--limit", type=int, default=None, help="Max traces to scan")
    parser.add_argument(
        "--stake", type=float, default=DEFAULT_STAKE_AMOUNT, help="Flat stake in dollars"
    )
    parser.add_argument(
        "--bankroll", type=float, default=DEFAULT_BANKROLL, help="Bankroll snapshot to record"
    )
    parser.add_argument("--apply", action="store_true", help="Write rows (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report only (default)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    apply = bool(args.apply) and not args.dry_run

    try:
        store = TraceStore(db_path=args.db)
        log_effective_db(store, logger)
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot open trace store: %s", exc)
        return 1

    summary = run_backfill(
        store,
        apply=apply,
        league=args.league,
        start=args.start,
        end=args.end,
        limit=args.limit,
        stake=args.stake,
        bankroll=args.bankroll,
    )
    _print_summary(summary, apply=apply)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
