"""
omega.trace.ledger_settlement - settle pending bet_ledger rows.

This module owns DB orchestration for settlement only. It does not create ledger
rows, fetch outcomes, or duplicate the pure grading math in bet_settlement.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from omega.trace.bet_settlement import compute_pnl, settle_game_bet, settle_prop_bet
from omega.trace.ledger_bet import LedgerBet, LedgerStatus
from omega.trace.store import TraceStore

_GRADEABLE_GAME_SIDES = {"home", "away", "draw", "over", "under"}


@dataclass
class SettlementSummary:
    pending_scanned: int = 0
    settled: Counter = field(default_factory=Counter)
    ungradeable: int = 0
    total_staked: float = 0.0
    total_net: float = 0.0


def grade_ledger_fields(
    store: TraceStore,
    *,
    trace_id: str,
    market: str,
    selection_descriptor: str,
    line: float | None,
    odds: float,
    stake: float,
) -> tuple[LedgerStatus, float | None, float | None] | None:
    """Grade one ledger selection against an already-attached outcome."""
    if market.startswith("player_prop:"):
        row = next(iter(store.get_prop_outcomes(trace_id)), None)
        if row is None:
            return None
        rec_side = (
            "over"
            if "_over" in selection_descriptor
            else "under"
            if "_under" in selection_descriptor
            else ""
        )
        if not rec_side:
            return None
        status = settle_prop_bet(rec_side, row["result"], row["side"])
    else:
        row = store.get_outcome(trace_id)
        if row is None:
            return None
        side = selection_descriptor.split("_", 1)[0]
        if side not in _GRADEABLE_GAME_SIDES:
            return None
        status = settle_game_bet(market, side, line, row["home_score"], row["away_score"])

    payout, net = compute_pnl(status, odds, stake)
    return status, payout, net


def grade_ledger_bet(
    store: TraceStore,
    bet: LedgerBet,
) -> tuple[LedgerStatus, float | None, float | None] | None:
    return grade_ledger_fields(
        store,
        trace_id=bet.trace_id,
        market=bet.market,
        selection_descriptor=bet.selection_descriptor,
        line=bet.line,
        odds=bet.odds,
        stake=bet.stake_amount,
    )


def _pending_rows(
    store: TraceStore,
    *,
    league: str | None,
    sport: str | None,
    provenance: str | None,
    start: str | None,
    end: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    return store.query_ledger(
        league=league,
        sport=sport,
        status="pending",
        provenance=provenance,
        start=start,
        end=end,
        limit=limit,
    )


def settle_pending_ledger(
    store: TraceStore,
    *,
    apply: bool = False,
    league: str | None = None,
    sport: str | None = None,
    provenance: str | None = "user_confirmed",
    start: str | None = None,
    end: str | None = None,
    limit: int = 100000,
) -> SettlementSummary:
    """Settle pending bet_ledger rows that already have attached outcomes."""
    summary = SettlementSummary()
    rows = _pending_rows(
        store,
        league=league,
        sport=sport,
        provenance=provenance,
        start=start,
        end=end,
        limit=limit,
    )
    for row in rows:
        summary.pending_scanned += 1
        graded = grade_ledger_fields(
            store,
            trace_id=row["trace_id"],
            market=row["market"],
            selection_descriptor=row["selection_descriptor"],
            line=row["line"],
            odds=row["odds"],
            stake=row["stake_amount"],
        )
        if graded is None:
            summary.ungradeable += 1
            continue

        status, payout, net = graded
        summary.settled[status.value] += 1
        summary.total_staked += row["stake_amount"]
        if net is not None:
            summary.total_net += net
        if apply:
            store.grade_ledger_bet(row["ledger_id"], status, payout, net)

    return summary
