"""
omega.trace.portfolio — pure aggregation over bet_ledger rows.

No DB access here: ``summarize_ledger`` is a deterministic transform over rows
returned by ``TraceStore.query_ledger`` so it can be unit-tested without a
database and reused by the ``omega_get_portfolio_summary`` MCP tool.

This computes *reporting* numbers only (counts, staked, net PnL, ROI, current
bankroll). It does not compute any protected betting output (probability, edge,
EV, Kelly, units, tiers) — those stay in the deterministic engine.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from omega.core.betting.odds import american_to_decimal
from omega.trace.ledger_bet import DEFAULT_BANKROLL

# Statuses that count as settled (money realized) vs. still-open.
_GRADED = frozenset({"won", "lost", "push", "void"})
_DECIDED = frozenset({"won", "lost"})  # exclude push/void from win%


def _f(value: Any) -> float:
    """Coerce a possibly-None numeric DB cell to float (None -> 0.0)."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _active_ledger(row: dict[str, Any]) -> dict[str, Any]:
    stake = round(_f(row.get("stake_amount")), 2)
    odds = _f(row.get("odds"))
    potential_payout = round(stake * american_to_decimal(odds), 2) if odds else 0.0
    last_updated = row.get("graded_at") or row.get("decision_timestamp") or row.get("created_at")
    return {
        "ledger_id": row.get("ledger_id"),
        "league": row.get("league"),
        "market_type": row.get("market"),
        "status": str(row.get("status") or "pending").lower(),
        "stake": stake,
        "potential_payout": potential_payout,
        "pnl": round(_f(row.get("net_pnl")), 2),
        "last_updated": last_updated,
    }


def summarize_ledger(
    rows: Iterable[dict[str, Any]],
    base_bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, Any]:
    """Summarize bet-ledger rows into a financial snapshot.

    Args:
        rows: ledger rows (dicts) as returned by ``TraceStore.query_ledger`` —
            each carries ``status``, ``stake_amount``, ``net_pnl``.
        base_bankroll: starting bankroll the PnL is measured against.

    Returns a JSON-friendly dict. ``roi_pct`` and ``win_pct`` are percentages
    (e.g. 4.2 == 4.2%); both are ``0.0`` when their denominator is empty.
    ``current_bankroll`` is ``base_bankroll`` plus net PnL over graded bets.
    """
    status_counts: dict[str, int] = {}
    total_bets = 0
    pending_count = 0
    pending_stake = 0.0
    total_staked = 0.0  # graded only
    net_pnl = 0.0  # graded only
    active_ledgers: list[dict[str, Any]] = []

    for row in rows:
        total_bets += 1
        status = str(row.get("status") or "pending").lower()
        status_counts[status] = status_counts.get(status, 0) + 1
        stake = _f(row.get("stake_amount"))
        if status in _GRADED:
            total_staked += stake
            net_pnl += _f(row.get("net_pnl"))
        else:
            pending_count += 1
            pending_stake += stake
            active_ledgers.append(_active_ledger(row))

    wins = status_counts.get("won", 0)
    losses = status_counts.get("lost", 0)
    decided = wins + losses

    roi_pct = (net_pnl / total_staked * 100.0) if total_staked else 0.0
    win_pct = (wins / decided * 100.0) if decided else 0.0

    return {
        "base_bankroll": round(base_bankroll, 2),
        "current_bankroll": round(base_bankroll + net_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "realized_pnl": round(net_pnl, 2),
        "unrealized_pnl": 0.0,
        "total_staked": round(total_staked, 2),
        "roi_pct": round(roi_pct, 2),
        "total_bets": total_bets,
        "pending_count": pending_count,
        "pending_stake": round(pending_stake, 2),
        "pending_exposure": round(pending_stake, 2),
        "open_positions_count": pending_count,
        "active_ledgers": active_ledgers,
        "won": wins,
        "lost": losses,
        "push": status_counts.get("push", 0),
        "void": status_counts.get("void", 0),
        "decided": decided,
        "win_pct": round(win_pct, 2),
        "status_counts": status_counts,
    }
