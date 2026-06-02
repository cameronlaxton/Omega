"""
omega.trace.ledger_bet — dollar-denominated bet-log row tied to a trace.

A LedgerBet is the engine's RECOMMENDED selection on a trace, logged as a flat
dollar wager for PnL/ROI tracking and a future betting dashboard. This is
deliberately distinct from `bet_record.BetRecord`:

- BetRecord  → units-based, "what the user ACTUALLY wagered", feeds CLV and the
  live prop-outcome sweep. Must never see auto-logged/backfilled phantom bets.
- LedgerBet  → dollar-based (stake_amount, payout_amount, net_pnl), carries a
  `provenance` tag (backfill | engine_auto | user_confirmed), and lives in its
  own `bet_ledger` table so the CLV/grading semantics above stay clean.

One row per (trace_id, market, selection_descriptor).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

UTC = timezone.utc

# Defaults until dynamic unit sizing lands (see plan §3). 1 unit == 1% bankroll,
# so a $25 stake on a $1000 bankroll is 2.5 units — but bet_ledger tracks dollars
# directly, not units.
DEFAULT_STAKE_AMOUNT = 25.0
DEFAULT_BANKROLL = 1000.0


class LedgerStatus(str, Enum):
    """Lifecycle of a single ledger bet (mirrors BetStatus, dollar-graded)."""

    PENDING = "pending"  # Logged, outcome not yet attached
    WON = "won"
    LOST = "lost"
    PUSH = "push"  # Tied to the line/total — stake returned, net 0
    VOID = "void"  # Cancelled by book — stake returned, net 0


class BetProvenance(str, Enum):
    """How a ledger row came to exist."""

    BACKFILL = "backfill"  # Created by backfill_bets.py from historical traces
    ENGINE_AUTO = "engine_auto"  # Auto-logged by TraceStore.persist() dual-write
    USER_CONFIRMED = "user_confirmed"  # Reserved: a real wager the user confirmed


class LedgerBet(BaseModel):
    """One recommended selection logged as a flat dollar bet.

    `selection_descriptor` is the canonical form (snake_case, line embedded) used
    as the idempotency key alongside trace_id + market; `selection` is the
    human-readable label surfaced in the dashboard. Money fields are None while
    `status == PENDING` and are filled at grade time by `compute_pnl`.
    """

    ledger_id: str = Field(description="Unique within the local DB (uuid hex)")
    trace_id: str = Field(description="FK to traces.trace_id")
    bet_date: str | None = Field(
        default=None, description="Event/decision date, YYYY-MM-DD (for dashboard slicing)"
    )
    league: str | None = None
    sport: str | None = Field(default=None, description="Derived via get_league_config()[sport]")
    matchup: str = Field(default="", description="Display label / dashboard identifier column")
    market: str = Field(description="moneyline | spread | total | player_prop:<stat>")
    bookmaker: str = Field(default="consensus", description="Sportsbook; 'consensus' if unknown")
    selection: str = Field(description="Human-readable, e.g. 'Boston Celtics -3.5'")
    selection_descriptor: str = Field(
        description="Canonical, snake_case, line embedded — e.g. 'home_spread_-3.5'"
    )
    line: float | None = Field(default=None, description="Point/total value; None for moneyline")
    odds: float = Field(description="American odds (e.g. -110, +135)")
    stake_amount: float = Field(default=DEFAULT_STAKE_AMOUNT, gt=0, description="Dollars wagered")
    payout_amount: float | None = Field(
        default=None, description="Total returned incl. stake; None while pending"
    )
    net_pnl: float | None = Field(
        default=None, description="payout - stake; None while pending"
    )
    bankroll_at_open: float | None = Field(
        default=DEFAULT_BANKROLL, description="Bankroll snapshot for future dynamic sizing"
    )
    status: LedgerStatus = LedgerStatus.PENDING
    provenance: BetProvenance = Field(description="backfill | engine_auto | user_confirmed")
    decision_timestamp: str = Field(description="ISO 8601 timestamp of the analysis/decision")
    graded_at: str | None = None
    recorded_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
