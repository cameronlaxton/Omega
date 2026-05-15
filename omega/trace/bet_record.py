"""
omega.trace.bet_record — user-confirmed wager metadata tied to a trace.

A BetRecord captures what the user ACTUALLY took: which book, which price, which
stake, when. This is distinct from the engine's recommendation (in the trace
itself) — the user may have shopped a better price, taken a partial position, or
deferred. CLV computation depends on these fields; without them, a trace can
still be graded for calibration but cannot contribute CLV signal.

One row per (trace_id, market, selection_descriptor) — a slate trace may have
multiple bets, each addressed by its descriptor.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BetStatus(str, Enum):
    """Lifecycle of a single bet."""
    PENDING = "pending"   # Recorded, outcome not yet attached
    WON = "won"
    LOST = "lost"
    VOID = "void"         # Cancelled by book (rainout, etc.)
    PUSH = "push"         # Tied to the line/total


class BetRecord(BaseModel):
    """User's actual taken bet for one selection on one trace.

    The schema enforces enough provenance for CLV resolution and outcome grading.
    `selection_descriptor` is the canonical form (snake_case, line embedded) the
    closing-line job uses for matching; `selection` is the human-readable form
    surfaced in chat.
    """

    bet_id: str = Field(description="Unique within the local DB (uuid hex)")
    trace_id: str = Field(description="FK to traces.trace_id")
    book: str = Field(description="DraftKings, FanDuel, BetMGM, Caesars, ESPN BET, Hard Rock, PointsBet, Fanatics, or other:<name>")
    market: str = Field(description="moneyline | spread | total | player_prop:<stat>")
    selection: str = Field(description="Human-readable, e.g. 'Boston Celtics -3.5'")
    selection_descriptor: str = Field(
        description="Canonical, snake_case, line embedded — e.g. 'home_spread_-3.5', 'Tatum_over_27.5_pts'"
    )
    line_taken: Optional[float] = Field(
        default=None,
        description="The point/total value at which the user bet; None for moneyline",
    )
    odds_taken: float = Field(description="American odds (e.g. -110, +135)")
    stake_units: float = Field(gt=0, description="In units (typically 1.0 = 1% bankroll)")
    decision_timestamp: str = Field(
        description="ISO 8601 timestamp of when the user confirmed the bet"
    )
    status: BetStatus = BetStatus.PENDING
    recorded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_export_block(
        cls,
        trace_id: str,
        bet_id: str,
        block: dict,
    ) -> "BetRecord":
        """Build a BetRecord from the `bet_record` sub-dict emitted by the LLM per system prompt §10.

        The export block has fields: book, market, selection, line_taken, odds_taken,
        stake_units, decision_timestamp. The `selection_descriptor` may come from
        the sibling `clv_capture_instructions` block; the caller resolves which.
        """
        return cls(
            bet_id=bet_id,
            trace_id=trace_id,
            book=block["book"],
            market=block["market"],
            selection=block["selection"],
            selection_descriptor=block.get("selection_descriptor", block["selection"]),
            line_taken=block.get("line_taken"),
            odds_taken=float(block["odds_taken"]),
            stake_units=float(block["stake_units"]),
            decision_timestamp=block["decision_timestamp"],
        )
