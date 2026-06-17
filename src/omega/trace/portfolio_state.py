"""omega.trace.portfolio_state — a read-only portfolio model for bet sizing.

``summarize_ledger`` (portfolio.py) reduces ledger rows to *reporting* numbers.
This module is its richer sibling: it derives the state a portfolio-aware
**sizing** decision needs — current bankroll, the open positions still at risk,
and how much stake is exposed to each entity (sport / league / game / selection).

Like ``summarize_ledger`` it is a pure transform over the dicts
``TraceStore.query_ledger`` returns (no DB access here), so it works identically
on the SQLite and Postgres read paths and is unit-testable without a database.

It computes no protected betting output (probability, edge, EV, Kelly, units,
tiers) — those stay in the deterministic engine. This is the *input* a future
``StakingPolicy`` / portfolio selector reads to size new bets against existing
exposure; ``entity_keys_for`` is the shared key derivation so live selection and
historical state label exposure the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from omega.trace.ledger_bet import DEFAULT_BANKROLL

# Settled (money-realized) statuses. Mirrors ``portfolio._GRADED``: a bet is
# "open" (still at risk) iff its status is not one of these.
SETTLED_STATUSES = frozenset({"won", "lost", "push", "void"})


def _to_float(value: Any) -> float:
    """Coerce a possibly-None numeric DB cell to float (None -> 0.0)."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _settle_ts(row: Mapping[str, Any]) -> str:
    """Best-available settle timestamp for ordering the bankroll timeline."""
    return str(row.get("graded_at") or row.get("decision_timestamp") or row.get("created_at") or "")


def entity_keys_for(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Deterministic, namespaced exposure keys for one ledger row.

    Derived only from reliably-present ledger fields. League/sport are
    upper-cased so keys match regardless of source casing (consistent with the
    case-insensitive registry lookups). Finer-grained team/player and
    correlated-group keys are layered on in PR4 (ExposurePolicy) via
    ``parlay.correlation_group_key`` — this keeps PR3 honest about what a ledger
    row alone can attribute.
    """
    sport = (row.get("sport") or "").strip()
    league = (row.get("league") or "").strip()
    matchup = (row.get("matchup") or "").strip()
    market = (row.get("market") or "").strip()
    selection = (row.get("selection_descriptor") or "").strip()

    keys: list[str] = []
    if sport:
        keys.append(f"sport:{sport.upper()}")
    if league:
        keys.append(f"league:{league.upper()}")
    if matchup:
        scope = league.upper() or sport.upper()
        keys.append(f"game:{scope}:{matchup}" if scope else f"game:{matchup}")
    if selection:
        keys.append(f"selection:{market}:{selection}")
    return tuple(keys)


@dataclass(frozen=True)
class OpenPosition:
    """One bet still at risk (status not settled)."""

    ledger_id: str
    trace_id: str
    league: str | None
    sport: str | None
    market: str
    selection_descriptor: str
    stake_amount: float
    entity_keys: tuple[str, ...]


@dataclass(frozen=True)
class BankrollTimeline:
    """Bankroll as a function of settle time.

    ``points`` is ``(settle_ts, bankroll_after)`` for each settled bet in
    ascending settle-time order, where ``bankroll_after`` is the running
    ``base_bankroll`` + cumulative net PnL. Timestamps are compared
    lexicographically (ISO-8601 with a consistent offset sorts correctly).
    """

    base_bankroll: float
    points: tuple[tuple[str, float], ...] = ()

    def current(self) -> float:
        """Bankroll after all settled bets (== base when nothing is settled)."""
        return self.points[-1][1] if self.points else self.base_bankroll

    def bankroll_at(self, iso_ts: str | None) -> float:
        """Bankroll as of ``iso_ts`` — the last point with settle_ts <= iso_ts.

        ``None`` returns the current bankroll. A timestamp before the first
        settlement returns ``base_bankroll``.
        """
        if iso_ts is None:
            return self.current()
        result = self.base_bankroll
        for settle_ts, bankroll_after in self.points:
            if settle_ts <= iso_ts:
                result = bankroll_after
            else:
                break
        return result


@dataclass(frozen=True)
class PortfolioState:
    """Read-only snapshot a sizing decision reads against.

    ``bankroll`` is the *realized* bankroll (base + settled PnL); open stakes are
    not subtracted from it — they are tracked as exposure (``exposure_by_entity``
    and ``open_positions``) so the exposure policy, not the bankroll, governs how
    much new risk a position may add.
    """

    bankroll: float
    open_positions: tuple[OpenPosition, ...] = ()
    timeline: BankrollTimeline = field(
        default_factory=lambda: BankrollTimeline(base_bankroll=DEFAULT_BANKROLL)
    )
    exposure_by_entity: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_ledger_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        base_bankroll: float = DEFAULT_BANKROLL,
    ) -> PortfolioState:
        """Derive a portfolio state from ``query_ledger`` row dicts.

        Deterministic and order-independent: the timeline is sorted by settle
        time and per-entity exposure is a sum, so shuffling ``rows`` yields an
        identical state.
        """
        rows = list(rows)

        # Bankroll timeline from settled bets, ordered by settle time.
        settled = [
            r for r in rows if str(r.get("status") or "pending").lower() in SETTLED_STATUSES
        ]
        settled.sort(key=_settle_ts)
        points: list[tuple[str, float]] = []
        running = base_bankroll
        for r in settled:
            running += _to_float(r.get("net_pnl"))
            points.append((_settle_ts(r), round(running, 2)))
        timeline = BankrollTimeline(base_bankroll=round(base_bankroll, 2), points=tuple(points))

        # Open positions + per-entity exposure.
        open_positions: list[OpenPosition] = []
        exposure: dict[str, float] = {}
        for r in rows:
            if str(r.get("status") or "pending").lower() in SETTLED_STATUSES:
                continue
            stake = round(_to_float(r.get("stake_amount")), 2)
            keys = entity_keys_for(r)
            open_positions.append(
                OpenPosition(
                    ledger_id=str(r.get("ledger_id") or ""),
                    trace_id=str(r.get("trace_id") or ""),
                    league=r.get("league"),
                    sport=r.get("sport"),
                    market=str(r.get("market") or ""),
                    selection_descriptor=str(r.get("selection_descriptor") or ""),
                    stake_amount=stake,
                    entity_keys=keys,
                )
            )
            for key in keys:
                exposure[key] = round(exposure.get(key, 0.0) + stake, 2)

        return cls(
            bankroll=timeline.current(),
            open_positions=tuple(open_positions),
            timeline=timeline,
            exposure_by_entity=exposure,
        )
