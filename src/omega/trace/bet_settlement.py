"""
omega.trace.bet_settlement — pure helpers for the dollar bet ledger.

No DB access here: every function is a deterministic transform, so the same code
serves both `TraceStore.persist()`'s gated dual-write and `backfill_bets.py`.

Responsibilities:
- extract_recommended_bet: turn a trace's engine recommendation into a LedgerBet
  (the single recommended selection — game best edge or prop over/under).
- settle_game_bet / settle_prop_bet: grade a logged bet against an attached
  outcome into a LedgerStatus.
- compute_pnl: dollar payout + net PnL from status, American odds, and stake.

Legacy-odds safety: American odds are always >= +100 or <= -100. Anything in the
open interval (-100, +100) — e.g. a decimal price like 1.91 mislabeled as
American — is rejected (returns None / skip_bad_odds) rather than mis-settled.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from omega.core.betting.odds import american_to_decimal
from omega.core.config.leagues import get_league_config
from omega.trace.ledger_bet import (
    DEFAULT_BANKROLL,
    DEFAULT_STAKE_AMOUNT,
    BetProvenance,
    LedgerBet,
    LedgerStatus,
)

# Reasons returned by extract_recommended_bet so callers can tally skips.
REASON_OK = "ok"
REASON_SKIP_PASS = "skip_pass"  # engine recommended pass / no actionable edge
REASON_SKIP_NO_EDGE = "skip_no_edge"  # no edges and no best_bet to log
REASON_SKIP_BAD_ODDS = "skip_bad_odds"  # odds missing or not valid American
REASON_SKIP_SLATE = "skip_slate"  # slate trace — out of scope for one-bet-per-trace
REASON_SKIP_UNKNOWN_KIND = "skip_unknown_kind"  # cannot classify game vs prop


@dataclass
class ExtractResult:
    """Outcome of extracting a recommended bet from a trace."""

    bet: LedgerBet | None
    reason: str


def coerce_american_odds(value: object) -> float | None:
    """Return a usable American-odds float, or None if invalid/ambiguous.

    Rejects 0, non-numerics, and any value strictly inside (-100, +100) — that
    band is impossible for real American odds and almost always a decimal price
    that slipped through. Skipping is safer than fabricating PnL from it.
    """
    if value is None:
        return None
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if -100.0 < odds < 100.0:
        return None
    return odds


def _norm_market(raw: str | None) -> str:
    """Collapse edge market labels to canonical moneyline | spread | total."""
    m = (raw or "moneyline").strip().lower()
    if "spread" in m:
        return "spread"
    if "total" in m:
        return "total"
    if m in ("moneyline", "home", "away", "draw", "ml"):
        return "moneyline"
    return m


def _slug(text: object) -> str:
    """snake_case a value for use inside a selection_descriptor."""
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9.+-]+", "_", s)
    return s.strip("_")


def _fmt_line(line: float | None) -> str:
    """Render a line for a descriptor: drop trailing .0, keep sign."""
    if line is None:
        return ""
    if float(line).is_integer():
        return str(int(line))
    return str(line)


def build_selection_descriptor(
    market: str,
    side: str,
    *,
    line: float | None = None,
    player: str | None = None,
    stat: str | None = None,
    team: str | None = None,
) -> tuple[str, str]:
    """Build the canonical ``(selection_descriptor, human_label)`` for a bet.

    The descriptor is the snake_case idempotency key stored alongside
    ``(trace_id, market)``; its first token is the gradeable side so settlement
    can recover it (see ``settle_game_bet`` / ``backfill_bets._grade_fields``).
    This is the single source of truth shared by ``extract_recommended_bet``
    (engine/backfill auto-logging) and the ``omega_record_flat_bet`` MCP tool,
    so the two never drift.

    Forms::

        moneyline      -> "{side}_moneyline"
        spread         -> "{side}_spread_{line}"
        total          -> "{side}_total_{line}"
        player_prop:*  -> "{player}_{side}_{line}_{stat}"
    """
    s = (side or "").strip().lower()
    line_str = _fmt_line(line)

    if market.startswith("player_prop"):
        stat_name = stat or (market.split(":", 1)[1] if ":" in market else "")
        descriptor = "_".join(
            p for p in (_slug(player), s, line_str, _slug(stat_name)) if p
        )
        label = f"{player or ''} {s.title()} {line_str} {stat_name}".strip()
        return descriptor, label

    m = _norm_market(market)
    label_team = team or s
    if m == "spread":
        return f"{s}_spread_{line_str}", f"{label_team} {line_str}".strip()
    if m == "total":
        return f"{s}_total_{line_str}", f"{s.title()} {line_str}".strip()
    if m == "moneyline":
        return f"{s}_moneyline", f"{label_team} ML".strip()
    # Exotic / unrecognized market — keep side + slugged market in the descriptor.
    descriptor = "_".join(p for p in (s, _slug(m), line_str) if p)
    return descriptor, f"{label_team} {m} {line_str}".strip()


def _trace_kind(trace: dict) -> str:
    """Best-effort classification of a trace as game | prop | slate | unknown."""
    kind = str(trace.get("kind") or "").lower()
    if kind in ("game", "prop", "slate"):
        return kind
    result = trace.get("result") or {}
    if "recommendation" in result or "over_prob" in result or "under_prob" in result:
        return "prop"
    if result.get("edges") or result.get("best_bet") or "analyses" in result:
        return "slate" if "analyses" in result else "game"
    return "unknown"


def _common_fields(trace: dict) -> tuple[str | None, str | None, str, str]:
    """Return (league, sport, matchup, decision_timestamp) for a trace."""
    input_snap = trace.get("input_snapshot") or {}
    result = trace.get("result") or {}
    league = trace.get("league") or input_snap.get("league") or result.get("league")
    sport = None
    if league:
        sport = get_league_config(str(league)).get("sport")
    matchup = trace.get("matchup") or result.get("matchup") or ""
    ts = str(trace.get("timestamp") or result.get("analyzed_at") or "")
    return league, sport, matchup, ts


def _date_of(timestamp: str) -> str | None:
    """Extract YYYY-MM-DD from an ISO timestamp, if present."""
    if not timestamp:
        return None
    return timestamp[:10] if len(timestamp) >= 10 else None


CONSENSUS_BOOK = "consensus"


def _nrm(s: object) -> str:
    return str(s or "").strip().casefold()


def _resolve_prop_book(trace: dict) -> str:
    """Book recorded on a prop trace's request (provenance only), else consensus."""
    input_snap = trace.get("input_snapshot") or {}
    result = trace.get("result") or {}
    book = input_snap.get("bookmaker") or result.get("bookmaker")
    return str(book) if book else CONSENSUS_BOOK


def _resolve_game_book(
    trace: dict, *, market: str, side: str, team: str | None, line: float | None
) -> str:
    """Recover the sportsbook behind a game selection from input_snapshot.odds.

    Matches the chosen market/selection/line against the persisted
    ``OddsInput.markets`` quotes (each carries its source bookmaker). Falls back
    to the single book if every quote shares one, else 'consensus' (unknown).
    """
    odds = (trace.get("input_snapshot") or {}).get("odds") or {}
    markets = odds.get("markets") or []
    if not isinstance(markets, list):
        return CONSENSUS_BOOK
    labels = {_nrm(side)} if market == "total" else {_nrm(team), _nrm(side)}
    labels.discard("")
    for q in markets:
        if not isinstance(q, dict) or str(q.get("market_type") or "") != market:
            continue
        sel = _nrm(q.get("selection"))
        # Match a real quote for this market. The reverse-prefix form is only
        # safe when the recommended bet has a line and the quote carries the
        # same line; otherwise a spread label like "B -3.5" can incorrectly
        # attach to a plain moneyline quote for "B".
        if not (
            sel in labels
            or any(label and sel.startswith(f"{label} ") for label in labels)
            or (
                line is not None
                and q.get("line") is not None
                and any(label and sel and label.startswith(f"{sel} ") for label in labels)
            )
        ):
            continue
        if line is not None and q.get("line") is not None:
            try:
                if float(q["line"]) != float(line):
                    continue
            except (TypeError, ValueError):
                pass
        if q.get("bookmaker"):
            return str(q["bookmaker"])
    books = {
        str(q["bookmaker"])
        for q in markets
        if isinstance(q, dict) and q.get("bookmaker")
    }
    return books.pop() if len(books) == 1 else CONSENSUS_BOOK


def _best_game_edge(result: dict) -> dict | None:
    """Pick the highest-EV edge whose confidence tier is actionable (not Pass)."""
    edges = result.get("edges") or []
    actionable = [
        e
        for e in edges
        if isinstance(e, dict) and str(e.get("confidence_tier") or "").lower() != "pass"
    ]
    if not actionable:
        return None
    return max(actionable, key=lambda e: e.get("ev_pct") or float("-inf"))


def extract_recommended_bet(
    trace: dict,
    *,
    provenance: BetProvenance,
    stake_amount: float = DEFAULT_STAKE_AMOUNT,
    bankroll: float | None = DEFAULT_BANKROLL,
    ledger_id: str | None = None,
) -> ExtractResult:
    """Build a LedgerBet from a trace's single recommended selection.

    Returns ExtractResult(bet=None, reason=...) when there is nothing actionable
    to log (pass, no edge, unparseable odds, or a slate/unknown trace).
    """
    kind = _trace_kind(trace)
    if kind == "slate":
        return ExtractResult(None, REASON_SKIP_SLATE)
    if kind == "unknown":
        return ExtractResult(None, REASON_SKIP_UNKNOWN_KIND)

    result = trace.get("result") or {}
    input_snap = trace.get("input_snapshot") or {}
    league, sport, matchup, ts = _common_fields(trace)
    bankroll_val = bankroll if bankroll is not None else DEFAULT_BANKROLL

    def _mk(market, selection, descriptor, line, odds, bookmaker) -> ExtractResult:
        american = coerce_american_odds(odds)
        if american is None:
            return ExtractResult(None, REASON_SKIP_BAD_ODDS)
        bet = LedgerBet(
            ledger_id=ledger_id or uuid.uuid4().hex[:12],
            trace_id=str(trace.get("trace_id") or ""),
            bet_date=_date_of(ts),
            league=league,
            sport=sport,
            matchup=matchup,
            market=market,
            bookmaker=bookmaker,
            selection=selection,
            selection_descriptor=descriptor,
            line=line,
            odds=american,
            stake_amount=stake_amount,
            bankroll_at_open=bankroll_val,
            status=LedgerStatus.PENDING,
            provenance=provenance,
            decision_timestamp=ts,
        )
        return ExtractResult(bet, REASON_OK)

    if kind == "prop":
        rec = str(result.get("recommendation") or "").strip().lower()
        if rec not in ("over", "under"):
            return ExtractResult(None, REASON_SKIP_PASS)
        player = input_snap.get("player_name") or result.get("player_name") or ""
        stat = input_snap.get("prop_type") or result.get("prop_type") or "prop"
        line = input_snap.get("line")
        if line is None:
            line = result.get("line")
        odds = result.get("bet_side_odds")
        if odds is None:
            odds = input_snap.get("odds_over") if rec == "over" else input_snap.get("odds_under")
        market = f"player_prop:{_slug(stat)}"
        descriptor, label = build_selection_descriptor(
            market, rec, line=line, player=player, stat=stat
        )
        return _mk(market, label, descriptor, line, odds, _resolve_prop_book(trace))

    # kind == "game"
    edge = _best_game_edge(result)
    best_bet = result.get("best_bet")
    if edge is None:
        # No structured actionable edge. best_bet alone lacks side/line for safe
        # settlement, so we only log it as a moneyline-style flat bet if its tier
        # is actionable and its odds parse.
        if isinstance(best_bet, dict) and str(best_bet.get("confidence_tier") or "").lower() != "pass":
            sel = str(best_bet.get("selection") or "").strip()
            if not sel:
                return ExtractResult(None, REASON_SKIP_NO_EDGE)
            descriptor = _slug(sel) or "best_bet"
            book = _resolve_game_book(trace, market="moneyline", side="", team=sel, line=None)
            return _mk("moneyline", sel, descriptor, None, best_bet.get("odds"), book)
        return ExtractResult(None, REASON_SKIP_PASS)

    market = _norm_market(edge.get("market"))
    side = str(edge.get("side") or "").strip().lower()
    line = edge.get("line")
    odds = edge.get("market_odds")
    team = edge.get("team") or side

    descriptor, label = build_selection_descriptor(market, side, line=line, team=team)

    # Prefer the human-readable best_bet selection when it lines up.
    if isinstance(best_bet, dict) and best_bet.get("selection"):
        label = str(best_bet["selection"])
    book = _resolve_game_book(trace, market=market, side=side, team=team, line=line)
    return _mk(market, label, descriptor, line, odds, book)


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def settle_game_bet(
    market: str,
    side: str,
    line: float | None,
    home_score: int,
    away_score: int,
) -> LedgerStatus:
    """Grade a game bet against a final score.

    Conventions:
    - moneyline: `side` in {home, away, draw}; a tie on a home/away selection is
      treated as PUSH (safe default; sports without a draw market never tie).
    - total: `side` in {over, under}; compares home+away to `line`.
    - spread: `line` is the signed handicap APPLIED TO `side` (e.g. -3.5 for a
      3.5-point favorite). The side covers when side_margin + line > 0.
    """
    m = _norm_market(market)
    s = (side or "").strip().lower()

    if m == "total":
        if line is None:
            return LedgerStatus.PENDING
        total = home_score + away_score
        if total == line:
            return LedgerStatus.PUSH
        over_wins = total > line
        if s == "over":
            return LedgerStatus.WON if over_wins else LedgerStatus.LOST
        return LedgerStatus.LOST if over_wins else LedgerStatus.WON

    if m == "spread":
        if line is None:
            return LedgerStatus.PENDING
        if s == "home":
            side_margin = home_score - away_score
        elif s == "away":
            side_margin = away_score - home_score
        else:
            return LedgerStatus.PENDING
        cover = side_margin + line
        if cover == 0:
            return LedgerStatus.PUSH
        return LedgerStatus.WON if cover > 0 else LedgerStatus.LOST

    # moneyline
    if home_score > away_score:
        winner = "home"
    elif away_score > home_score:
        winner = "away"
    else:
        winner = "draw"
    if s == "draw":
        return LedgerStatus.WON if winner == "draw" else LedgerStatus.LOST
    if winner == "draw":
        return LedgerStatus.PUSH
    return LedgerStatus.WON if s == winner else LedgerStatus.LOST


def settle_prop_bet(
    recommended_side: str,
    outcome_result: str,
    outcome_side: str,
) -> LedgerStatus:
    """Map a prop_outcomes row (result relative to outcome_side) to a status.

    prop_outcomes.result is win|loss|push for `outcome_side`. If the recommended
    side matches the graded side, map directly; if it's the opposite side, invert
    win/loss (push stays push).
    """
    res = (outcome_result or "").strip().lower()
    if res == "push":
        return LedgerStatus.PUSH
    rec = (recommended_side or "").strip().lower()
    graded = (outcome_side or "").strip().lower()
    same_side = rec == graded
    won = (res == "win") if same_side else (res == "loss")
    return LedgerStatus.WON if won else LedgerStatus.LOST


def compute_pnl(
    status: LedgerStatus,
    odds: float,
    stake: float,
) -> tuple[float | None, float | None]:
    """Return (payout_amount, net_pnl) in dollars for a graded bet.

    - WON:  payout = stake * decimal;  net = stake * (decimal - 1)
    - LOST: payout = 0;                net = -stake
    - PUSH/VOID: payout = stake;       net = 0
    - PENDING:  (None, None)
    """
    if status == LedgerStatus.WON:
        decimal = american_to_decimal(odds)
        payout = round(stake * decimal, 2)
        return payout, round(payout - stake, 2)
    if status == LedgerStatus.LOST:
        return 0.0, round(-stake, 2)
    if status in (LedgerStatus.PUSH, LedgerStatus.VOID):
        return round(stake, 2), 0.0
    return None, None
