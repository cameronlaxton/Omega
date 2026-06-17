"""Staking for historical replay.

Reuses the engine's *single* selection+sizing path: ``extract_recommended_bet``
turns a replayed trace's edges (already sized by the production staking policy
during ``analyze()``) into a ``LedgerBet``. Replay stamps it with
``provenance=historical_replay`` and decision-time odds — closing odds never
enter selection. The resulting :class:`ReplayCandidateSelection` is the audit
bridge tying the trace, the applied calibration profile, the decision odds, the
staking decision, and the ledger row together.
"""

from __future__ import annotations

from dataclasses import dataclass

from omega.historical.contracts import ReplayCandidateSelection
from omega.trace.bet_settlement import extract_recommended_bet
from omega.trace.ledger_bet import (
    DEFAULT_BANKROLL,
    DEFAULT_STAKE_AMOUNT,
    BetProvenance,
    LedgerBet,
)


@dataclass
class StakingResult:
    bet: LedgerBet | None
    selection: ReplayCandidateSelection | None
    reason: str


def _norm_market(market: str) -> str:
    m = (market or "").lower()
    if m.startswith("player_prop"):
        return "player_prop"
    if "spread" in m:
        return "spread"
    if "total" in m:
        return "total"
    return "moneyline"


def _match_edge(trace: dict, bet: LedgerBet) -> dict | None:
    """Find the result edge that produced ``bet`` (by market + side)."""
    result = trace.get("result") or {}
    edges = result.get("edges") or []
    side = bet.selection_descriptor.split("_", 1)[0]
    want = _norm_market(bet.market)
    for e in edges:
        if not isinstance(e, dict):
            continue
        if _norm_market(str(e.get("market"))) != want:
            continue
        if str(e.get("side") or "").lower() == side:
            return e
    return None


def _fallback_raw_prob(trace: dict) -> float:
    """Home win prob from the simulation block, normalized to [0, 1]."""
    sim = (trace.get("result") or {}).get("simulation") or trace.get("predictions") or {}
    p = sim.get("home_win_prob")
    if p is None:
        return 0.0
    return p / 100.0 if p > 1 else float(p)


def size_historical_bet(
    trace: dict,
    *,
    replay_id: str,
    event_id: str,
    decision_time: str,
    stake_amount: float = DEFAULT_STAKE_AMOUNT,
    bankroll: float = DEFAULT_BANKROLL,
) -> StakingResult:
    """Size one simulated historical bet from a replayed trace.

    Returns ``StakingResult(bet=None, selection=None, reason=...)`` when the
    engine recommended no actionable bet.
    """
    extract = extract_recommended_bet(
        trace,
        provenance=BetProvenance.HISTORICAL_REPLAY,
        stake_amount=stake_amount,
        bankroll=bankroll,
    )
    if extract.bet is None:
        return StakingResult(None, None, extract.reason)

    bet = extract.bet
    edge = _match_edge(trace, bet)
    audit = (edge or {}).get("calibration_audit") or {}

    selection = ReplayCandidateSelection(
        replay_id=replay_id,
        event_id=event_id,
        trace_id=bet.trace_id,
        market=bet.market,
        selection_descriptor=bet.selection_descriptor,
        raw_prob=(edge or {}).get("true_prob", _fallback_raw_prob(trace)),
        calibrated_prob=(edge or {}).get("calibrated_prob"),
        profile_id=audit.get("profile_id") or "none",
        profile_hash="",  # replay-time prod profile; fold-profile hashes live in walk-forward
        decision_odds=bet.odds,
        decision_line=bet.line,
        book=bet.bookmaker,
        decision_time=decision_time,
        edge=(edge or {}).get("edge_pct"),
        ev=(edge or {}).get("ev_pct"),
        units=(edge or {}).get("recommended_units"),
        kelly_fraction=None,
        stake_amount=bet.stake_amount,
        capped_by=[],
        ledger_id=bet.ledger_id,
        clv=None,
    )
    return StakingResult(bet, selection, extract.reason)
