"""Tests for select_portfolio + the flag-gated service adapter (Stage C PR5)."""

from __future__ import annotations

import random

import pytest

from omega.core.betting.exposure import ExposureLimits, ExposurePolicy
from omega.core.betting.portfolio_selection import (
    BetCandidate,
    select_portfolio,
)
from omega.core.betting.staking_policy import FractionalKellyByTier, StakingContext


def _cand(
    selection,
    ev,
    *,
    edge=5.0,
    prob=0.62,
    odds=-110,
    tier="A",
    market="moneyline",
    keys=("game:NBA:A @ B",),
):
    return BetCandidate(
        selection=selection,
        selection_descriptor=selection.lower().replace(" ", "_"),
        market=market,
        calibrated_prob=prob,
        odds=odds,
        edge_pct=edge,
        ev_pct=ev,
        confidence_tier=tier,
        entity_keys=keys,
        league="NBA",
    )


# --- filtering -----------------------------------------------------------------
def test_empty_candidates():
    sel = select_portfolio([], bankroll=1000.0)
    assert sel.bets == ()
    assert sel.skipped == ()


def test_filters_non_actionable():
    cands = [
        _cand("a", 5.0, tier="C"),
        _cand("b", 5.0, tier="Pass"),
        _cand("c", -1.0, tier="A"),  # ev <= 0
        _cand("d", 3.0, tier="B"),  # kept
    ]
    sel = select_portfolio(cands, bankroll=1000.0)
    assert [b.candidate.selection for b in sel.bets] == ["d"]
    reasons = {c.selection: r for c, r in sel.skipped}
    assert reasons == {
        "a": "tier_or_ev_filtered",
        "b": "tier_or_ev_filtered",
        "c": "tier_or_ev_filtered",
    }


# --- ordering / determinism ----------------------------------------------------
def test_sorted_by_ev_desc():
    # budget_pct=1.0 so the slate budget doesn't truncate; we're testing order.
    cands = [_cand("lo", 2.0, keys=()), _cand("hi", 9.0, keys=()), _cand("mid", 5.0, keys=())]
    sel = select_portfolio(cands, bankroll=1000.0, budget_pct=1.0)
    assert [b.candidate.selection for b in sel.bets] == ["hi", "mid", "lo"]


def test_order_independent():
    cands = [
        _cand("a", 9.0, edge=4.0, keys=()),
        _cand("b", 9.0, edge=6.0, keys=()),  # same EV, higher edge -> ranks first
        _cand("c", 3.0, keys=()),
    ]
    base = select_portfolio(cands, bankroll=1000.0, budget_pct=1.0)
    for seed in range(5):
        shuffled = cands[:]
        random.Random(seed).shuffle(shuffled)
        assert select_portfolio(shuffled, bankroll=1000.0, budget_pct=1.0).bets == base.bets
    assert [b.candidate.selection for b in base.bets] == ["b", "a", "c"]


# --- sizing parity with the staking policy (no downsize) -----------------------
def test_full_stake_matches_policy_units():
    c = _cand("x", 5.0, prob=0.62, odds=-110, tier="A", keys=())
    sel = select_portfolio([c], bankroll=1000.0)
    decision = FractionalKellyByTier().size(
        StakingContext(true_prob=0.62, odds=-110, bankroll=1000.0, confidence_tier="A")
    )
    assert sel.bets[0].units == decision.units
    assert sel.bets[0].kelly_fraction == decision.kelly_fraction
    assert sel.bets[0].capped_by == decision.capped_by


# --- exposure interaction ------------------------------------------------------
def test_exposure_downsizes_and_records_reason():
    # Big bet (5u = $50) but the game already has $45 open -> headroom $5.
    c = _cand("big", 9.0, prob=0.95, odds=250, tier="A", keys=("game:NBA:A @ B",))
    sel = select_portfolio(
        [c], bankroll=1000.0, exposure_by_entity={"game:NBA:A @ B": 45.0}, total_open=45.0
    )
    bet = sel.bets[0]
    assert bet.stake_amount == pytest.approx(5.0)
    assert bet.units == pytest.approx(0.5)
    assert "exposure_headroom" in bet.capped_by


def test_exposure_skips_when_capped():
    c = _cand("x", 9.0, prob=0.95, odds=250, tier="A", keys=("game:NBA:A @ B",))
    sel = select_portfolio(
        [c], bankroll=1000.0, exposure_by_entity={"game:NBA:A @ B": 50.0}, total_open=50.0
    )
    assert sel.bets == ()
    assert sel.skipped[0][1] == "game:NBA:A @ B"


# --- budget / max_bets ---------------------------------------------------------
def test_budget_exhaustion_skips_remainder():
    cands = [
        _cand("first", 9.0, prob=0.95, odds=250, tier="A", keys=("game:NBA:A @ B",)),
        _cand("second", 8.0, prob=0.95, odds=250, tier="A", keys=("game:NBA:C @ D",)),
    ]
    sel = select_portfolio(cands, bankroll=1000.0, budget_pct=0.005)  # $5 budget
    assert [b.candidate.selection for b in sel.bets] == ["first"]
    assert sel.bets[0].stake_amount == pytest.approx(5.0)
    assert "budget" in sel.bets[0].capped_by
    assert "second" in {c.selection for c, _ in sel.skipped}


def test_max_bets_limits_count():
    cands = [_cand(f"g{i}", 9.0 - i, keys=(f"game:NBA:{i}",)) for i in range(4)]
    sel = select_portfolio(cands, bankroll=1000.0, max_bets=2)
    assert len(sel.bets) == 2
    assert any(r == "max_bets_reached" for _, r in sel.skipped)


def test_custom_exposure_policy_limits_respected():
    pol = ExposurePolicy(ExposureLimits(max_per_game_pct=0.01))  # $10 game cap
    c = _cand("x", 9.0, prob=0.95, odds=250, tier="A", keys=("game:NBA:A @ B",))
    sel = select_portfolio([c], bankroll=1000.0, exposure_policy=pol)
    assert sel.bets[0].stake_amount == pytest.approx(10.0)  # downsized to the tighter cap


# --- service adapter parity (flag off == on for the top bet) -------------------
def _edges():
    from omega.core.contracts.schemas import EdgeDetail

    def e(side, team, ev, edge, market="moneyline", line=None, odds=-110, prob=0.6, tier="A"):
        return EdgeDetail(
            side=side,
            team=team,
            market=market,
            line=line,
            true_prob=prob,
            calibrated_prob=prob,
            market_implied=0.5,
            edge_pct=edge,
            ev_pct=ev,
            market_odds=odds,
            confidence_tier=tier,
        )

    return [
        e("home", "Knicks", ev=6.0, edge=5.0),
        e("away", "Spurs", ev=2.0, edge=1.5, odds=120, prob=0.55, tier="B"),
        e("home", "Knicks", ev=4.0, edge=3.0, market="spread", line=-3.5),
    ]


def test_service_adapter_parity_top_bet(monkeypatch):
    from omega.core.contracts import service

    edges = _edges()
    monkeypatch.delenv("OMEGA_PORTFOLIO_SELECTION", raising=False)
    legacy = service._pick_best_bet(edges, 1000.0, league="NBA", matchup="Spurs @ Knicks")

    monkeypatch.setenv("OMEGA_PORTFOLIO_SELECTION", "1")
    portfolio = service._pick_best_bet(edges, 1000.0, league="NBA", matchup="Spurs @ Knicks")

    assert legacy == portfolio  # same BetSlip for the top pick


def test_service_flag_off_is_legacy(monkeypatch):
    from omega.core.contracts import service

    monkeypatch.delenv("OMEGA_PORTFOLIO_SELECTION", raising=False)
    assert service._portfolio_selection_enabled() is False
    slip = service._pick_best_bet(_edges(), 1000.0, league="NBA", matchup="Spurs @ Knicks")
    assert slip is not None
    assert slip.selection == "Knicks home"  # highest-EV moneyline edge
