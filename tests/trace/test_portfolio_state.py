"""Tests for PortfolioState / BankrollTimeline derivation (Stage C PR3)."""

from __future__ import annotations

import random

import pytest

from omega.trace.ledger_bet import DEFAULT_BANKROLL
from omega.trace.portfolio_state import (
    SETTLED_STATUSES,
    OpenPosition,
    PortfolioState,
    entity_keys_for,
)


def _row(**over):
    """A query_ledger-shaped row with sensible defaults."""
    base = {
        "ledger_id": "L1",
        "trace_id": "sandbox-1",
        "bet_date": "2026-06-01",
        "league": "NBA",
        "sport": "basketball",
        "matchup": "Spurs @ Knicks",
        "market": "prop",
        "bookmaker": "consensus",
        "selection": "Player over 24.5",
        "selection_descriptor": "player_points_over_24.5",
        "line": 24.5,
        "odds": -110.0,
        "stake_amount": 25.0,
        "payout_amount": None,
        "net_pnl": None,
        "bankroll_at_open": 1000.0,
        "status": "pending",
        "provenance": "user_confirmed",
        "decision_timestamp": "2026-06-01T18:00:00+00:00",
        "graded_at": None,
        "session_id": "s1",
        "created_at": "2026-06-01T17:00:00+00:00",
    }
    base.update(over)
    return base


# --- empty ---------------------------------------------------------------------
def test_empty_rows_is_base_bankroll():
    state = PortfolioState.from_ledger_rows([], base_bankroll=2000.0)
    assert state.bankroll == 2000.0
    assert state.open_positions == ()
    assert state.exposure_by_entity == {}
    assert state.timeline.current() == 2000.0


def test_default_base_bankroll():
    state = PortfolioState.from_ledger_rows([])
    assert state.bankroll == DEFAULT_BANKROLL


# --- bankroll / timeline -------------------------------------------------------
def test_bankroll_is_base_plus_settled_pnl():
    rows = [
        _row(ledger_id="w", status="won", net_pnl=22.73, graded_at="2026-06-01T22:00:00+00:00"),
        _row(ledger_id="l", status="lost", net_pnl=-25.0, graded_at="2026-06-02T22:00:00+00:00"),
        _row(ledger_id="p", status="pending"),  # open: ignored for bankroll
    ]
    state = PortfolioState.from_ledger_rows(rows, base_bankroll=1000.0)
    assert state.bankroll == pytest.approx(997.73)
    assert state.timeline.current() == pytest.approx(997.73)


def test_push_and_void_are_settled_not_open():
    rows = [
        _row(ledger_id="pu", status="push", net_pnl=0.0, graded_at="2026-06-01T22:00:00+00:00"),
        _row(ledger_id="vo", status="void", net_pnl=0.0, graded_at="2026-06-01T23:00:00+00:00"),
    ]
    state = PortfolioState.from_ledger_rows(rows, base_bankroll=1000.0)
    assert state.open_positions == ()
    assert state.bankroll == pytest.approx(1000.0)
    assert {"push", "void"} <= SETTLED_STATUSES


def test_timeline_points_sorted_and_cumulative():
    # Provided out of settle-time order; timeline must sort and accumulate.
    rows = [
        _row(ledger_id="b", status="won", net_pnl=10.0, graded_at="2026-06-03T00:00:00+00:00"),
        _row(ledger_id="a", status="lost", net_pnl=-5.0, graded_at="2026-06-01T00:00:00+00:00"),
        _row(ledger_id="c", status="won", net_pnl=20.0, graded_at="2026-06-05T00:00:00+00:00"),
    ]
    tl = PortfolioState.from_ledger_rows(rows, base_bankroll=100.0).timeline
    assert [ts for ts, _ in tl.points] == [
        "2026-06-01T00:00:00+00:00",
        "2026-06-03T00:00:00+00:00",
        "2026-06-05T00:00:00+00:00",
    ]
    assert [b for _, b in tl.points] == [95.0, 105.0, 125.0]


def test_bankroll_at_historical_timestamp():
    rows = [
        _row(ledger_id="a", status="lost", net_pnl=-5.0, graded_at="2026-06-01T00:00:00+00:00"),
        _row(ledger_id="b", status="won", net_pnl=10.0, graded_at="2026-06-03T00:00:00+00:00"),
    ]
    tl = PortfolioState.from_ledger_rows(rows, base_bankroll=100.0).timeline
    assert tl.bankroll_at("2026-05-31T00:00:00+00:00") == 100.0  # before any settlement
    assert tl.bankroll_at("2026-06-02T00:00:00+00:00") == 95.0  # after the loss only
    assert tl.bankroll_at("2026-06-04T00:00:00+00:00") == 105.0  # after both
    assert tl.bankroll_at(None) == 105.0  # current


# --- open positions + exposure -------------------------------------------------
def test_open_positions_and_exposure_keys():
    rows = [_row(ledger_id="o1", status="pending", stake_amount=30.0)]
    state = PortfolioState.from_ledger_rows(rows)
    assert len(state.open_positions) == 1
    pos = state.open_positions[0]
    assert isinstance(pos, OpenPosition)
    assert pos.stake_amount == 30.0
    assert pos.entity_keys == (
        "sport:BASKETBALL",
        "league:NBA",
        "game:NBA:Spurs @ Knicks",
        "selection:prop:player_points_over_24.5",
    )
    assert state.exposure_by_entity["league:NBA"] == 30.0
    assert state.exposure_by_entity["game:NBA:Spurs @ Knicks"] == 30.0


def test_exposure_sums_across_open_positions():
    rows = [
        _row(ledger_id="o1", status="pending", stake_amount=30.0),
        _row(
            ledger_id="o2",
            status="pending",
            stake_amount=20.0,
            selection_descriptor="player_assists_over_6.5",
        ),
    ]
    state = PortfolioState.from_ledger_rows(rows)
    # both share sport/league/game -> exposure accumulates there
    assert state.exposure_by_entity["league:NBA"] == 50.0
    assert state.exposure_by_entity["game:NBA:Spurs @ Knicks"] == 50.0
    # distinct selections -> separate keys
    assert state.exposure_by_entity["selection:prop:player_points_over_24.5"] == 30.0
    assert state.exposure_by_entity["selection:prop:player_assists_over_6.5"] == 20.0


def test_settled_bets_do_not_count_as_exposure():
    rows = [
        _row(ledger_id="o", status="pending", stake_amount=40.0),
        _row(
            ledger_id="s",
            status="won",
            stake_amount=99.0,
            net_pnl=90.0,
            graded_at="2026-06-01T22:00:00+00:00",
        ),
    ]
    state = PortfolioState.from_ledger_rows(rows)
    assert state.exposure_by_entity["league:NBA"] == 40.0  # only the open one


# --- entity_keys_for edge cases ------------------------------------------------
def test_entity_keys_skip_missing_fields():
    keys = entity_keys_for({"market": "game", "selection_descriptor": "home_ml"})
    assert keys == ("selection:game:home_ml",)  # no sport/league/matchup


def test_entity_keys_case_normalized():
    a = entity_keys_for(
        {
            "league": "nba",
            "sport": "Basketball",
            "matchup": "A @ B",
            "market": "game",
            "selection_descriptor": "x",
        }
    )
    b = entity_keys_for(
        {
            "league": "NBA",
            "sport": "basketball",
            "matchup": "A @ B",
            "market": "game",
            "selection_descriptor": "x",
        }
    )
    assert a == b
    assert "league:NBA" in a and "sport:BASKETBALL" in a


# --- determinism ---------------------------------------------------------------
def test_order_independent():
    rows = [
        _row(
            ledger_id="a",
            status="won",
            net_pnl=10.0,
            graded_at="2026-06-01T00:00:00+00:00",
            stake_amount=10.0,
        ),
        _row(
            ledger_id="b",
            status="lost",
            net_pnl=-7.0,
            graded_at="2026-06-02T00:00:00+00:00",
            stake_amount=20.0,
        ),
        _row(ledger_id="c", status="pending", stake_amount=15.0, selection_descriptor="other"),
    ]
    base = PortfolioState.from_ledger_rows(rows, base_bankroll=500.0)
    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        s = PortfolioState.from_ledger_rows(shuffled, base_bankroll=500.0)
        assert s.bankroll == base.bankroll
        assert s.exposure_by_entity == base.exposure_by_entity
        assert s.timeline.points == base.timeline.points
