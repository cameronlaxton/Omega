"""Tests for backfill_closing_lines fixes: soccer canonicalizer + backfill provenance."""

from __future__ import annotations

import tempfile

import pytest

from omega.ops.backfill_closing_lines import _identity, _load_canonicalizer, _pending_bets_needing_close
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore


# --- Fix 1: soccer leagues resolve through espn_soccer.canonical_team ----------
@pytest.mark.parametrize("league", ["WORLD_CUP", "CHAMPIONS_LEAGUE", "MLS", "EPL", "world_cup"])
def test_load_canonicalizer_soccer_uses_espn_soccer(league):
    canon = _load_canonicalizer(league)
    # Previously these fell through to _identity (no normalization). Now they
    # normalize national-team aliases so trace names match Odds-API snapshot names.
    assert canon("USA") == "United States"
    assert canon("Ecuador") == "Ecuador"
    assert canon("Saudi Arabia") == "Saudi Arabia"


def test_load_canonicalizer_unknown_league_falls_back_to_identity():
    canon = _load_canonicalizer("NOT_A_REAL_LEAGUE")
    assert canon is _identity
    assert canon("Whatever FC") == "Whatever FC"


def test_load_canonicalizer_soccer_pair_matches_after_normalization():
    # The matching builds events_by_pair via the same canonicalizer on both the
    # snapshot side and the trace side; "USA"/"United States" must collapse.
    canon = _load_canonicalizer("WORLD_CUP")
    assert canon("USA") == canon("United States")


# --- Fix 2: backfill-provenance bets are eligible for closing-line backfill -----
@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("OMEGA_BET_LEDGER_AUTOLOG", "0")  # isolate ledger contents
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    s = TraceStore(db_path=tmp.name)
    yield s
    s.close()


def _persist_trace(store, trace_id, league="WORLD_CUP"):
    store.persist({
        "trace_id": trace_id,
        "run_id": "r",
        "timestamp": "2026-06-02T18:00:00Z",
        "prompt": "p",
        "league": league,
        "matchup": "Saudi Arabia @ Ecuador",
        "execution_mode": "native_sim",
        "kind": "game",
        "input_snapshot": {"home_team": "Ecuador", "away_team": "Saudi Arabia"},
        "result": {"status": "success"},
    })


def _ledger_bet(trace_id, provenance, *, ledger_id, status=LedgerStatus.WON):
    return LedgerBet(
        ledger_id=ledger_id,
        trace_id=trace_id,
        league="WORLD_CUP",
        market="moneyline",
        selection="Ecuador ML",
        selection_descriptor="home_moneyline",
        odds=-120,
        status=status,
        provenance=provenance,
        decision_timestamp="2026-06-02T18:00:00+00:00",
    )


def test_pending_bets_includes_backfill_provenance(store):
    _persist_trace(store, "t_bf")
    _persist_trace(store, "t_uc")
    _persist_trace(store, "t_ea")
    store.record_ledger_bet(_ledger_bet("t_bf", BetProvenance.BACKFILL, ledger_id="lb_bf"))
    store.record_ledger_bet(_ledger_bet("t_uc", BetProvenance.USER_CONFIRMED, ledger_id="lb_uc"))
    store.record_ledger_bet(_ledger_bet("t_ea", BetProvenance.ENGINE_AUTO, ledger_id="lb_ea"))

    bets = _pending_bets_needing_close(store, None, None, None, None)
    ids = {b["bet_id"] for b in bets}
    # All three provenances are now eligible (was: backfill excluded).
    assert {"lb_bf", "lb_uc", "lb_ea"} <= ids


def test_pending_bets_excludes_those_with_closing_line(store):
    _persist_trace(store, "t1")
    store.record_ledger_bet(_ledger_bet("t1", BetProvenance.BACKFILL, ledger_id="lb1"))
    # Attach a closing line for this exact (trace, market, selection_descriptor).
    store.attach_closing_line(
        trace_id="t1",
        market="moneyline",
        selection_descriptor="home_moneyline",
        closing_odds=-118,
        closing_line=None,
        closing_timestamp="2026-06-02T23:00:00+00:00",
        source="test",
    )
    bets = _pending_bets_needing_close(store, None, None, None, None)
    assert "lb1" not in {b["bet_id"] for b in bets}


def test_pending_bets_graded_status_included(store):
    # The earlier `b.status = 'pending'` restriction was lifted; a graded (won)
    # bet still needing a close must be returned.
    _persist_trace(store, "tg")
    store.record_ledger_bet(
        _ledger_bet("tg", BetProvenance.USER_CONFIRMED, ledger_id="lbg", status=LedgerStatus.WON)
    )
    bets = _pending_bets_needing_close(store, None, None, None, None)
    assert "lbg" in {b["bet_id"] for b in bets}
