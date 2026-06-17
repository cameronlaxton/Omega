"""CLI tests for omega.ops.settle_bets."""

from __future__ import annotations

import tempfile
from pathlib import Path

from omega.ops.settle_bets import main
from omega.trace.ledger_bet import BetProvenance, LedgerBet
from omega.trace.store import TraceStore


def _make_db(monkeypatch) -> str:
    monkeypatch.setenv("OMEGA_BET_LEDGER_AUTOLOG", "0")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = TraceStore(db_path=tmp.name)
    store.persist(
        {
            "trace_id": "g1",
            "run_id": "r",
            "timestamp": "2026-05-01T00:00:00Z",
            "prompt": "p",
            "league": "NBA",
            "matchup": "A @ B",
            "execution_mode": "native_sim",
            "kind": "game",
            "result": {"status": "success"},
        }
    )
    store.record_ledger_bet(
        LedgerBet(
            ledger_id="led1",
            trace_id="g1",
            bet_date="2026-05-01",
            league="NBA",
            sport="basketball",
            matchup="A @ B",
            market="moneyline",
            bookmaker="betmgm",
            selection="home_moneyline",
            selection_descriptor="home_moneyline",
            odds=-110,
            provenance=BetProvenance.USER_CONFIRMED,
            decision_timestamp="2026-05-01T00:00:00Z",
        )
    )
    store.attach_outcome("g1", home_score=110, away_score=104)
    store.close()
    return tmp.name


def test_settle_bets_dry_run_writes_nothing(monkeypatch):
    db = _make_db(monkeypatch)

    rc = main(["--db", db])

    assert rc == 0
    store = TraceStore(db_path=db)
    try:
        assert store.get_ledger_bets("g1")[0]["status"] == "pending"
    finally:
        store.close()


def test_settle_bets_apply_settles_pending_row(monkeypatch):
    db = _make_db(monkeypatch)

    rc = main(["--db", db, "--apply"])

    assert rc == 0
    store = TraceStore(db_path=db)
    try:
        row = store.get_ledger_bets("g1")[0]
        assert row["status"] == "won"
        assert row["net_pnl"] is not None
    finally:
        store.close()


def test_settle_bets_explicit_db_does_not_create_cwd_root_db(monkeypatch, tmp_path):
    db = _make_db(monkeypatch)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    rc = main(["--db", db])

    assert rc == 0
    assert not (cwd / "omega_traces.db").exists()
    assert not Path("omega_traces.db").exists()
