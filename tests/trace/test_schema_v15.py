"""Schema V15: bet_ledger sizing-audit columns (Stage C PR6)."""

from __future__ import annotations

import json
import sqlite3
import tempfile

import pytest

from omega.trace.ledger_bet import BetProvenance, LedgerBet
from omega.trace.schema import CURRENT_VERSION, V15_COLUMNS, apply_v15_migration
from omega.trace.store import TraceStore

_V15_NAMES = {name for name, _ in V15_COLUMNS}


def _tmp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def _bet_ledger_columns(conn):
    return {row[1] for row in conn.execute("PRAGMA table_info(bet_ledger)").fetchall()}


def test_current_version_is_seventeen():
    assert CURRENT_VERSION == 17


def test_fresh_db_has_v15_columns_and_version():
    store = TraceStore(db_path=_tmp_db())
    try:
        assert store.schema_version() == 17
        assert _V15_NAMES <= _bet_ledger_columns(store.conn)
    finally:
        store.close()


def test_apply_v15_is_idempotent_on_fresh_db():
    store = TraceStore(db_path=_tmp_db())
    try:
        before = _bet_ledger_columns(store.conn)
        apply_v15_migration(store.conn)  # re-run: must be a no-op, no error
        apply_v15_migration(store.conn)
        assert _bet_ledger_columns(store.conn) == before
    finally:
        store.close()


def test_apply_v15_upgrades_a_pre_v15_table():
    # Simulate a legacy bet_ledger that predates the sizing-audit columns.
    conn = sqlite3.connect(_tmp_db())
    conn.execute(
        "CREATE TABLE bet_ledger ("
        " ledger_id TEXT PRIMARY KEY, trace_id TEXT, market TEXT,"
        " selection_descriptor TEXT, odds REAL)"
    )
    conn.commit()
    assert not (_V15_NAMES & {r[1] for r in conn.execute("PRAGMA table_info(bet_ledger)")})

    apply_v15_migration(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(bet_ledger)")}
    assert _V15_NAMES <= cols
    # idempotent second run
    apply_v15_migration(conn)
    assert {r[1] for r in conn.execute("PRAGMA table_info(bet_ledger)")} == cols
    conn.close()


def test_sizing_audit_round_trips():
    store = TraceStore(db_path=_tmp_db())
    try:
        store.persist({
            "trace_id": "t1",
            "run_id": "r",
            "timestamp": "2026-06-01T00:00:00Z",
            "prompt": "p",
            "league": "NBA",
            "matchup": "A @ B",
            "execution_mode": "native_sim",
            "kind": "game",
            "result": {"status": "success"},
        })
        bet = LedgerBet(
            ledger_id="lb1",
            trace_id="t1",
            market="moneyline",
            selection="B ML",
            selection_descriptor="home_moneyline",
            odds=-110,
            provenance=BetProvenance.ENGINE_AUTO,
            decision_timestamp="2026-06-01T00:00:00+00:00",
            staking_policy_id="fractional_kelly_by_tier",
            staking_policy_version=1,
            exposure_limits_version=1,
            sizing_reasons=["unit_cap", "exposure_headroom"],
            correlation_group="corr:player:lebron james",
        )
        store.record_ledger_bet(bet)

        row = store.get_ledger_bets("t1")[0]
        assert row["staking_policy_id"] == "fractional_kelly_by_tier"
        assert row["staking_policy_version"] == 1
        assert row["exposure_limits_version"] == 1
        assert json.loads(row["sizing_reasons"]) == ["unit_cap", "exposure_headroom"]
        assert row["correlation_group"] == "corr:player:lebron james"
    finally:
        store.close()


def test_audit_columns_default_null_when_absent():
    store = TraceStore(db_path=_tmp_db())
    try:
        store.persist({
            "trace_id": "t2", "run_id": "r", "timestamp": "2026-06-01T00:00:00Z",
            "prompt": "p", "league": "NBA", "matchup": "A @ B",
            "execution_mode": "native_sim", "kind": "game", "result": {"status": "success"},
        })
        store.record_ledger_bet(LedgerBet(
            ledger_id="lb2", trace_id="t2", market="moneyline", selection="B",
            selection_descriptor="home_moneyline", odds=-110,
            provenance=BetProvenance.BACKFILL, decision_timestamp="2026-06-01T00:00:00+00:00",
        ))
        row = store.get_ledger_bets("t2")[0]
        assert row["staking_policy_id"] is None
        assert row["sizing_reasons"] is None
        assert row["correlation_group"] is None
    finally:
        store.close()
