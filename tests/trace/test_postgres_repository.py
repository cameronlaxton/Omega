from __future__ import annotations

import importlib
import os
import re
import threading

import pytest

pytest.importorskip("sqlalchemy")

from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.repository import PostgresRepository
from omega.trace.store import TraceStore

DISPATCHED_METHODS = {
    "persist",
    "get_simulation_distributions",
    "get_evidence_signals",
    "write_qa_verdict",
    "get_qa_verdict",
    "upsert_signal_performance",
    "get_signal_performance",
    "attach_outcome",
    "get_outcome",
    "attach_prop_outcome",
    "get_prop_outcomes",
    "record_bet",
    "get_bet_records",
    "query_ungraded_prop_bet_traces",
    "update_bet_status",
    "record_ledger_bet",
    "grade_ledger_bet",
    "get_ledger_bets",
    "query_ledger",
    "attach_closing_line",
    "get_closing_lines",
    "record_early_market_snapshot",
    "get_early_market_snapshots",
    "record_market_snapshot",
    "get_market_snapshots",
    "compute_market_movement",
    "get_trace",
    "query_traces",
    "get_graded_traces",
    "query_by_session",
    "get_session_summary",
    "schema_version",
    "count",
    "close",
}


def _make_trace(trace_id: str = "pg-trace-1") -> dict:
    return {
        "trace_id": trace_id,
        "run_id": "run-1",
        "timestamp": "2026-06-01T12:00:00Z",
        "prompt": "postgres parity",
        "league": "NBA",
        "matchup": "Pacers @ Celtics",
        "execution_mode": "native_sim",
        "simulation_seed": 99,
        "aggregate_quality": 0.9,
        "predictions": {"home_win_prob": 0.58},
        "recommendations": [],
        "odds_snapshot": {"moneyline_home": -120},
        "downgrades": [],
        "kind": "game",
        "input_snapshot": {
            "evidence": [
                {
                    "signal_type": "pace_up",
                    "category": "team_form",
                    "plane": "game",
                    "source": "test",
                    "confidence": 0.8,
                    "window": "last_5",
                    "direction": "over",
                    "value": True,
                }
            ]
        },
        "simulation_distributions": [
            {
                "kind": "game",
                "league": "NBA",
                "target": "home_score",
                "market": "score",
                "stat_key": "points",
                "distribution_type": "normal",
                "distribution_params": {"mu": 112.0, "sigma": 11.0},
                "sample_mean": 112.0,
                "sample_std": 11.0,
            }
        ],
        "trace_quality": {
            "calibration_eligible": True,
            "context_source": "provided",
            "identity_status": "complete",
        },
    }


def test_repository_public_method_parity_inventory():
    missing = sorted(name for name in DISPATCHED_METHODS if not hasattr(PostgresRepository, name))
    assert missing == []


@pytest.fixture
def postgres_url(monkeypatch):
    url = os.environ.get("OMEGA_TEST_DATABASE_URL")
    if not url:
        pytest.skip("OMEGA_TEST_DATABASE_URL not set")
    # The reset below DROPs and recreates the public schema. Refuse to do that
    # unless the operator explicitly confirms the target is disposable, so a
    # mistargeted OMEGA_TEST_DATABASE_URL cannot nuke a real database.
    if os.environ.get("OMEGA_TEST_DB_ALLOW_DESTROY") != "1":
        pytest.skip(
            "OMEGA_TEST_DATABASE_URL is set but OMEGA_TEST_DB_ALLOW_DESTROY=1 is not; "
            "the schema-reset fixture refuses to run against a non-confirmed database"
        )
    pytest.importorskip("alembic")

    from alembic.config import Config
    from sqlalchemy import text

    from alembic import command
    from omega.trace.db import create_postgres_engine

    monkeypatch.setenv("DATABASE_URL", url)
    engine = create_postgres_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    return url


def _type_category(sqltype: object) -> str:
    """Collapse dialect-specific types into comparable categories.

    SQLite reflects Float as REAL while Postgres reflects DOUBLE PRECISION, etc.,
    so raw type strings never compare equal across dialects.
    """
    name = str(sqltype).upper()
    if "INT" in name:
        return "int"
    if any(tok in name for tok in ("REAL", "DOUBLE", "FLOAT", "NUMERIC", "DECIMAL")):
        return "float"
    if any(tok in name for tok in ("CHAR", "TEXT", "CLOB", "STRING")):
        return "text"
    return "other"


def _semantic_shape(engine) -> dict:
    """Normalized, dialect-agnostic schema shape.

    Compares *semantic* structure — column name/nullability/type-category, the
    set of column-tuples backing unique + foreign-key constraints, view names,
    and non-unique index coverage — never dialect-specific names or default SQL.
    """
    from sqlalchemy import inspect

    insp = inspect(engine)
    tables = {t for t in insp.get_table_names() if t != "alembic_version"}
    shape: dict = {
        "tables": tables,
        "views": set(insp.get_view_names()),
        "columns": {},
        "unique": {},
        "fks": {},
        "indexes": {},
    }
    for table in tables:
        pk_cols = set(insp.get_pk_constraint(table).get("constrained_columns", []))
        shape["columns"][table] = {
            col["name"]: (
                False if col["name"] in pk_cols else bool(col["nullable"]),
                _type_category(col["type"]),
            )
            for col in insp.get_columns(table)
        }
        # Unique coverage = explicit unique constraints + unique indexes, by the
        # *set of columns* they cover (names/representation ignored). This makes a
        # SQLite unique index equivalent to a Postgres UNIQUE constraint.
        unique = {frozenset(uc["column_names"]) for uc in insp.get_unique_constraints(table)}
        nonunique = set()
        for idx in insp.get_indexes(table):
            cols = frozenset(c for c in idx["column_names"] if c is not None)
            if not cols:
                continue
            (unique if idx.get("unique") else nonunique).add(cols)
        shape["unique"][table] = unique
        shape["indexes"][table] = nonunique
        shape["fks"][table] = {
            (
                frozenset(fk["constrained_columns"]),
                fk["referred_table"],
                frozenset(fk["referred_columns"]),
            )
            for fk in insp.get_foreign_keys(table)
        }
    return shape


def test_postgres_schema_parity_with_sqlite_v14(postgres_url, tmp_path, monkeypatch):
    from sqlalchemy import create_engine

    with monkeypatch.context() as m:
        m.delenv("DATABASE_URL", raising=False)
        sqlite_path = tmp_path / "omega.db"
        sqlite_store = TraceStore(db_path=str(sqlite_path))  # creates V14 schema
        sqlite_store.close()
    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
    sqlite_shape = _semantic_shape(sqlite_engine)
    sqlite_engine.dispose()

    store = TraceStore()
    pg_shape = _semantic_shape(store._repo.engine)

    assert pg_shape["tables"] == sqlite_shape["tables"]
    assert pg_shape["views"] == sqlite_shape["views"]
    assert pg_shape["columns"] == sqlite_shape["columns"]
    assert pg_shape["unique"] == sqlite_shape["unique"]
    assert pg_shape["fks"] == sqlite_shape["fks"]
    # Non-unique index coverage: every SQLite index must exist on Postgres.
    for table, cols in sqlite_shape["indexes"].items():
        assert cols <= pg_shape["indexes"].get(table, set()), table
    assert store.schema_version() == 14
    store.close()


def test_postgres_repository_core_parity(postgres_url):
    store = TraceStore()
    store.persist(_make_trace())

    assert store.count() == 1
    assert store.get_trace("pg-trace-1")["trace_id"] == "pg-trace-1"
    assert len(store.get_evidence_signals("pg-trace-1")) == 1
    assert store.get_simulation_distributions("pg-trace-1")[0]["distribution_params"] == {
        "mu": 112.0,
        "sigma": 11.0,
    }

    ledger = LedgerBet(
        ledger_id="engine-row",
        trace_id="pg-trace-1",
        bet_date="2026-06-01",
        league="NBA",
        sport="basketball",
        matchup="Pacers @ Celtics",
        market="moneyline",
        bookmaker="consensus",
        selection="home ML",
        selection_descriptor="home_moneyline",
        odds=-110,
        stake_amount=25.0,
        bankroll_at_open=1000.0,
        provenance=BetProvenance.ENGINE_AUTO,
        decision_timestamp="2026-06-01T12:00:00Z",
    )
    assert store.record_ledger_bet(ledger) == "engine-row"
    upgraded = ledger.model_copy(
        update={
            "ledger_id": "user-row",
            "bookmaker": "betmgm",
            "odds": -105,
            "provenance": BetProvenance.USER_CONFIRMED,
        }
    )
    assert store.record_ledger_bet(upgraded) == "engine-row"
    row = store.get_ledger_bets("pg-trace-1")[0]
    assert row["provenance"] == "user_confirmed"
    assert row["bookmaker"] == "betmgm"

    # Timestamp columns must match SQLite's datetime('now') text format exactly
    # ("YYYY-MM-DD HH:MM:SS"), not ISO-with-T or a fractional/offset variant.
    sqlite_now = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    assert sqlite_now.match(row["created_at"]), row["created_at"]

    store.attach_outcome("pg-trace-1", home_score=110, away_score=104)
    assert sqlite_now.match(store.get_outcome("pg-trace-1")["attached_at"])
    store.grade_ledger_bet("engine-row", LedgerStatus.WON, 48.81, 23.81)
    assert store.query_ledger(status="won")[0]["net_pnl"] == 23.81
    assert len(store.get_graded_traces(league="NBA")) == 1
    store.close()


def test_postgres_query_traces_batched_attachments(postgres_url):
    """query_traces must attach prop outcomes + distributions (batched) and
    match the per-trace getters exactly, attaching keys only when non-empty."""
    store = TraceStore()
    store.persist(_make_trace("pg-batch-1"))
    store.persist(_make_trace("pg-batch-2"))
    store.attach_prop_outcome(
        "pg-batch-1",
        player_name="Tyrese Haliburton",
        stat_type="points",
        stat_value=20.0,
        line=18.5,
        side="over",
    )

    traces = {t["trace_id"]: t for t in store.query_traces(league="NBA", limit=10)}

    # Trace with a prop outcome carries the batched attachment matching the getter.
    assert traces["pg-batch-1"]["_prop_outcomes"] == store.get_prop_outcomes("pg-batch-1")
    # Trace without prop outcomes must NOT carry an empty key (parity w/ SQLite).
    assert "_prop_outcomes" not in traces["pg-batch-2"]
    # Distributions were persisted on both and must be attached for both.
    assert traces["pg-batch-1"]["_simulation_distributions"] == store.get_simulation_distributions(
        "pg-batch-1"
    )
    assert "_simulation_distributions" in traces["pg-batch-2"]
    store.close()


def test_postgres_repository_concurrent_read_write_smoke(postgres_url):
    errors: list[BaseException] = []

    def writer(i: int) -> None:
        try:
            local = TraceStore()
            local.persist(_make_trace(f"pg-thread-{i}"))
            local.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            local = TraceStore()
            local.count()
            local.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
    threads.extend(threading.Thread(target=reader) for _ in range(6))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_migration_tool_is_idempotent(postgres_url, tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.delenv("DATABASE_URL", raising=False)
        source = tmp_path / "source.db"
        sqlite_store = TraceStore(db_path=str(source))
        sqlite_store.persist(_make_trace("migrate-1"))
        sqlite_store.attach_outcome("migrate-1", 100, 95)
        sqlite_store.close()

    module = importlib.import_module("tools.migrate_sqlite_to_postgres")
    dry = module.run_migration(source=str(source), database_url=None, dry_run=True)
    assert dry["traces"]["source_rows"] == 1

    first = module.run_migration(source=str(source), database_url=postgres_url, dry_run=False)
    second = module.run_migration(source=str(source), database_url=postgres_url, dry_run=False)
    assert first["traces"]["insert_attempted"] == 1
    assert second["traces"]["insert_attempted"] == 1

    store = TraceStore()
    assert store.count() == 1
    assert len(store.query_traces(has_outcome=True)) == 1
    store.close()
