"""Copy Omega trace persistence rows from SQLite V14 to Postgres.

Dry-run mode reads only the source SQLite DB and reports table counts. A real
run requires a Postgres URL and inserts idempotently with ON CONFLICT DO NOTHING.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from omega.paths import default_trace_db_path
from omega.trace.db import create_postgres_engine, create_session_factory
from omega.trace.models import (
    BetLedgerRow,
    ClosingLineRow,
    EarlyMarketSnapshotRow,
    EvidenceSignalRow,
    MarketSnapshotRow,
    OutcomeRow,
    PropOutcomeRow,
    SchemaVersionRow,
    SignalPerformanceRow,
    SimulationDistributionRow,
    TraceQaVerdictRow,
    TraceRow,
)
from omega.trace.store import TraceStore

TABLES = (
    ("traces", TraceRow.__table__),
    ("outcomes", OutcomeRow.__table__),
    ("prop_outcomes", PropOutcomeRow.__table__),
    ("bet_ledger", BetLedgerRow.__table__),
    ("evidence_signals", EvidenceSignalRow.__table__),
    ("signal_performance", SignalPerformanceRow.__table__),
    ("simulation_distributions", SimulationDistributionRow.__table__),
    ("closing_lines", ClosingLineRow.__table__),
    ("early_market_snapshots", EarlyMarketSnapshotRow.__table__),
    ("trace_qa_verdicts", TraceQaVerdictRow.__table__),
    ("market_snapshots", MarketSnapshotRow.__table__),
    ("schema_versions", SchemaVersionRow.__table__),
)

SEQUENCE_TABLES = {
    "evidence_signals": ("id", "evidence_signals_id_seq"),
    "signal_performance": ("id", "signal_performance_id_seq"),
    "simulation_distributions": (
        "distribution_id",
        "simulation_distributions_distribution_id_seq",
    ),
}


@contextmanager
def _force_sqlite_tracestore():
    old = os.environ.pop("DATABASE_URL", None)
    try:
        yield
    finally:
        if old is not None:
            os.environ["DATABASE_URL"] = old


def _open_source(path: str) -> TraceStore:
    with _force_sqlite_tracestore():
        return TraceStore(db_path=path, read_only=True)


def _table_exists(store: TraceStore, table: str) -> bool:
    row = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _read_rows(store: TraceStore, table: str) -> list[dict[str, Any]]:
    if not _table_exists(store, table):
        return []
    rows = store.conn.execute(f"SELECT * FROM {table}").fetchall()
    return [dict(row) for row in rows]


def _set_sequences(session) -> None:
    for table, (column, sequence) in SEQUENCE_TABLES.items():
        session.execute(
            text(
                f"SELECT setval('{sequence}', "
                f"COALESCE((SELECT MAX({column}) FROM {table}), 1), true)"
            )
        )


def run_migration(
    *,
    source: str,
    database_url: str | None,
    dry_run: bool,
) -> dict[str, dict[str, int]]:
    source_store = _open_source(source)
    try:
        rows_by_table = {name: _read_rows(source_store, name) for name, _table in TABLES}
    finally:
        source_store.close()

    result = {
        name: {"source_rows": len(rows), "insert_attempted": 0}
        for name, rows in rows_by_table.items()
    }
    if dry_run:
        return result
    if not database_url:
        raise RuntimeError("DATABASE_URL or --database-url is required for a real migration")

    engine = create_postgres_engine(database_url)
    Session = create_session_factory(engine)
    with Session() as session:
        with session.begin():
            for name, table in TABLES:
                rows = rows_by_table[name]
                if not rows:
                    continue
                session.execute(pg_insert(table).values(rows).on_conflict_do_nothing())
                result[name]["insert_attempted"] = len(rows)
            _set_sequences(session)
    engine.dispose()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy SQLite V14 trace data into Postgres")
    parser.add_argument(
        "--source",
        default=str(default_trace_db_path()),
        help="Source SQLite DB path (default: var/omega_traces.db)",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Target Postgres SQLAlchemy URL (default: DATABASE_URL)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report source counts only")
    args = parser.parse_args(argv)

    counts = run_migration(
        source=args.source,
        database_url=args.database_url,
        dry_run=args.dry_run,
    )
    for table, data in counts.items():
        if args.dry_run:
            print(f"{table}: source_rows={data['source_rows']}")
        else:
            print(
                f"{table}: source_rows={data['source_rows']} "
                f"insert_attempted={data['insert_attempted']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
