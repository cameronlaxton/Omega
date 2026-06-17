"""
Fresh-DB schema migration-chain test.

Complements the version-specific tests (test_schema_v7, test_schema_v9,
test_trace_store::test_v3_to_v4_migration_idempotent) with a single convergence
check: a brand-new DB stamps EVERY version 1..CURRENT_VERSION, creates all
expected tables/views, and re-opening is a no-op (idempotent).

This is the guard that catches a future version bump that adds DDL but forgets
the matching _record_version() call (or vice-versa).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.trace.schema import CURRENT_VERSION  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

_EXPECTED_TABLES = {
    "traces",
    "outcomes",
    "schema_versions",
    "bet_ledger",  # bet_records was consolidated into this at V14
    "closing_lines",
    "market_snapshots",
    "prop_outcomes",
    "evidence_signals",
    "signal_performance",
    "simulation_distributions",
    "early_market_snapshots",
    "trace_qa_verdicts",
}
_EXPECTED_VIEWS = {"v_distribution_outcomes", "v_bet_ledger_dashboard"}

# V14 consolidation removed this table; assert it stays gone on a fresh DB.
_REMOVED_TABLES = {"bet_records"}


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _stamped_versions(store: TraceStore) -> set[int]:
    return {
        row[0] for row in store.conn.execute("SELECT version FROM schema_versions").fetchall()
    }


def _objects(store: TraceStore, obj_type: str) -> set[str]:
    return {
        row[0]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ?", (obj_type,)
        ).fetchall()
    }


class TestMigrationChain:
    def test_every_version_is_stamped(self):
        store = TraceStore(db_path=_tmp_db())
        assert _stamped_versions(store) == set(range(1, CURRENT_VERSION + 1))
        store.close()

    def test_all_expected_tables_created(self):
        store = TraceStore(db_path=_tmp_db())
        tables = _objects(store, "table")
        missing = _EXPECTED_TABLES - tables
        assert not missing, f"missing tables: {missing}"
        leftover = _REMOVED_TABLES & tables
        assert not leftover, f"tables that should be dropped: {leftover}"
        store.close()

    def test_distribution_outcomes_view_created(self):
        store = TraceStore(db_path=_tmp_db())
        assert _EXPECTED_VIEWS <= _objects(store, "view")
        store.close()

    def test_outcomes_unique_index_present(self):
        # V8 enforces one game outcome per trace via a UNIQUE index.
        store = TraceStore(db_path=_tmp_db())
        indexes = _objects(store, "index")
        assert "idx_outcomes_trace_id_unique" in indexes
        store.close()

    def test_reopen_is_idempotent(self):
        db = _tmp_db()
        store = TraceStore(db_path=db)
        first = _stamped_versions(store)
        store.close()

        store = TraceStore(db_path=db)
        assert _stamped_versions(store) == first
        assert store.schema_version() == CURRENT_VERSION
        store.close()
