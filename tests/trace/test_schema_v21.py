"""
Tests for trace schema V21 (issue #28): signal_performance CLV columns,
signal_proposals table, and traces.llm_reasoning — plus migration idempotency.
"""

from __future__ import annotations

import tempfile

from omega.strategy.signal_performance import SignalPerformanceRow
from omega.trace.schema import CURRENT_VERSION
from omega.trace.store import TraceStore


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _columns(store: TraceStore, table: str) -> set[str]:
    return {row[1] for row in store.conn.execute(f"PRAGMA table_info({table})").fetchall()}


class TestSchemaV21:
    def test_fresh_db_reaches_v21(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            version = store.conn.execute("SELECT MAX(version) FROM schema_versions").fetchone()[0]
            assert version == CURRENT_VERSION >= 21
        finally:
            store.close()

    def test_signal_performance_has_clv_columns(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            cols = _columns(store, "signal_performance")
            assert {
                "clv_aligned",
                "clv_cents_when_followed",
                "clv_sample",
                "clv_cents_std",
            } <= cols
        finally:
            store.close()

    def test_traces_has_llm_reasoning(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            assert "llm_reasoning" in _columns(store, "traces")
        finally:
            store.close()

    def test_signal_proposals_table_exists(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            cols = _columns(store, "signal_proposals")
            assert {"name", "feature_combo", "lifecycle", "clv_aligned"} <= cols
        finally:
            store.close()

    def test_migration_idempotent_on_reopen(self):
        path = _tmp_db()
        TraceStore(db_path=path).close()
        # Re-opening replays _ensure_schema; must be a clean no-op at V21.
        store = TraceStore(db_path=path)
        try:
            version = store.conn.execute("SELECT MAX(version) FROM schema_versions").fetchone()[0]
            assert version == CURRENT_VERSION
        finally:
            store.close()


class TestSignalPerformanceClvRoundTrip:
    def test_upsert_without_clv_attrs_writes_nulls(self):
        # A SignalPerformanceRow built without CLV fields must still persist
        # (graceful degradation), leaving the CLV columns NULL.
        store = TraceStore(db_path=_tmp_db())
        try:
            row = SignalPerformanceRow(
                signal_type="recent_form",
                source="boxscore_derived",
                obs_window="last_5",
                league="NBA",
                sample_size=40,
                direction_correct=18,
                direction_accuracy=0.45,
                mean_confidence=0.7,
                realized_hit_rate=0.45,
                calibration_gap=0.25,
                brier=0.3,
            )
            store.upsert_signal_performance([row], dataset_hash="hash_a")
            got = store.get_signal_performance(league="NBA")
            assert len(got) == 1
            assert got[0]["clv_aligned"] is None
            assert got[0]["clv_cents_when_followed"] is None
            assert got[0]["clv_sample"] in (0, None)
        finally:
            store.close()


class TestSignalProposals:
    def test_upsert_get_roundtrip(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            combo = {"op": "AND", "terms": [{"feature": "usage", "op": ">", "threshold": 0.3}]}
            store.upsert_signal_proposal(
                name="usage_when_teammate_out",
                feature_combo=combo,
                thesis="Usage spikes when the primary teammate is ruled out.",
                plane="player",
                direction_rule="over",
            )
            props = store.get_signal_proposals()
            assert len(props) == 1
            p = props[0]
            assert p["name"] == "usage_when_teammate_out"
            assert p["lifecycle"] == "probation"
            assert p["feature_combo"] == combo  # decoded back to dict

        finally:
            store.close()

    def test_upsert_idempotent_preserves_created_at(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            store.upsert_signal_proposal(name="p1", thesis="v1")
            created = store.get_signal_proposals()[0]["created_at"]
            store.upsert_signal_proposal(name="p1", thesis="v2")
            rows = store.get_signal_proposals()
            assert len(rows) == 1  # no duplicate
            assert rows[0]["thesis"] == "v2"  # definition updated
            assert rows[0]["created_at"] == created  # created_at preserved
        finally:
            store.close()

    def test_set_lifecycle(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            store.upsert_signal_proposal(name="p1")
            assert store.set_proposal_lifecycle("p1", "active") is True
            assert store.get_signal_proposals(lifecycle="active")[0]["name"] == "p1"
            assert store.get_signal_proposals(lifecycle="probation") == []
            assert store.set_proposal_lifecycle("missing", "active") is False
        finally:
            store.close()


class TestLlmReasoningColumn:
    """Issue #28 WS5: the agent narrative is queryable in the llm_reasoning column."""

    def _trace(self, **extra) -> dict:
        base = {
            "trace_id": "t-reason",
            "run_id": "t-reason",
            "timestamp": "2026-06-24T00:00:00Z",
            "prompt": "NBA game",
        }
        base.update(extra)
        return base

    def _read(self, store: TraceStore) -> object:
        return store.conn.execute(
            "SELECT llm_reasoning FROM traces WHERE trace_id = 't-reason'"
        ).fetchone()[0]

    def test_reasoning_narrative_surfaced_to_column(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            store.persist(self._trace(reasoning_narrative="Zone D forces midrange; fade the over."))
            assert self._read(store) == "Zone D forces midrange; fade the over."
        finally:
            store.close()

    def test_explicit_llm_reasoning_preferred(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            store.persist(self._trace(llm_reasoning="explicit", reasoning_narrative="fallback"))
            assert self._read(store) == "explicit"
        finally:
            store.close()

    def test_absent_is_null(self):
        store = TraceStore(db_path=_tmp_db())
        try:
            store.persist(self._trace())
            assert self._read(store) is None
        finally:
            store.close()
