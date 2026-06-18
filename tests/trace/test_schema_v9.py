"""
Tests for V9 schema migration and evidence persistence:
- evidence_signals and signal_performance tables created and indexed
- persist() explodes input_snapshot.evidence into evidence_signals rows
- evidence rows are idempotent (re-persisting a trace does not duplicate)
- traces without evidence write zero rows (zero behavior change)
- Phase-B evidence_application data is recorded when present
- a fresh DB and a reopened DB both converge to CURRENT_VERSION
"""

from __future__ import annotations

import tempfile

from omega.trace.schema import CURRENT_VERSION
from omega.trace.store import TraceStore


def _tmp_db_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _evidence_signal(**overrides) -> dict:
    base = {
        "signal_type": "recent_form",
        "category": "player_form",
        "plane": "player",
        "value": [27.0, 31.0, 24.0],
        "source": "boxscore_derived",
        "confidence": 0.8,
        "window": "last_5",
        "direction": "over",
        "stat_key": "pts",
        "note": None,
    }
    base.update(overrides)
    return base


def _trace(trace_id: str, *, evidence: list[dict] | None = None, **extra) -> dict:
    blob = {
        "trace_id": trace_id,
        "run_id": "r-" + trace_id,
        "timestamp": "2026-05-22T12:00:00Z",
        "prompt": "x",
        "league": "NBA",
        "matchup": f"away @ home ({trace_id})",
        "execution_mode": "sandbox_prop",
        "kind": "prop",
        "input_snapshot": {"league": "NBA", "evidence": evidence or []},
    }
    blob.update(extra)
    return blob


class TestV9Tables:
    def test_evidence_signals_table_exists(self):
        store = TraceStore(db_path=_tmp_db_path())
        row = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='evidence_signals'"
        ).fetchone()
        assert row is not None
        store.close()

    def test_signal_performance_table_exists(self):
        store = TraceStore(db_path=_tmp_db_path())
        row = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_performance'"
        ).fetchone()
        assert row is not None
        store.close()

    def test_indexes_exist(self):
        store = TraceStore(db_path=_tmp_db_path())
        names = {
            r[0]
            for r in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_evidence_signals_trace_id" in names
        assert "idx_evidence_signals_type" in names
        assert "idx_signal_performance_key" in names
        store.close()

    def test_current_version_is_nineteen(self):
        store = TraceStore(db_path=_tmp_db_path())
        assert CURRENT_VERSION == 19
        assert store.schema_version() == 19
        store.close()

    def test_reopening_store_converges_to_current_version(self):
        path = _tmp_db_path()
        store_a = TraceStore(db_path=path)
        store_a.close()
        store_b = TraceStore(db_path=path)
        assert store_b.schema_version() == CURRENT_VERSION
        store_b.close()


class TestEvidencePersistence:
    def test_evidence_exploded_into_rows(self):
        store = TraceStore(db_path=_tmp_db_path())
        store.persist(
            _trace(
                "t-ev-1",
                evidence=[
                    _evidence_signal(),
                    _evidence_signal(signal_type="usage_spike", value=0.12, stat_key=None),
                ],
            )
        )
        rows = store.get_evidence_signals("t-ev-1")
        assert len(rows) == 2
        assert rows[0]["signal_type"] == "recent_form"
        assert rows[0]["obs_window"] == "last_5"
        assert rows[0]["direction"] == "over"
        assert rows[0]["league"] == "NBA"
        assert rows[0]["applied"] == 0
        assert rows[1]["signal_type"] == "usage_spike"
        store.close()

    def test_value_serialized_as_json(self):
        store = TraceStore(db_path=_tmp_db_path())
        store.persist(_trace("t-ev-json", evidence=[_evidence_signal()]))
        rows = store.get_evidence_signals("t-ev-json")
        assert rows[0]["value_json"] == "[27.0, 31.0, 24.0]"
        store.close()

    def test_no_evidence_writes_zero_rows(self):
        store = TraceStore(db_path=_tmp_db_path())
        store.persist(_trace("t-no-ev", evidence=[]))
        assert store.get_evidence_signals("t-no-ev") == []
        store.close()

    def test_legacy_trace_without_input_snapshot(self):
        # Pre-evidence traces have no input_snapshot at all — must not crash.
        store = TraceStore(db_path=_tmp_db_path())
        blob = {
            "trace_id": "t-legacy",
            "run_id": "r-legacy",
            "timestamp": "2026-05-22T12:00:00Z",
            "prompt": "x",
            "league": "NBA",
        }
        store.persist(blob)
        assert store.get_evidence_signals("t-legacy") == []
        store.close()

    def test_re_persist_does_not_duplicate_evidence(self):
        store = TraceStore(db_path=_tmp_db_path())
        trace = _trace("t-ev-idem", evidence=[_evidence_signal()])
        store.persist(trace)
        store.persist(trace)
        store.persist(trace)
        assert len(store.get_evidence_signals("t-ev-idem")) == 1
        store.close()

    def test_evidence_application_recorded_when_present(self):
        # Phase B supplies an evidence_application list aligned by index.
        store = TraceStore(db_path=_tmp_db_path())
        trace = _trace(
            "t-ev-applied",
            evidence=[_evidence_signal()],
            evidence_mode="shadow",
            evidence_application=[
                {
                    "applied": True,
                    "factor": 1.05,
                    "policy_version": "v1",
                    "evidence_mode": "shadow",
                }
            ],
        )
        store.persist(trace)
        rows = store.get_evidence_signals("t-ev-applied")
        assert rows[0]["applied"] == 1
        assert rows[0]["applied_factor"] == 1.05
        assert rows[0]["policy_version"] == "v1"
        assert rows[0]["evidence_mode"] == "shadow"
        store.close()
