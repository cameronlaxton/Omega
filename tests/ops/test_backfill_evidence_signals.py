"""
Tests for scripts/backfill_evidence_signals.py â€” provenance-safe re-explosion
of evidence_signals from frozen pre-decision trace snapshots.

The backfill must: default to dry-run, re-derive rows only from
input_snapshot.evidence, never fabricate signals from outcomes/engine math,
mark genuinely-empty traces unrecoverable, and be idempotent.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import backfill_evidence_signals  # type: ignore  # noqa: E402

from omega.trace.store import TraceStore  # noqa: E402


def _tmp_store() -> tuple[TraceStore, str]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return TraceStore(db_path=path), path


_SIGNAL = {
    "signal_type": "rest_advantage",
    "category": "schedule",
    "plane": "game",
    "source": "manual",
    "confidence": 0.7,
    "window": "last_5",
    "direction": "home",
    "stat_key": "rest_days",
    "value": 2,
}


def _trace(trace_id: str, evidence: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-20T12:00:00Z",
        "kind": "game",
        "league": "NBA",
        "input_snapshot": {"evidence": evidence if evidence is not None else []},
        "predictions": {"home_win_prob": 55.0},
        "trace_quality": {"evidence_status": "present" if evidence else "empty"},
    }


def _persist_legacy_with_evidence(store: TraceStore, trace_id: str) -> None:
    """Persist a trace with evidence, then delete its exploded rows to simulate
    a pre-V9 trace: evidence lives in the blob but the table has no rows."""
    store.persist(_trace(trace_id, [dict(_SIGNAL)]))
    store.conn.execute("DELETE FROM evidence_signals WHERE trace_id = ?", (trace_id,))
    store.conn.commit()
    assert store.get_evidence_signals(trace_id) == []


def test_backfill_defaults_to_dry_run(monkeypatch):
    store, path = _tmp_store()
    _persist_legacy_with_evidence(store, "t-default")
    store.close()

    # No --apply and no --dry-run => dry-run; nothing written.
    monkeypatch.setattr(sys, "argv", ["backfill_evidence_signals.py", "--db", path])
    rc = backfill_evidence_signals.main()
    assert rc == 0

    store2 = TraceStore(db_path=path)
    assert store2.get_evidence_signals("t-default") == []
    store2.close()


def test_backfill_dry_run_writes_nothing():
    store, _ = _tmp_store()
    _persist_legacy_with_evidence(store, "t-dry")
    summary = backfill_evidence_signals.run_backfill(store, apply=False)
    assert summary.would_explode_rows == 1
    assert summary.applied_rows == 0
    assert store.get_evidence_signals("t-dry") == []
    store.close()


def test_backfill_reexplodes_from_input_snapshot_evidence():
    store, _ = _tmp_store()
    _persist_legacy_with_evidence(store, "t-reexplode")
    summary = backfill_evidence_signals.run_backfill(store, apply=True)
    assert summary.applied_traces == 1
    assert summary.applied_rows == 1
    rows = store.get_evidence_signals("t-reexplode")
    assert len(rows) == 1
    assert rows[0]["signal_type"] == "rest_advantage"
    assert rows[0]["source"] == "manual"
    store.close()


def test_backfill_apply_writes_rows_and_updates_evidence_status():
    store, _ = _tmp_store()
    _persist_legacy_with_evidence(store, "t-status")
    # Force a stale empty status to prove apply corrects it.
    trace = store.get_trace("t-status")
    trace["trace_quality"]["evidence_status"] = "empty"
    store.conn.execute(
        "UPDATE traces SET full_trace = ? WHERE trace_id = ?",
        (__import__("json").dumps(trace, default=str), "t-status"),
    )
    store.conn.commit()

    backfill_evidence_signals.run_backfill(store, apply=True)
    refreshed = store.get_trace("t-status")
    assert refreshed["trace_quality"]["evidence_status"] == "present"
    assert refreshed["trace_quality"]["evidence_provenance"] == "original"
    store.close()


def test_backfill_marks_unrecoverable_without_fake_signals():
    store, _ = _tmp_store()
    store.persist(_trace("t-empty", []))  # genuinely empty evidence
    summary = backfill_evidence_signals.run_backfill(store, apply=True)
    assert summary.unrecoverable_empty_count == 1
    assert summary.applied_rows == 0
    assert store.get_evidence_signals("t-empty") == []
    store.close()


def test_backfill_is_idempotent_on_rerun():
    store, _ = _tmp_store()
    _persist_legacy_with_evidence(store, "t-idem")
    backfill_evidence_signals.run_backfill(store, apply=True)
    first = store.get_evidence_signals("t-idem")
    # Second run: rows already present => counted as already_exploded, no dupes.
    summary = backfill_evidence_signals.run_backfill(store, apply=True)
    assert summary.already_exploded == 1
    assert summary.applied_rows == 0
    assert len(store.get_evidence_signals("t-idem")) == len(first) == 1
    store.close()


def test_backfill_never_reads_outcomes_or_engine_outputs():
    """A trace with empty snapshot evidence but an attached outcome must NOT
    produce any evidence rows â€” evidence cannot be manufactured from results."""
    store, _ = _tmp_store()
    store.persist(_trace("t-outcome", []))
    store.attach_outcome("t-outcome", home_score=110, away_score=104)
    summary = backfill_evidence_signals.run_backfill(store, apply=True)
    assert summary.graded_traces == 1
    assert summary.applied_rows == 0
    assert store.get_evidence_signals("t-outcome") == []
    store.close()

