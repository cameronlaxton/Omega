"""
Tests for omega-score-evidence-signals coverage reporting.

The scorer must report empty-evidence traces by status (an evidence-learning
gap), keep scoring traces that DO carry evidence, and never treat
"no evidence rows" as a total failure.
"""

from __future__ import annotations

import logging
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

import score_evidence_signals  # type: ignore  # noqa: E402

from omega.trace.store import TraceStore  # noqa: E402


def _tmp_store() -> tuple[TraceStore, str]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return TraceStore(db_path=path), path


_HOME_SIGNAL = {
    "signal_type": "rest_edge",
    "category": "schedule",
    "plane": "game",
    "source": "manual",
    "confidence": 0.7,
    "window": "last_5",
    "direction": "home",
    "stat_key": "rest_days",
    "value": 1,
}


def _game_trace(trace_id: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-20T12:00:00Z",
        "kind": "game",
        "league": "NBA",
        "input_snapshot": {"evidence": evidence},
        "predictions": {"home_win_prob": 58.0},
        "trace_quality": {
            "evidence_status": "present" if evidence else "empty",
            "calibration_eligible": True,
            "context_source": "provided",
            "identity_status": "complete",
        },
    }


def _seed_graded(store: TraceStore, trace_id: str, evidence: list[dict[str, Any]]) -> None:
    store.persist(_game_trace(trace_id, evidence))
    store.attach_outcome(trace_id, home_score=110, away_score=100)  # home_win


def test_empty_evidence_traces_are_reported_by_status():
    store, _ = _tmp_store()
    _seed_graded(store, "t-empty", [])
    graded = store.query_traces(has_outcome=True, limit=1000)
    scored, summary = score_evidence_signals.collect_scores(store, graded)

    assert summary.graded_traces == 1
    assert summary.evidence_present == 0
    assert summary.skipped_empty == 1
    assert scored == []
    store.close()


def test_signal_scoring_still_produces_rows_for_available_evidence():
    store, _ = _tmp_store()
    _seed_graded(store, "t-evi", [dict(_HOME_SIGNAL)])
    graded = store.query_traces(has_outcome=True, limit=1000)
    scored, summary = score_evidence_signals.collect_scores(store, graded)

    assert summary.evidence_present == 1
    assert len(scored) == 1
    assert scored[0].direction_correct is True  # home signal, home_win outcome
    store.close()


def test_no_evidence_rows_is_not_reported_as_total_failure(monkeypatch, caplog):
    store, path = _tmp_store()
    _seed_graded(store, "t-has", [dict(_HOME_SIGNAL)])
    _seed_graded(store, "t-none", [])
    store.close()

    monkeypatch.setattr(sys, "argv", ["score_evidence_signals.py", "--db", path])
    with caplog.at_level(logging.INFO):
        rc = score_evidence_signals.main()
    assert rc == 0  # success, not failure

    text = caplog.text
    assert "Skipped: empty evidence:       1" in text
    assert "Evidence-eligible (present):   1" in text

    # The trace with evidence still produced and wrote a signal-performance row.
    store2 = TraceStore(db_path=path)
    rows = store2.conn.execute("SELECT COUNT(*) FROM signal_performance").fetchone()[0]
    assert rows >= 1
    store2.close()


def test_qa_failed_graded_trace_is_skipped_by_status():
    store, _ = _tmp_store()
    _seed_graded(store, "t-qa", [dict(_HOME_SIGNAL)])
    # Mark a failed QA verdict for this trace.
    from omega.trace.session_sidecar import TraceQaVerdict

    store.write_qa_verdict("t-qa", TraceQaVerdict(verdict="fail", scope="trace_id"), session_id="s")
    graded = store.query_traces(has_outcome=True, limit=1000)
    scored, summary = score_evidence_signals.collect_scores(store, graded)

    assert summary.skipped_qa_failed == 1
    assert summary.evidence_present == 0
    assert scored == []
    store.close()
