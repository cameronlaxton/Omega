from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import mark_session_qa_failed  # type: ignore  # noqa: E402

from omega.trace.session_sidecar import SessionSidecar  # noqa: E402


def _sidecar_payload(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "opened_at": "2026-05-28T18:00:00Z",
        "closed_at": "2026-05-28T19:00:00Z",
        "model_version": "omega-core-phase6h",
        "purpose": "qa failed test",
        "bankroll": 1000.0,
        "bankroll_confirmed": True,
        "exec_stats": {},
        "agent_notes": "",
        "audit_events": [],
    }


def test_mark_sidecar_appends_quality_gate_once(tmp_path):
    path = tmp_path / "sess-test.json"
    path.write_text(json.dumps(_sidecar_payload("sess-test")), encoding="utf-8")

    assert mark_session_qa_failed.mark_sidecar(path, reason="QA failed") is True
    assert mark_session_qa_failed.mark_sidecar(path, reason="QA failed") is False

    sidecar = SessionSidecar.from_path(path)
    assert len(sidecar.audit_events) == 1
    event = sidecar.audit_events[0]
    assert event.event_type == "quality_gate"
    assert event.step == "qa_failed_quarantine_0528"
    assert event.status == "fail"
    assert event.notes == "QA failed"

