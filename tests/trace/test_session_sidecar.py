from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from omega.trace.session_sidecar import (
    AuditEvent,
    ProtectedValueError,
    SessionSidecar,
    append_audit_events,
    append_null_data_audit,
)


def _valid_sidecar() -> dict:
    return {
        "session_id": "sess-20260521-test",
        "opened_at": "2026-05-21T18:00:00Z",
        "closed_at": None,
        "model_version": "claude-sonnet-4-6",
        "purpose": "contract test",
        "bankroll": 1000.0,
        "bankroll_confirmed": False,
        "exec_stats": {"traces_emitted": 1},
        "agent_notes": "",
    }


def _valid_event() -> dict:
    return {
        "ts": "2026-05-27T17:00:00Z",
        "event_type": "preflight",
        "step": "cowork_preflight",
        "status": "ok",
        "notes": "engine green",
        "trace_ids": [],
    }


def test_session_sidecar_validates_required_schema(tmp_path):
    path = tmp_path / "sess.json"
    path.write_text(json.dumps(_valid_sidecar()), encoding="utf-8")

    sidecar = SessionSidecar.from_path(path)

    assert sidecar.session_id == "sess-20260521-test"
    assert sidecar.to_report_dict()["exec_stats"]["traces_emitted"] == 1


def test_session_sidecar_rejects_legacy_timestamp_keys():
    payload = _valid_sidecar()
    payload.pop("opened_at")
    payload["started_at"] = "2026-05-21T18:00:00Z"

    with pytest.raises(ValidationError):
        SessionSidecar.model_validate(payload)


def test_session_sidecar_rejects_inline_outcomes():
    payload = _valid_sidecar()
    payload["outcomes"] = {"SGA_pts_under_31.5": {"actual": 19.0, "result": "win"}}

    with pytest.raises(ValidationError):
        SessionSidecar.model_validate(payload)


def test_session_sidecar_audit_events_defaults_to_empty():
    sidecar = SessionSidecar.model_validate(_valid_sidecar())
    assert sidecar.audit_events == []


def test_session_sidecar_accepts_valid_audit_events():
    payload = _valid_sidecar()
    payload["audit_events"] = [_valid_event()]

    sidecar = SessionSidecar.model_validate(payload)

    assert len(sidecar.audit_events) == 1
    assert sidecar.audit_events[0].event_type == "preflight"
    assert sidecar.audit_events[0].status == "ok"


def test_audit_event_accepts_quality_gate_type():
    event = _valid_event()
    event["event_type"] = "quality_gate"
    event["step"] = "null_data_audit"
    event["status"] = "warn"

    validated = AuditEvent.model_validate(event)

    assert validated.event_type == "quality_gate"
    assert validated.step == "null_data_audit"


def test_audit_event_rejects_invalid_event_type():
    event = _valid_event()
    event["event_type"] = "unknown_type"

    with pytest.raises(ValidationError):
        AuditEvent.model_validate(event)


def test_audit_event_rejects_invalid_status():
    event = _valid_event()
    event["status"] = "maybe"

    with pytest.raises(ValidationError):
        AuditEvent.model_validate(event)


def test_append_audit_events_atomically_adds_events(tmp_path):
    path = tmp_path / "sess.json"
    path.write_text(json.dumps(_valid_sidecar()), encoding="utf-8")

    append_audit_events(path, [_valid_event()])

    sidecar = SessionSidecar.from_path(path)
    assert len(sidecar.audit_events) == 1
    assert sidecar.audit_events[0].step == "cowork_preflight"


def test_append_audit_events_accumulates_across_calls(tmp_path):
    path = tmp_path / "sess.json"
    path.write_text(json.dumps(_valid_sidecar()), encoding="utf-8")

    append_audit_events(path, [_valid_event()])
    second = _valid_event()
    second["event_type"] = "engine_run"
    second["step"] = "analyze"
    append_audit_events(path, [second])

    sidecar = SessionSidecar.from_path(path)
    assert len(sidecar.audit_events) == 2
    assert sidecar.audit_events[1].step == "analyze"


def test_append_audit_events_rejects_protected_inputs_key(tmp_path):
    path = tmp_path / "sess.json"
    path.write_text(json.dumps(_valid_sidecar()), encoding="utf-8")

    bad_event = _valid_event()
    bad_event["inputs"] = {"edge_pct": 5.2, "some_safe_key": "value"}

    with pytest.raises(ProtectedValueError, match="edge_pct"):
        append_audit_events(path, [bad_event])

    sidecar = SessionSidecar.from_path(path)
    assert sidecar.audit_events == []


def test_append_audit_events_rejects_protected_outputs_key(tmp_path):
    path = tmp_path / "sess.json"
    path.write_text(json.dumps(_valid_sidecar()), encoding="utf-8")

    bad_event = _valid_event()
    bad_event["outputs"] = {"kelly_fraction": 0.05}

    with pytest.raises(ProtectedValueError, match="kelly_fraction"):
        append_audit_events(path, [bad_event])

    sidecar = SessionSidecar.from_path(path)
    assert sidecar.audit_events == []


def test_append_null_data_audit_writes_quality_gate_event(tmp_path):
    path = tmp_path / "sess.json"
    path.write_text(json.dumps(_valid_sidecar()), encoding="utf-8")

    append_null_data_audit(
        path,
        ["game_context.rest_days", "injury_impact"],
        critical=True,
        trace_ids=["sandbox-null"],
    )

    sidecar = SessionSidecar.from_path(path)
    event = sidecar.audit_events[0]
    assert event.event_type == "quality_gate"
    assert event.step == "null_data_audit"
    assert event.status == "fail"
    assert event.notes == "NULL detected: game_context.rest_days, injury_impact"
    assert event.trace_ids == ["sandbox-null"]
