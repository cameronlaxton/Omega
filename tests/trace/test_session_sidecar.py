from __future__ import annotations

import json
from pathlib import Path

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


def test_session_sidecar_from_path_strips_trailing_null_pad(tmp_path):
    """BUG-sess-20260524-nba1: a fixed-size write buffer left valid JSON
    followed by a run of null bytes, which json.load rejected as 'Extra
    data'. from_path must strip the pad and parse the valid content."""
    path = tmp_path / "padded.json"
    raw = json.dumps(_valid_sidecar()).encode("utf-8") + b"\x00" * 255
    path.write_bytes(raw)

    sidecar = SessionSidecar.from_path(path)

    assert sidecar.session_id == "sess-20260521-test"


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
    assert sidecar.pipeline_status == {}
    assert sidecar.next_required_action is None


def test_optional_actionability_fields_round_trip():
    payload = _valid_sidecar()
    payload.update(
        {
            "league": "NBA",
            "window": "2026-05-29",
            "effective_db_path": "C:/repos/Omega/var/omega_traces.db",
            "runtime_db_status": "ok",
            "pipeline_status": {"overall": "needs_outcomes", "ingest": "ok"},
            "next_required_action": "attach outcomes when games are final",
        }
    )

    sidecar = SessionSidecar.model_validate(payload)

    assert sidecar.league == "NBA"
    assert sidecar.pipeline_status["overall"] == "needs_outcomes"
    assert sidecar.next_required_action == "attach outcomes when games are final"


def test_existing_repo_sidecars_remain_valid():
    sidecar_dir = Path(__file__).resolve().parents[2] / "inbox" / "sessions"
    if not sidecar_dir.exists():
        pytest.skip("repo sidecar fixtures not present")

    paths = sorted(sidecar_dir.glob("sess-*.json"))
    if not paths:
        pytest.skip("no repo sidecar json files present")
    for path in paths:
        SessionSidecar.from_path(path)


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
