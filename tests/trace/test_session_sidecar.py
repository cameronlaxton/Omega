from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from omega.trace.session_sidecar import SessionSidecar


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
