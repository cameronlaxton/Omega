"""In-process command runner: exit-code policy + JSONL audit ledger."""

from __future__ import annotations

import json

import pytest

from omega.historical.lab.runner import LabCommandRunner, LabStepError
from omega.trace.session_sidecar import AuditEvent


def _runner(tmp_path):
    return LabCommandRunner(tmp_path / "command_log.jsonl")


def _read_events(path):
    return [
        AuditEvent.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_run_cli_ok(tmp_path):
    r = _runner(tmp_path)
    rc = r.run_cli("replay", lambda argv: 0, ["--league", "X"])
    assert rc == 0
    events = _read_events(r.log_path)
    assert len(events) == 1
    assert events[0].event_type == "command"
    assert events[0].step == "replay"
    assert events[0].status == "ok"
    assert events[0].outputs["exit_code"] == 0


def test_run_cli_required_failure_raises(tmp_path):
    r = _runner(tmp_path)
    with pytest.raises(LabStepError) as exc:
        r.run_cli("fit", lambda argv: 1, [])
    assert exc.value.exit_code == 1
    assert exc.value.step == "fit"
    # The failure is still recorded before raising.
    events = _read_events(r.log_path)
    assert events[-1].status == "fail"


def test_run_cli_allowed_nonzero_is_warn(tmp_path):
    r = _runner(tmp_path)
    rc = r.run_cli("live-parity", lambda argv: 2, [], ok_exits=(0, 2))
    assert rc == 2
    assert _read_events(r.log_path)[-1].status == "warn"


def test_run_cli_optional_failure_does_not_raise(tmp_path):
    r = _runner(tmp_path)
    rc = r.run_cli("registry-audit", lambda argv: 1, [], required=False)
    assert rc == 1
    assert _read_events(r.log_path)[-1].status == "fail"


def test_record_appends_event(tmp_path):
    r = _runner(tmp_path)
    r.record("grid", "ok", notes="fit 4 variants", outputs={"n_variants": 4})
    r.record("seal", "ok", outputs={"holdout_access_count": 1})
    events = _read_events(r.log_path)
    assert [e.step for e in events] == ["grid", "seal"]
    assert events[0].outputs["n_variants"] == 4
    assert len(r.events) == 2
