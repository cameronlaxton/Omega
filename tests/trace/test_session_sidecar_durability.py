"""
Sidecar durability: atomic create, warn-only safe load, opt-in quarantine,
diagnostic JSONL mirror, tri-state quality-gate status, and JSONL recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

from omega.trace.session_sidecar import (
    SessionSidecar,
    ProtectedValueError,
    append_audit_events,
    bootstrap_payload,
    create_sidecar,
    load_sidecar_safe,
    quality_gate_status,
    quarantine_sidecar,
    rebuild_sidecar_from_jsonl,
    append_null_data_audit,
)


def _open(path: Path):
    create_sidecar(
        path,
        bootstrap_payload(
            "sess-20260528-zzzz",
            model_version="claude-test",
            purpose="unit",
            bankroll=1000.0,
            bankroll_confirmed=True,
        ),
    )


def _event(status: str = "ok", event_type: str = "preflight") -> dict:
    return {
        "ts": "2026-05-28T00:00:00Z",
        "event_type": event_type,
        "step": "x",
        "status": status,
        "trace_ids": [],
    }


class TestSafeLoadAndQuarantine:
    def test_load_sidecar_safe_returns_none_and_does_not_move(self, tmp_path):
        path = tmp_path / "sess.json"
        path.write_text("{ this is not valid json", encoding="utf-8")
        assert load_sidecar_safe(path) is None
        # warn-only: file stays put, no quarantine side effect
        assert path.exists()
        assert not (tmp_path / "invalid").exists()

    def test_quarantine_moves_with_reason_idempotently(self, tmp_path):
        path = tmp_path / "sess.json"
        path.write_text("{ broken", encoding="utf-8")
        dst = quarantine_sidecar(path, "JSONDecodeError: boom")
        assert dst is not None and dst.exists()
        assert not path.exists()  # moved
        reason = dst.with_suffix(dst.suffix + ".reason.txt")
        assert reason.exists() and "boom" in reason.read_text(encoding="utf-8")
        # Idempotent: quarantining something already under invalid/ is a no-op.
        assert quarantine_sidecar(dst, "again") is None


class TestQualityGateStatus:
    def test_unknown_for_unreadable(self):
        assert quality_gate_status(None) == "unknown"

    def test_pass_and_fail(self, tmp_path):
        path = tmp_path / "s.json"
        _open(path)
        assert quality_gate_status(load_sidecar_safe(path)) == "pass"
        append_audit_events(path, [_event(status="fail", event_type="quality_gate")])
        assert quality_gate_status(load_sidecar_safe(path)) == "fail"


class TestJsonlMirror:
    def test_mirror_written_on_append(self, tmp_path):
        path = tmp_path / "s.json"
        _open(path)
        append_audit_events(path, [_event(), _event(status="warn")])
        jsonl = path.with_suffix(".events.jsonl")
        assert jsonl.exists()
        lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        # 0 from bootstrap (no audit_events) + 2 appended
        assert len(lines) == 2
        for ln in lines:
            json.loads(ln)  # each line parses

    def test_mirror_survives_truncated_summary(self, tmp_path):
        path = tmp_path / "s.json"
        _open(path)
        append_audit_events(path, [_event(event_type="engine_run"), _event(status="fail", event_type="quality_gate")])
        jsonl = path.with_suffix(".events.jsonl")

        # Simulate a truncated summary JSON.
        path.write_text('{"session_id": "sess-2026', encoding="utf-8")
        assert load_sidecar_safe(path) is None  # summary unreadable

        recovered = rebuild_sidecar_from_jsonl(jsonl)
        assert recovered["event_count"] == 2
        statuses = {e["status"] for e in recovered["audit_events"]}
        assert "fail" in statuses  # the quality_gate fail survived for recovery


class TestAtomicCreate:
    def test_create_sidecar_is_round_trippable(self, tmp_path):
        path = tmp_path / "s.json"
        sidecar = create_sidecar(
            path,
            bootstrap_payload(
                "sess-20260528-bbbb",
                model_version="m",
                purpose="p",
                bankroll=500.0,
            ),
        )
        assert sidecar.session_id == "sess-20260528-bbbb"
        reloaded = load_sidecar_safe(path)
        assert reloaded is not None and reloaded.bankroll == 500.0
        assert not path.with_suffix(path.suffix + ".tmp").exists()  # no temp leftover


class TestNotedBugFixes:
    def test_rebuild_sidecar_is_schema_compliant(self, tmp_path):
        path = tmp_path / "sess-20260530-tst1.json"
        _open(path)
        append_audit_events(path, [_event(), _event(status="warn")])
        
        jsonl = path.with_suffix(".events.jsonl")
        recovered = rebuild_sidecar_from_jsonl(jsonl)
        
        # Verify it has diagnostic helper keys
        assert recovered["event_count"] == 2
        assert recovered["source_jsonl"] == str(jsonl)
        
        # Verify it validates cleanly against the SessionSidecar schema!
        recovered_copy = {k: v for k, v in recovered.items() if k not in ("event_count", "source_jsonl")}
        sidecar = SessionSidecar.model_validate(recovered_copy)
        assert sidecar.session_id == "sess-20260530-tst1"
        assert sidecar.opened_at.endswith("Z")
        assert len(sidecar.audit_events) == 2

    def test_timezone_format_consistency(self, tmp_path):
        payload = bootstrap_payload("sess-1", model_version="m", purpose="p", bankroll=100.0)
        assert payload["opened_at"].endswith("Z")
        
        path = tmp_path / "sess-20260530-tst2.json"
        create_sidecar(path, payload)
        
        append_null_data_audit(path, ["test_var"])
        reloaded = load_sidecar_safe(path)
        assert reloaded is not None
        assert reloaded.opened_at.endswith("Z")
        
        null_event = next(e for e in reloaded.audit_events if e.step == "null_data_audit")
        assert null_event.ts.endswith("Z")

    def test_recursive_protected_field_check(self, tmp_path):
        path = tmp_path / "sess-20260530-tst3.json"
        _open(path)
        
        # Nested protected field inside lists/dicts
        nested_event = _event()
        nested_event["inputs"] = {"traces": [{"edge_pct": 0.05}]}
        
        import pytest
        with pytest.raises(ProtectedValueError) as excinfo:
            append_audit_events(path, [nested_event])
        assert "contains protected engine field" in str(excinfo.value)
