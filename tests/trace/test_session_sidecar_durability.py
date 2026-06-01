"""
Sidecar durability: atomic create, warn-only safe load, opt-in quarantine,
diagnostic JSONL mirror, tri-state quality-gate status, and JSONL recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

from omega.trace.session_sidecar import (
    ProtectedValueError,
    SessionSidecar,
    append_audit_events,
    append_null_data_audit,
    bootstrap_payload,
    create_sidecar,
    load_sidecar_safe,
    quality_gate_status,
    quality_gate_verdict_for_trace,
    quarantine_sidecar,
    rebuild_sidecar_from_jsonl,
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


def _sidecar_with(events: list[dict]) -> SessionSidecar:
    """Build an in-memory sidecar carrying the given audit events."""
    payload = bootstrap_payload(
        "sess-20260528-verd",
        model_version="m",
        purpose="p",
        bankroll=100.0,
    )
    payload["audit_events"] = events
    return SessionSidecar.model_validate(payload)


def _gate(status: str, *, ts: str, step: str = "qa", trace_ids: list[str] | None = None) -> dict:
    return {
        "ts": ts,
        "event_type": "quality_gate",
        "step": step,
        "status": status,
        "trace_ids": trace_ids or [],
    }


class TestTraceScopedQualityGateVerdict:
    def test_no_sidecar_returns_unknown_no_sidecar(self):
        verdict = quality_gate_verdict_for_trace(None, "trace-x", "2026-05-28T12:00:00Z")
        assert verdict.verdict == "unknown"
        assert verdict.scope == "no_sidecar"

    def test_failed_gate_for_trace_a_does_not_fail_trace_b(self):
        sidecar = _sidecar_with(
            [_gate("fail", ts="2026-05-28T12:00:00Z", trace_ids=["trace-A"])]
        )
        a = quality_gate_verdict_for_trace(sidecar, "trace-A", "2026-05-28T12:00:00Z")
        b = quality_gate_verdict_for_trace(sidecar, "trace-B", "2026-05-28T12:00:00Z")
        assert a.verdict == "fail" and a.scope == "trace_id"
        assert a.matched_trace_id == "trace-A"
        # The unrelated trace is not condemned by trace-A's failure.
        assert b.verdict == "pass" and b.scope == "unrelated_session_failure"

    def test_passed_gate_for_trace_marks_pass_by_trace_id(self):
        sidecar = _sidecar_with(
            [_gate("ok", ts="2026-05-28T12:00:00Z", step="repaired", trace_ids=["trace-A"])]
        )
        verdict = quality_gate_verdict_for_trace(sidecar, "trace-A", "2026-05-28T12:00:00Z")
        assert verdict.verdict == "pass" and verdict.scope == "trace_id"

    def test_timestamp_scoped_gate_only_applies_to_matching_trace_window(self):
        # One unscoped failed gate. A trace running next to it is tied by time;
        # a trace running hours later is not time-matched and only falls to the
        # conservative session fallback — proving the window matcher is scoped.
        sidecar = _sidecar_with([_gate("fail", ts="2026-05-28T12:00:30Z")])
        near = quality_gate_verdict_for_trace(sidecar, "trace-near", "2026-05-28T12:00:00Z")
        far = quality_gate_verdict_for_trace(sidecar, "trace-far", "2026-05-28T20:00:00Z")
        assert near.verdict == "fail" and near.scope == "timestamp_window"
        assert far.verdict == "fail" and far.scope == "session_fallback"

    def test_session_fallback_is_marked_as_fallback_not_trace_specific(self):
        # Unstructured failed gate (no trace_ids), trace has no usable ran_at:
        # conservative fallback fail, explicitly flagged as session-scoped.
        sidecar = _sidecar_with([_gate("fail", ts="2026-05-28T12:00:00Z")])
        verdict = quality_gate_verdict_for_trace(sidecar, "trace-x", None)
        assert verdict.verdict == "fail"
        assert verdict.scope == "session_fallback"
        assert verdict.matched_trace_id is None

    def test_later_recovery_event_prevents_pre_trace_fatal_from_poisoning_later_trace(self):
        preflight_fail = {
            "ts": "2026-05-28T10:00:00Z",
            "event_type": "preflight",
            "step": "boot",
            "status": "fail",
            "trace_ids": [],
        }
        recovery = {
            "ts": "2026-05-28T11:00:00Z",
            "event_type": "preflight",
            "step": "boot_retry",
            "status": "ok",
            "trace_ids": [],
        }
        ran_at = "2026-05-28T12:00:00Z"

        poisoned = quality_gate_verdict_for_trace(
            _sidecar_with([preflight_fail]), "trace-x", ran_at
        )
        assert poisoned.verdict == "fail" and poisoned.scope == "pre_trace_fatal"

        recovered = quality_gate_verdict_for_trace(
            _sidecar_with([preflight_fail, recovery]), "trace-x", ran_at
        )
        assert recovered.verdict == "pass"


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
