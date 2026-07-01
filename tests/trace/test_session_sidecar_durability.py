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
    write_sidecar,
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
        sidecar = _sidecar_with([_gate("fail", ts="2026-05-28T12:00:00Z", trace_ids=["trace-A"])])
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
        append_audit_events(
            path,
            [_event(event_type="engine_run"), _event(status="fail", event_type="quality_gate")],
        )
        jsonl = path.with_suffix(".events.jsonl")

        # Simulate a truncated summary JSON.
        path.write_text('{"session_id": "sess-2026', encoding="utf-8")
        assert load_sidecar_safe(path) is None  # summary unreadable

        recovered = rebuild_sidecar_from_jsonl(jsonl)
        # Recovery preserved both events, and the rebuilt dict round-trips through
        # the schema as-is (extra="forbid" — no diagnostic-only keys).
        assert len(recovered["audit_events"]) == 2
        SessionSidecar.model_validate(recovered)
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

    def test_create_sidecar_creates_missing_parent_before_lock(self, tmp_path):
        path = tmp_path / "fresh" / "sessions" / "sess-new.json"

        create_sidecar(
            path,
            bootstrap_payload("sess-new", model_version="m", purpose="p", bankroll=500.0),
        )

        assert path.exists()
        assert not path.with_suffix(path.suffix + ".lock").exists()


class TestNotedBugFixes:
    def test_rebuild_sidecar_is_schema_compliant(self, tmp_path):
        path = tmp_path / "sess-20260530-tst1.json"
        _open(path)
        append_audit_events(path, [_event(), _event(status="warn")])

        jsonl = path.with_suffix(".events.jsonl")
        recovered = rebuild_sidecar_from_jsonl(jsonl)

        # The raw return MUST validate as-is: no stripping. SessionSidecar is
        # extra="forbid", so any diagnostic-only key would fail here. This pins
        # the round-trip the recovery path actually relies on.
        sidecar = SessionSidecar.model_validate(recovered)
        assert sidecar.session_id == "sess-20260530-tst1"
        assert sidecar.opened_at.endswith("Z")
        assert len(sidecar.audit_events) == 2

    def test_bootstrap_payload_carries_db_status_fields(self, tmp_path):
        # Default None (backward compatible) ...
        default = bootstrap_payload("sess-a", model_version="m", purpose="p", bankroll=100.0)
        assert default["effective_db_path"] is None
        assert default["runtime_db_status"] is None
        # ... and populated when supplied, surviving a create/reload round-trip.
        path = tmp_path / "sess-db.json"
        create_sidecar(
            path,
            bootstrap_payload(
                "sess-b",
                model_version="m",
                purpose="p",
                bankroll=100.0,
                effective_db_path="/var/omega_traces.db",
                runtime_db_status="default",
            ),
        )
        reloaded = load_sidecar_safe(path)
        assert reloaded is not None
        assert reloaded.effective_db_path == "/var/omega_traces.db"
        assert reloaded.runtime_db_status == "default"

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


class TestConcurrencyRaceF2:
    """SIDECAR_LOGGING_AUDIT_2026-06-07 F2: the read-modify-write in
    append_audit_events must be serialized so concurrent appends don't clobber
    each other (the mechanism behind the 18 audit_events count mismatches)."""

    def test_concurrent_appends_all_land(self, tmp_path):
        import threading

        path = tmp_path / "sess-conc.json"
        _open(path)

        n = 24
        barrier = threading.Barrier(n)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                barrier.wait()  # release all threads at once to maximize contention
                append_audit_events(
                    path,
                    [
                        {
                            "ts": "2026-05-28T00:00:00Z",
                            "event_type": "note",
                            "step": f"w{i}",
                            "status": "ok",
                            "trace_ids": [],
                        }
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        reloaded = load_sidecar_safe(path)
        assert reloaded is not None
        # No append was lost to a read-modify-write race.
        assert len(reloaded.audit_events) == n
        jsonl = path.with_suffix(".events.jsonl")
        lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == n
        # Every worker's distinct step survived (no silent overwrite).
        assert {e.step for e in reloaded.audit_events} == {f"w{i}" for i in range(n)}

    def test_dead_owner_lock_is_reclaimed(self, tmp_path):
        import time

        from omega.trace import session_sidecar as ss

        path = tmp_path / "sess-stale.json"
        lock_path = path.with_suffix(path.suffix + ss._LOCK_SUFFIX)
        lock_path.write_text(
            json.dumps({"pid": 0, "created_at": time.time(), "lock_version": 1}),
            encoding="utf-8",
        )

        acquired = False
        with ss._sidecar_lock(path):
            acquired = True
        assert acquired
        assert not lock_path.exists()  # released/cleaned up

    def test_live_owner_lock_is_not_reclaimed_by_age(self, tmp_path, monkeypatch):
        import os
        import time

        import pytest

        from omega.trace import session_sidecar as ss

        monkeypatch.setattr(ss, "_LOCK_TIMEOUT_SECONDS", 0.1)
        monkeypatch.setattr(ss, "_LOCK_POLL_SECONDS", 0.01)

        path = tmp_path / "sess-live.json"
        lock_path = path.with_suffix(path.suffix + ss._LOCK_SUFFIX)
        lock_path.write_text(
            json.dumps({"pid": os.getpid(), "created_at": time.time(), "lock_version": 1}),
            encoding="utf-8",
        )
        old = time.time() - 3600
        os.utime(lock_path, (old, old))

        with pytest.raises(TimeoutError, match="another writer may be stuck"):
            with ss._sidecar_lock(path):
                pass
        assert lock_path.exists()

    def test_lock_survives_lockfile_vanishing_during_metadata_read(self, tmp_path, monkeypatch):
        # TOCTOU: the holder can unlink the lock between the O_EXCL failure and the
        # metadata read. The waiter must retry, not crash.
        from omega.trace import session_sidecar as ss

        path = tmp_path / "sess-toctou.json"
        lock_path = path.with_suffix(path.suffix + ss._LOCK_SUFFIX)
        lock_path.write_text(
            json.dumps({"pid": 0, "created_at": 0.0, "lock_version": 1}),
            encoding="utf-8",
        )

        real_read_text = Path.read_text
        calls = {"n": 0}

        def flaky_read_text(self, *args, **kwargs):
            if self == lock_path:
                calls["n"] += 1
                if calls["n"] == 1:
                    lock_path.unlink(missing_ok=True)
                    raise FileNotFoundError
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)

        acquired = False
        with ss._sidecar_lock(path):  # must not raise
            acquired = True
        assert acquired
        assert calls["n"] >= 1


class TestMirrorSupersetF3:
    """SIDECAR_LOGGING_AUDIT_2026-06-07 F3: the JSONL mirror must stay a faithful
    superset — no duplicate events on re-write, no silent drops of distinct
    same-second events."""

    def test_write_sidecar_twice_does_not_duplicate_mirror(self, tmp_path):
        path = tmp_path / "sess-dup.json"
        _open(path)
        append_audit_events(path, [_event(event_type="engine_run")])
        reloaded = load_sidecar_safe(path)
        assert reloaded is not None
        write_sidecar(path, reloaded)
        write_sidecar(path, reloaded)
        jsonl = path.with_suffix(".events.jsonl")
        lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1  # not 3 — dedup-aware mirror

    def test_mirror_retains_distinct_same_second_events(self, tmp_path):
        # Two events share ts + step + event_type but differ in notes. A coarse
        # (ts, step, event_type) dedup key would drop the second; the full-content
        # signature keeps both — the exact F3 regression this guards (the observed
        # sess-20260701-ops1 same-second multiplicity).
        path = tmp_path / "sess-samesec.json"
        _open(path)
        append_audit_events(
            path,
            [
                {
                    "ts": "2026-05-28T12:57:33Z",
                    "event_type": "data_provenance",
                    "step": "inject",
                    "status": "ok",
                    "notes": "player A",
                    "trace_ids": [],
                },
                {
                    "ts": "2026-05-28T12:57:33Z",
                    "event_type": "data_provenance",
                    "step": "inject",
                    "status": "ok",
                    "notes": "player B",
                    "trace_ids": [],
                },
            ],
        )
        reloaded = load_sidecar_safe(path)
        assert reloaded is not None
        # Force the full-list mirror path (write_sidecar -> _mirror_missing_events)
        # and confirm it neither drops nor duplicates the two same-second events.
        write_sidecar(path, reloaded)
        jsonl = path.with_suffix(".events.jsonl")
        lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2
        assert sorted(json.loads(ln)["notes"] for ln in lines) == ["player A", "player B"]


class TestSessionIdCollision:
    """sess-20260701-ops1: three independent conversations reused one session ID
    and interleaved writes. create_sidecar now fails closed on collision."""

    def test_create_sidecar_fails_closed_on_existing_path(self, tmp_path):
        import pytest

        path = tmp_path / "sess-collide.json"
        _open(path)
        with pytest.raises(FileExistsError, match="already exists"):
            _open(path)  # a second, unrelated create at the same path

    def test_concurrent_create_sidecar_allows_only_one_winner(self, tmp_path):
        import threading

        path = tmp_path / "sess-concurrent-create.json"
        n = 8
        barrier = threading.Barrier(n)
        successes: list[str] = []
        collisions: list[FileExistsError] = []
        unexpected: list[Exception] = []

        def worker(i: int) -> None:
            payload = bootstrap_payload(
                f"sess-concurrent-{i}",
                model_version="m",
                purpose="p",
                bankroll=100.0,
            )
            try:
                barrier.wait()
                created = create_sidecar(path, payload)
                successes.append(created.session_id)
            except FileExistsError as exc:
                collisions.append(exc)
            except Exception as exc:  # noqa: BLE001
                unexpected.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not unexpected, unexpected
        assert len(successes) == 1
        assert len(collisions) == n - 1
        reloaded = load_sidecar_safe(path)
        assert reloaded is not None
        assert reloaded.session_id == successes[0]

    def test_allow_reopen_returns_open_session(self, tmp_path):
        path = tmp_path / "sess-reopen.json"
        _open(path)
        again = create_sidecar(
            path,
            bootstrap_payload("sess-x", model_version="m", purpose="p", bankroll=1.0),
            allow_reopen=True,
        )
        assert again.closed_at is None  # continuing the same still-open session

    def test_allow_reopen_rejects_closed_session(self, tmp_path):
        import pytest

        path = tmp_path / "sess-closed.json"
        _open(path)
        sc = load_sidecar_safe(path)
        assert sc is not None
        sc.closed_at = "2026-05-28T13:00:00Z"
        write_sidecar(path, sc)
        with pytest.raises(FileExistsError, match="already closed"):
            create_sidecar(
                path,
                bootstrap_payload("sess-y", model_version="m", purpose="p", bankroll=1.0),
                allow_reopen=True,
            )
