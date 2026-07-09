from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from omega.ops import sweep_stale_sessions
from omega.trace.session_sidecar import (
    append_audit_events,
    bootstrap_payload,
    close_sidecar,
    create_sidecar,
)


def _one_event() -> dict:
    return {
        "ts": "2026-05-27T17:00:00Z",
        "event_type": "step",
        "step": "test",
        "status": "ok",
    }


def _age_files(*paths: Path, hours: float = 0, days: float = 0) -> None:
    """Backdate mtime so staleness checks don't depend on real wall-clock waits."""
    target = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=hours, days=days)
    ).timestamp()
    for p in paths:
        if p.exists():
            os.utime(p, (target, target))


def _open_session(tmp_path: Path, session_id: str) -> Path:
    path = tmp_path / f"{session_id}.json"
    create_sidecar(
        path,
        bootstrap_payload(session_id, model_version="m", purpose="p", bankroll=500.0),
    )
    return path


class TestFindStaleOpenSessions:
    def test_recent_open_session_not_flagged(self, tmp_path):
        _open_session(tmp_path, "sess-fresh")
        stale = sweep_stale_sessions.find_stale_open_sessions(tmp_path, stale_hours=24.0)
        assert stale == []

    def test_old_open_session_is_flagged(self, tmp_path):
        path = _open_session(tmp_path, "sess-old")
        _age_files(path, hours=48)

        stale = sweep_stale_sessions.find_stale_open_sessions(tmp_path, stale_hours=24.0)

        assert len(stale) == 1
        assert stale[0][0] == path
        assert stale[0][2] >= 24.0

    def test_closed_session_never_flagged_regardless_of_age(self, tmp_path):
        path = _open_session(tmp_path, "sess-closed")
        close_sidecar(path, exec_stats={"traces_emitted": 1})
        _age_files(path, hours=1000)

        stale = sweep_stale_sessions.find_stale_open_sessions(tmp_path, stale_hours=24.0)

        assert stale == []

    def test_apply_closes_the_session_with_labeled_notes(self, tmp_path):
        path = _open_session(tmp_path, "sess-abandoned")
        _age_files(path, hours=48)

        rc = sweep_stale_sessions.main(
            ["--sessions-inbox", str(tmp_path), "--stale-hours", "24", "--apply"]
        )

        assert rc == 0
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["closed_at"] is not None
        assert "AUTO-CLOSED" in data["agent_notes"]

    def test_dry_run_does_not_close(self, tmp_path):
        path = _open_session(tmp_path, "sess-untouched")
        _age_files(path, hours=48)

        rc = sweep_stale_sessions.main(["--sessions-inbox", str(tmp_path), "--stale-hours", "24"])

        assert rc == 0
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["closed_at"] is None


class TestFindPrunableMirrors:
    def test_recently_closed_mirror_not_flagged(self, tmp_path):
        path = _open_session(tmp_path, "sess-recent-close")
        close_sidecar(path, exec_stats={})

        prunable = sweep_stale_sessions.find_prunable_mirrors(tmp_path, prune_mirrors_days=90)

        assert prunable == []

    def test_old_closed_mirror_is_flagged(self, tmp_path):
        path = _open_session(tmp_path, "sess-old-close")
        append_audit_events(path, [_one_event()])  # a mirror only exists once >=1 event is appended
        close_sidecar(path, exec_stats={})
        data = json.loads(path.read_text(encoding="utf-8"))
        data["closed_at"] = "2020-01-01T00:00:00Z"
        path.write_text(json.dumps(data), encoding="utf-8")

        prunable = sweep_stale_sessions.find_prunable_mirrors(tmp_path, prune_mirrors_days=90)

        mirror = path.with_suffix(".events.jsonl")
        assert len(prunable) == 1
        assert prunable[0][0] == mirror

    def test_open_session_mirror_never_flagged(self, tmp_path):
        path = _open_session(tmp_path, "sess-still-open")
        _age_files(path, hours=1000)

        prunable = sweep_stale_sessions.find_prunable_mirrors(tmp_path, prune_mirrors_days=1)

        assert prunable == []

    def test_apply_prune_mirrors_deletes_only_the_mirror(self, tmp_path):
        path = _open_session(tmp_path, "sess-to-prune")
        append_audit_events(path, [_one_event()])  # a mirror only exists once >=1 event is appended
        close_sidecar(path, exec_stats={})
        data = json.loads(path.read_text(encoding="utf-8"))
        data["closed_at"] = "2020-01-01T00:00:00Z"
        path.write_text(json.dumps(data), encoding="utf-8")
        mirror = path.with_suffix(".events.jsonl")
        assert mirror.exists()

        rc = sweep_stale_sessions.main(
            [
                "--sessions-inbox",
                str(tmp_path),
                "--prune-mirrors",
                "--prune-mirrors-days",
                "90",
                "--apply",
            ]
        )

        assert rc == 0
        assert not mirror.exists()
        assert path.exists()  # the sidecar JSON itself is never touched
