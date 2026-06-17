from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from omega.trace.audit_renderer import render_session_audit
from omega.trace.persistable import PersistableTrace
from omega.trace.session_sidecar import (
    AuditEvent,
    append_audit_events,
)
from omega.trace.store import TraceStore


@pytest.fixture()
def workspace(tmp_path: Path):
    sidecar_dir = tmp_path / "sessions"
    sidecar_dir.mkdir()
    out_dir = tmp_path / "reports"
    db_path = tmp_path / "test_traces.db"
    yield sidecar_dir, out_dir, db_path


def _persist_trace(
    store: TraceStore,
    *,
    trace_id: str,
    session_id: str,
    kind: str = "prop",
) -> None:
    analyze_out: dict = {
        "trace_id": trace_id,
        "ran_at": "2026-05-27T19:00:00Z",
        "kind": kind,
        "session_id": session_id,
        "input_snapshot": {
            "league": "NBA",
            "home_team": "OKC Thunder",
            "away_team": "SA Spurs",
            "player_name": "SGA",
            "prop_type": "pts",
            "line": 31.5,
        },
        "result": {
            "status": "success",
            "over_prob": 0.42,
            "under_prob": 0.58,
            "recommendation": "under",
            "confidence_tier": "B",
        },
        "trace_quality": {
            "aggregate_quality": 0.83,
            "calibration_eligible": True,
        },
    }
    trace = PersistableTrace.from_analyze_output(analyze_out)
    store.persist(trace)


def _write_sidecar(path: Path, session_id: str) -> None:
    payload = {
        "session_id": session_id,
        "opened_at": "2026-05-27T18:00:00Z",
        "closed_at": "2026-05-27T22:00:00Z",
        "model_version": "claude-opus-4-7",
        "purpose": "renderer test session",
        "league": "NBA",
        "window": "2026-05-27",
        "effective_db_path": "test_traces.db",
        "runtime_db_status": "ok",
        "pipeline_status": {"overall": "needs_outcomes", "ingest": "ok"},
        "next_required_action": "attach outcomes",
        "bankroll": 1000.0,
        "bankroll_confirmed": True,
        "exec_stats": {"traces_emitted": 2, "bets_recorded": 0},
        "agent_notes": "Slate scan complete.",
        "audit_events": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_renders_session_with_traces(workspace):
    sidecar_dir, out_dir, db_path = workspace
    session_id = "sess-render-1"
    _write_sidecar(sidecar_dir / f"{session_id}.json", session_id)

    store = TraceStore(db_path=str(db_path))
    _persist_trace(store, trace_id="sandbox-r1", session_id=session_id, kind="prop")
    _persist_trace(store, trace_id="sandbox-r2", session_id=session_id, kind="game")
    store.close()

    append_audit_events(
        sidecar_dir / f"{session_id}.json",
        [
            AuditEvent(
                ts="2026-05-27T18:30:00Z",
                event_type="preflight",
                step="cowork_preflight",
                status="ok",
                notes="all green",
                trace_ids=["sandbox-r1"],
            ),
            AuditEvent(
                ts="2026-05-27T19:00:00Z",
                event_type="downgrade",
                step="kelly_cap",
                status="warn",
                notes="reduced from B to C",
                assumptions=["confidence tier subjective"],
                bugs=["BUG-77: rest_advantage misread"],
            ),
        ],
    )

    written = render_session_audit(
        session_id, db_path=db_path, sidecar_dir=sidecar_dir, out_dir=out_dir
    )

    assert written.exists()
    body = written.read_text(encoding="utf-8")
    assert "Session Audit — sess-render-1" in body
    assert body.startswith("---\ncanonical: false\n")
    assert "source_db_path:" in body
    assert "trace_count_at_generation:" in body
    assert "## Actionability" in body
    assert "needs_outcomes" in body
    assert "attach outcomes" in body
    assert "`sandbox-r1`" in body
    assert "`sandbox-r2`" in body
    assert "## Traces (2)" in body
    assert "## audit_events (2)" in body
    assert "all green" in body
    assert "BUG-77" in body


def test_renderer_pulls_quality_from_db_row_not_sidecar_prose(workspace):
    sidecar_dir, out_dir, db_path = workspace
    session_id = "sess-quant-1"
    _write_sidecar(sidecar_dir / f"{session_id}.json", session_id)

    store = TraceStore(db_path=str(db_path))
    _persist_trace(store, trace_id="sandbox-q1", session_id=session_id, kind="prop")
    store.close()

    # The sidecar's agent_notes deliberately misstates aggregate_quality.
    # The renderer must show 0.83 (from DB), not 0.99.
    sidecar_path = sidecar_dir / f"{session_id}.json"
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    data["agent_notes"] = "WRONG: aggregate_quality=0.99 (lying)"
    sidecar_path.write_text(json.dumps(data), encoding="utf-8")

    written = render_session_audit(
        session_id, db_path=db_path, sidecar_dir=sidecar_dir, out_dir=out_dir
    )
    body = written.read_text(encoding="utf-8")

    assert "0.83" in body
    # The lying number from agent_notes is in the notes section, but the
    # Traces table must show the DB value.
    traces_section = body.split("## Traces")[1].split("##")[0]
    assert "0.83" in traces_section
    assert "0.99" not in traces_section


def test_renderer_handles_session_with_no_traces(workspace):
    sidecar_dir, out_dir, db_path = workspace
    session_id = "sess-empty"
    _write_sidecar(sidecar_dir / f"{session_id}.json", session_id)
    # Force schema creation but persist nothing
    TraceStore(db_path=str(db_path)).close()

    written = render_session_audit(
        session_id, db_path=db_path, sidecar_dir=sidecar_dir, out_dir=out_dir
    )
    body = written.read_text(encoding="utf-8")

    assert "## Traces (0)" in body
    assert "No traces in" in body


def test_renderer_degrades_without_sidecar(workspace):
    sidecar_dir, out_dir, db_path = workspace
    session_id = "sess-no-sidecar"

    store = TraceStore(db_path=str(db_path))
    _persist_trace(store, trace_id="sandbox-orphan", session_id=session_id)
    store.close()

    written = render_session_audit(
        session_id, db_path=db_path, sidecar_dir=sidecar_dir, out_dir=out_dir
    )
    body = written.read_text(encoding="utf-8")

    assert "degraded" in body.lower()
    assert "`sandbox-orphan`" in body
    # Sidecar file itself was not created
    assert not (sidecar_dir / f"{session_id}.json").exists()


def test_renderer_never_reads_legacy_run_artifacts(workspace, monkeypatch):
    sidecar_dir, out_dir, db_path = workspace
    session_id = "sess-no-legacy"
    _write_sidecar(sidecar_dir / f"{session_id}.json", session_id)

    store = TraceStore(db_path=str(db_path))
    _persist_trace(store, trace_id="sandbox-nl1", session_id=session_id)
    store.close()

    real_open = open
    opened: list[str] = []

    def tracking_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    with mock.patch("builtins.open", side_effect=tracking_open):
        render_session_audit(
            session_id, db_path=db_path, sidecar_dir=sidecar_dir, out_dir=out_dir
        )

    for path in opened:
        assert "RUN_TRACE" not in path
        assert "RUN_AUDIT" not in path


def test_renderer_atomic_write_no_tmp_leftover(workspace):
    sidecar_dir, out_dir, db_path = workspace
    session_id = "sess-atomic"
    _write_sidecar(sidecar_dir / f"{session_id}.json", session_id)
    TraceStore(db_path=str(db_path)).close()

    render_session_audit(
        session_id, db_path=db_path, sidecar_dir=sidecar_dir, out_dir=out_dir
    )
    assert list(out_dir.glob("*.tmp")) == []
