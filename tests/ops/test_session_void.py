from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omega.ops import session_void
from omega.ops.ingest_traces import ingest_file
from omega.trace.session_sidecar import SessionSidecar, bootstrap_payload, create_sidecar
from omega.trace.store import TraceStore


def _analyze_out(trace_id: str, session_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "model_version": "omega-core-phase6h",
        "ran_at": "2026-07-09T12:00:00Z",
        "kind": "prop",
        "session_id": session_id,
        "input_snapshot": {
            "player_name": "Tatum",
            "league": "NBA",
            "prop_type": "pts",
            "line": 27.5,
            "home_team": "Celtics",
            "away_team": "Heat",
            "game_date": "2026-07-09",
            "seed": 42,
        },
        "result": {
            "player_name": "Tatum",
            "league": "NBA",
            "prop_type": "pts",
            "status": "success",
            "over_prob": 0.55,
            "under_prob": 0.45,
            "recommendation": "over",
        },
        "quality_gate": {
            "applied": True,
            "aggregate_quality": 0.8,
            "data_completeness": {},
            "downgrades": [],
        },
    }


def _seed_session(db_path: Path, session_id: str, *, n_traces: int = 2) -> None:
    store = TraceStore(db_path=str(db_path))
    for i in range(n_traces):
        payload = {
            "trace": _analyze_out(f"sandbox-void-{i}", session_id),
            "bet_record": {
                "book": "DraftKings",
                "market": "player_prop:pts",
                "selection": "Tatum Over 27.5 pts",
                "selection_descriptor": f"Tatum_over_27.5_pts_{i}",
                "line_taken": 27.5,
                "odds_taken": -115,
                "stake_units": 1.0,
                "decision_timestamp": "2026-07-09T12:05:00Z",
            },
        }
        # ingest_file reads a JSON file; write payload to a scratch path first.
        tmp_file = db_path.parent / f"_seed_{session_id}_{i}.json"
        tmp_file.write_text(json.dumps(payload), encoding="utf-8")
        ingest_file(tmp_file, store)
    store.close()


class TestSessionVoidDryRun:
    def test_dry_run_reports_but_does_not_delete(self, tmp_path):
        db_path = tmp_path / "t.db"
        _seed_session(db_path, "sess-void-dry")

        rc = session_void.main(
            ["--session-id", "sess-void-dry", "--reason", "test", "--db", str(db_path)]
        )

        assert rc == 0
        store = TraceStore(db_path=str(db_path))
        remaining = store.conn.execute(
            "SELECT COUNT(*) FROM traces WHERE session_id = ?", ("sess-void-dry",)
        ).fetchone()[0]
        store.close()
        assert remaining == 2  # untouched

    def test_empty_session_is_a_no_op(self, tmp_path):
        db_path = tmp_path / "t.db"
        TraceStore(db_path=str(db_path)).close()  # create empty schema, no data

        rc = session_void.main(
            [
                "--session-id",
                "sess-nonexistent",
                "--reason",
                "test",
                "--db",
                str(db_path),
                "--apply",
            ]
        )

        assert rc == 0


class TestSessionVoidApply:
    def test_apply_exports_then_deletes_all_related_rows(self, tmp_path):
        db_path = tmp_path / "t.db"
        archive_dir = tmp_path / "void_archive"
        _seed_session(db_path, "sess-void-apply", n_traces=2)

        rc = session_void.main(
            [
                "--session-id",
                "sess-void-apply",
                "--reason",
                "duplicate rerun, repairing roster_context",
                "--db",
                str(db_path),
                "--archive-dir",
                str(archive_dir),
                "--apply",
            ]
        )

        assert rc == 0

        # Rows are gone from every table the audit's raw-SQL script touched.
        store = TraceStore(db_path=str(db_path))
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM traces WHERE session_id = ?", ("sess-void-apply",)
            ).fetchone()[0]
            == 0
        )
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM bet_ledger WHERE session_id = ?", ("sess-void-apply",)
            ).fetchone()[0]
            == 0
        )
        store.close()

        # An export landed before the delete -- recoverable by inspection.
        exports = list(archive_dir.glob("sess-void-apply_*.json"))
        assert len(exports) == 1
        exported = json.loads(exports[0].read_text(encoding="utf-8"))
        assert exported["session_id"] == "sess-void-apply"
        assert len(exported["traces"]) == 2

    def test_apply_records_audit_event_on_existing_sidecar(self, tmp_path):
        db_path = tmp_path / "t.db"
        sidecar_dir = tmp_path / "sessions"
        sidecar_dir.mkdir()
        sidecar_path = sidecar_dir / "sess-void-sidecar.json"
        create_sidecar(
            sidecar_path,
            bootstrap_payload("sess-void-sidecar", model_version="m", purpose="p", bankroll=500.0),
        )
        _seed_session(db_path, "sess-void-sidecar", n_traces=1)

        rc = session_void.main(
            [
                "--session-id",
                "sess-void-sidecar",
                "--reason",
                "bad roster_context, rerunning",
                "--db",
                str(db_path),
                "--sidecar-dir",
                str(sidecar_dir),
                "--archive-dir",
                str(tmp_path / "void_archive"),
                "--apply",
            ]
        )

        assert rc == 0
        sidecar = SessionSidecar.from_path(sidecar_path)
        void_events = [e for e in sidecar.audit_events if e.step == "session_void"]
        assert len(void_events) == 1
        assert "bad roster_context" in void_events[0].notes

    def test_apply_without_sidecar_still_completes(self, tmp_path, caplog):
        db_path = tmp_path / "t.db"
        _seed_session(db_path, "sess-void-no-sidecar", n_traces=1)

        with caplog.at_level("WARNING"):
            rc = session_void.main(
                [
                    "--session-id",
                    "sess-void-no-sidecar",
                    "--reason",
                    "test",
                    "--db",
                    str(db_path),
                    "--sidecar-dir",
                    str(tmp_path / "no_such_dir"),
                    "--archive-dir",
                    str(tmp_path / "void_archive"),
                    "--apply",
                ]
            )

        assert rc == 0
        assert any("NOT recorded" in r.message for r in caplog.records)
