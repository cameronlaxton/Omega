from __future__ import annotations

import json

from omega.ops import prove_lifecycle
from omega.trace.store import TraceStore


def test_prove_lifecycle_fixture_success_report_audit_and_replay(tmp_path):
    result = prove_lifecycle.run(
        work_dir=tmp_path,
        keep_artifacts=True,
        skip_preflight=True,
    )

    assert result.ok
    assert [stage.status for stage in result.stages if stage.stage != "preflight"] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
    ]

    db_path = tmp_path / "var" / "omega_traces.db"
    replay_db_path = tmp_path / "var" / "backtest_lifecycle.db"
    report_path = tmp_path / "var" / "reports" / "lifecycle_report.md"
    audit_path = tmp_path / "var" / "reports" / "run_audits" / "sess-lifecycle-proof.audit.md"

    assert db_path.exists()
    assert replay_db_path.exists()
    assert report_path.read_text(encoding="utf-8").startswith("---\ncanonical: false\n")
    assert audit_path.read_text(encoding="utf-8").startswith("---\ncanonical: false\n")

    store = TraceStore(db_path=str(db_path), read_only=True)
    replay_store = TraceStore(db_path=str(replay_db_path), read_only=True)
    try:
        assert len(store.query_by_session("sess-lifecycle-proof")) == 1
        assert len(replay_store.query_by_session("sess-lifecycle-replay")) == 4
        assert not store.query_by_session("sess-lifecycle-replay")
    finally:
        store.close()
        replay_store.close()


def test_prove_lifecycle_json_cli(tmp_path, capsys):
    rc = prove_lifecycle.main(
        [
            "--work-dir",
            str(tmp_path),
            "--keep-artifacts",
            "--skip-preflight",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["work_dir"] == str(tmp_path)
    assert {stage["stage"] for stage in payload["stages"]} >= {
        "ingest",
        "report",
        "audit",
        "replay",
    }
