"""Tests for `scripts/ingest_traces.py --explain` (no-write validation report)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ingest_traces  # type: ignore  # noqa: E402

from omega.trace.store import TraceStore  # noqa: E402


def _good_block() -> dict:
    return {
        "trace": {
            "trace_id": "sandbox-good",
            "ran_at": "2026-05-28T00:00:00Z",
            "kind": "game",
            "session_id": "sess-20260528-good",
            "input_snapshot": {"home_team": "BOS", "away_team": "NYY", "league": "MLB"},
            "result": {
                "status": "success",
                "simulation": {"home_win_prob": 0.5, "away_win_prob": 0.5},
            },
        },
        "bet_record": None,
    }


def test_explain_reports_valid_and_reject_without_writing(tmp_path, monkeypatch, caplog):
    inbox = tmp_path / "traces"
    inbox.mkdir()
    (inbox / "good.json").write_text(json.dumps(_good_block()), encoding="utf-8")
    (inbox / "bad.json").write_text(json.dumps({"foo": 1}), encoding="utf-8")
    db = tmp_path / "t.db"
    sessions = tmp_path / "sessions"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_traces.py",
            "--inbox",
            str(inbox),
            "--db",
            str(db),
            "--explain",
            "--sidecar-dir",
            str(sessions),
        ],
    )
    with caplog.at_level(logging.INFO):
        rc = ingest_traces.main()

    assert rc == 0
    text = "\n".join(r.message for r in caplog.records)
    assert "VALID good.json" in text
    assert "REJECT bad.json" in text
    # No-write contract: files stay in inbox, nothing persisted.
    assert (inbox / "good.json").exists()
    assert (inbox / "bad.json").exists()
    store = TraceStore(db_path=str(db))
    try:
        assert store.count() == 0
    finally:
        store.close()
