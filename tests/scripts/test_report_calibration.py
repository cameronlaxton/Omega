from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import report_calibration  # type: ignore  # noqa: E402

from omega.trace.store import TraceStore  # noqa: E402


def test_report_counts_default_deny_legacy_calibration_rows(tmp_path):
    store = TraceStore(db_path=tmp_path / "omega.db")
    store.persist(
        {
            "trace_id": "legacy-nba",
            "run_id": "run-legacy",
            "timestamp": "2026-05-20T00:00:00+00:00",
            "prompt": "legacy",
            "league": "NBA",
            "kind": "game",
            "result": {"status": "success"},
            "predictions": {"home_win_prob": 0.55, "away_win_prob": 0.45},
        }
    )
    store.attach_outcome("legacy-nba", home_score=100, away_score=90)

    counts = report_calibration._section_counts(store, "NBA", "2026-05-01T00:00:00+00:00")
    crps = report_calibration._section_distribution_crps(
        store,
        "NBA",
        "2026-05-01T00:00:00+00:00",
    )

    assert counts["with_predictions"] == 0
    assert counts["graded_calibration"] == 0
    assert crps is None
    store.close()


def test_report_calibration_writes_derived_header(tmp_path, monkeypatch):
    db = tmp_path / "omega.db"
    out = tmp_path / "latest.md"
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    TraceStore(db_path=db).close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_calibration.py",
            "--league",
            "NBA",
            "--db",
            str(db),
            "--out",
            str(out),
            "--sessions-inbox",
            str(sessions),
        ],
    )

    assert report_calibration.main() == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("---\ncanonical: false\n")
    assert f"source_db_path: {str(db)!r}" in text
    assert "trace_count_at_generation:" in text
