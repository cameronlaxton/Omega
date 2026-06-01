from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import quarantine_legacy_traces  # type: ignore  # noqa: E402

from omega.trace.store import TraceStore  # noqa: E402


def test_quarantine_dry_run_reports_without_writing(tmp_path):
    db_path = tmp_path / "omega.db"
    store = TraceStore(db_path=db_path)
    store.persist(
        {
            "trace_id": "legacy-prop",
            "run_id": "run-legacy",
            "timestamp": "2026-05-20T00:00:00Z",
            "prompt": "legacy prop",
            "league": "NBA",
            "kind": "prop",
            "input_snapshot": {"player_name": "Test Player", "line": 20.5},
            "result": {"status": "success"},
            "predictions": {"over_prob": 0.55, "under_prob": 0.45},
        }
    )
    store.close()

    changed, counts = quarantine_legacy_traces.quarantine(db_path, apply=False)

    assert changed == 1
    assert counts["legacy_missing_context_source"] == 1
    assert counts["legacy_missing_identity"] == 1

    reopened = TraceStore(db_path=db_path)
    trace = reopened.get_trace("legacy-prop")
    assert "trace_quality" not in trace
    reopened.close()


def test_quarantine_apply_tags_without_deleting(tmp_path):
    db_path = tmp_path / "omega.db"
    store = TraceStore(db_path=db_path)
    store.persist(
        {
            "trace_id": "baseline-game",
            "run_id": "run-baseline",
            "timestamp": "2026-05-20T00:00:00Z",
            "prompt": "baseline game",
            "league": "NBA",
            "kind": "game",
            "input_snapshot": {
                "home_team": "Home",
                "away_team": "Away",
                "game_date": "2026-05-20",
            },
            "result": {
                "status": "success",
                "context_source": "league_default",
                "baseline_used": True,
            },
            "predictions": {"home_win_prob": 0.55, "away_win_prob": 0.45},
        }
    )
    store.close()

    changed, counts = quarantine_legacy_traces.quarantine(db_path, apply=True)

    assert changed == 1
    assert counts["baseline_default_context"] == 1

    reopened = TraceStore(db_path=db_path)
    assert reopened.count() == 1
    trace = reopened.get_trace("baseline-game")
    assert trace["trace_quality"]["calibration_eligible"] is False
    assert trace["trace_quality"]["context_source"] == "league_default"
    assert "baseline_default_context" in trace["trace_quality"]["calibration_exclusion_reasons"]
    reopened.close()

