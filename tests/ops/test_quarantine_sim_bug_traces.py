from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import quarantine_sim_bug_traces  # type: ignore  # noqa: E402

from omega.trace.store import TraceStore  # noqa: E402


def _persist_mlb_trace(db_path: Path, trace_id: str = "mlb-bug") -> None:
    store = TraceStore(db_path=db_path)
    store.persist(
        {
            "trace_id": trace_id,
            "run_id": "run-mlb",
            "timestamp": "2026-05-24T12:00:00+00:00",
            "prompt": "MLB game",
            "league": "MLB",
            "kind": "game",
            "input_snapshot": {
                "home_team": "Home",
                "away_team": "Away",
                "home_context": {"off_rating": 4.5, "def_rating": 3.5},
                "away_context": {"off_rating": 4.0, "def_rating": 4.4},
            },
            "result": {"status": "success", "context_source": "provided"},
            "trace_quality": {
                "calibration_eligible": True,
                "calibration_exclusion_reasons": [],
            },
        }
    )
    store.close()


def test_quarantine_dry_run_does_not_write(tmp_path):
    db_path = tmp_path / "omega.db"
    _persist_mlb_trace(db_path)

    matched, changed = quarantine_sim_bug_traces.quarantine(
        db_path, cutoff="2026-05-25T00:00:00+00:00", apply=False
    )

    assert matched == 1
    assert changed == 0
    store = TraceStore(db_path=db_path)
    trace = store.get_trace("mlb-bug")
    assert trace["trace_quality"]["calibration_eligible"] is True
    store.close()


def test_quarantine_apply_tags_without_deleting_and_rolls_back(tmp_path):
    db_path = tmp_path / "omega.db"
    _persist_mlb_trace(db_path)

    matched, changed = quarantine_sim_bug_traces.quarantine(
        db_path, cutoff="2026-05-25T00:00:00+00:00", apply=True
    )

    assert matched == 1
    assert changed == 1
    store = TraceStore(db_path=db_path)
    assert store.count() == 1
    trace = store.get_trace("mlb-bug")
    assert trace["trace_quality"]["calibration_eligible"] is False
    assert (
        quarantine_sim_bug_traces._REASON
        in trace["trace_quality"]["calibration_exclusion_reasons"]
    )
    store.close()

    matched, changed = quarantine_sim_bug_traces.quarantine(
        db_path, cutoff="2026-05-25T00:00:00+00:00", apply=True, rollback=True
    )
    assert matched == 1
    assert changed == 1
    store = TraceStore(db_path=db_path)
    trace = store.get_trace("mlb-bug")
    assert trace["trace_quality"]["calibration_eligible"] is True
    assert quarantine_sim_bug_traces._REASON not in trace["trace_quality"][
        "calibration_exclusion_reasons"
    ]
    store.close()

