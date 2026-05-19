"""
Tests for the BUG-4 and BUG-5 validation/warning hooks in
scripts/ingest_traces.py.

- BUG-4: a bet_record on a prop trace requires home/away/game_date; the
  ingest must reject and the file gets routed to failed/.
- BUG-5: line/odds drift between trace.input_snapshot and bet_record is
  logged as a warning but does NOT fail ingest.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ingest_traces  # type: ignore  # noqa: E402

from omega.trace.store import TraceStore  # noqa: E402


def _make_prop_export(
    trace_id: str = "sandbox-validate-1",
    *,
    line: float = 27.5,
    odds_over: float = -115,
    line_taken: float = 27.5,
    odds_taken: float = -115,
    descriptor: str = "Tatum_over_27.5_pts",
    with_identity: bool = True,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "player_name": "Jayson Tatum",
        "league": "NBA",
        "prop_type": "pts",
        "line": line,
        "odds_over": odds_over,
        "odds_under": -105,
        "seed": 42,
    }
    if with_identity:
        snap["home_team"] = "Miami Heat"
        snap["away_team"] = "Boston Celtics"
        snap["game_date"] = "2026-05-14"
    return {
        "trace": {
            "trace_id": trace_id,
            "ran_at": "2026-05-14T19:23:11Z",
            "kind": "prop",
            "input_snapshot": snap,
            "result": {
                "recommendation": "over",
                "over_prob": 0.55,
                "under_prob": 0.45,
            },
        },
        "bet_record": {
            "book": "DraftKings",
            "market": "player_prop:pts",
            "selection": "Tatum Over",
            "selection_descriptor": descriptor,
            "line_taken": line_taken,
            "odds_taken": odds_taken,
            "stake_units": 1.0,
            "decision_timestamp": "2026-05-14T19:25:00Z",
        },
    }


@pytest.fixture()
def workspace(tmp_path: Path):
    inbox = tmp_path / "inbox" / "traces"
    inbox.mkdir(parents=True)
    db = tmp_path / "t.db"
    return inbox, db


def _write(inbox: Path, name: str, payload: dict[str, Any]) -> Path:
    p = inbox / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestBug4Validation:
    def test_prop_bet_missing_identity_raises(self, workspace):
        inbox, db = workspace
        path = _write(inbox, "no_id.json", _make_prop_export(with_identity=False))
        store = TraceStore(db_path=str(db))
        with pytest.raises(ValueError, match="OMEGA_COWORK.md"):
            ingest_traces.ingest_file(path, store)
        store.close()

    def test_prop_bet_with_identity_passes(self, workspace):
        inbox, db = workspace
        path = _write(inbox, "ok.json", _make_prop_export(with_identity=True))
        store = TraceStore(db_path=str(db))
        trace_id, bet_id = ingest_traces.ingest_file(path, store)
        assert trace_id == "sandbox-validate-1"
        assert bet_id is not None
        store.close()

    def test_prop_no_bet_does_not_require_identity(self, workspace):
        """Analysis-only prop traces (no bet) can still be missing identity —
        validation only fires when a bet_record is present."""
        inbox, db = workspace
        payload = _make_prop_export(with_identity=False)
        payload["bet_record"] = None
        path = _write(inbox, "analysis_only.json", payload)
        store = TraceStore(db_path=str(db))
        trace_id, bet_id = ingest_traces.ingest_file(path, store)
        assert trace_id == "sandbox-validate-1"
        assert bet_id is None
        store.close()

    def test_failed_file_routed_to_failed_dir(self, workspace, monkeypatch):
        inbox, db = workspace
        _write(inbox, "bad.json", _make_prop_export(with_identity=False))

        monkeypatch.setattr(sys, "argv", [
            "ingest_traces.py", "--inbox", str(inbox), "--db", str(db),
        ])
        rc = ingest_traces.main()
        assert rc == 0  # main returns 0 even when individual files fail
        assert (inbox / "failed" / "bad.json").exists()
        assert (inbox / "failed" / "bad.json.error.txt").exists()


class TestBug5DriftWarnings:
    def test_line_drift_logs_warning(self, workspace, caplog):
        inbox, db = workspace
        path = _write(inbox, "drift.json", _make_prop_export(
            line=25.5, line_taken=26.5,  # 1.0 — exactly at threshold (not >)
        ))
        store = TraceStore(db_path=str(db))
        with caplog.at_level(logging.WARNING, logger="ingest_traces"):
            ingest_traces.ingest_file(path, store)
        # 1.0 is NOT > 1.0; no warning
        assert not any("line drift" in r.message for r in caplog.records)
        store.close()

    def test_line_drift_above_threshold_logs_warning(self, workspace, caplog):
        inbox, db = workspace
        path = _write(inbox, "drift2.json", _make_prop_export(
            line=25.5, line_taken=27.0,
        ))
        store = TraceStore(db_path=str(db))
        with caplog.at_level(logging.WARNING, logger="ingest_traces"):
            ingest_traces.ingest_file(path, store)
        msgs = [r.message for r in caplog.records if "line drift" in r.message]
        assert len(msgs) == 1
        assert "delta=1.50" in msgs[0]
        store.close()

    def test_line_drift_does_not_fail_ingest(self, workspace):
        inbox, db = workspace
        path = _write(inbox, "drift_ok.json", _make_prop_export(
            line=25.5, line_taken=30.0,  # 4.5 drift
        ))
        store = TraceStore(db_path=str(db))
        trace_id, bet_id = ingest_traces.ingest_file(path, store)
        assert trace_id and bet_id  # both persisted despite drift
        store.close()

    def test_odds_drift_warns_when_side_resolvable(self, workspace, caplog):
        inbox, db = workspace
        # over side; analysis odds_over=-115, bet odds_taken=-160 -> delta=45
        path = _write(inbox, "odds.json", _make_prop_export(
            odds_over=-115, odds_taken=-160, descriptor="Tatum_over_27.5_pts",
        ))
        store = TraceStore(db_path=str(db))
        with caplog.at_level(logging.WARNING, logger="ingest_traces"):
            ingest_traces.ingest_file(path, store)
        msgs = [r.message for r in caplog.records if "odds drift" in r.message]
        assert len(msgs) == 1
        store.close()

    def test_no_warning_when_within_threshold(self, workspace, caplog):
        inbox, db = workspace
        path = _write(inbox, "clean.json", _make_prop_export(
            line=27.5, line_taken=27.5, odds_over=-115, odds_taken=-110,
        ))
        store = TraceStore(db_path=str(db))
        with caplog.at_level(logging.WARNING, logger="ingest_traces"):
            ingest_traces.ingest_file(path, store)
        assert not any(
            "drift" in r.message for r in caplog.records
        )
        store.close()
