"""
Tests for scripts/ingest_traces.py — sandbox export ingestion.

Covers:
- Export-block (shape A): wrapped trace + bet_record + clv_capture_instructions
- Raw analyze() output (shape B): top-level trace_id/kind, no wrapper
- Bad JSON / missing trace_id → moved to failed/ with sibling .error.txt
- Idempotent re-run on processed/ files (no duplicate row)
- bet_record persisted with selection_descriptor inferred from clv_capture_instructions
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make `scripts/` importable
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from omega.trace.store import TraceStore  # noqa: E402
import ingest_traces  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyze_out(trace_id: str = "sandbox-abc123", kind: str = "prop") -> Dict[str, Any]:
    """A minimal but realistic omega_lite_standalone.analyze() return value."""
    if kind == "prop":
        return {
            "trace_id": trace_id,
            "model_version": "omega-lite-v1",
            "ran_at": "2026-05-14T19:23:11Z",
            "kind": "prop",
            "input_snapshot": {
                "player_name": "Jayson Tatum",
                "league": "NBA",
                "prop_type": "pts",
                "line": 27.5,
                "odds_over": -115,
                "odds_under": -105,
                "seed": 42,
            },
            "result": {
                "player_name": "Jayson Tatum",
                "league": "NBA",
                "prop_type": "pts",
                "line": 27.5,
                "status": "success",
                "over_prob": 0.56,
                "under_prob": 0.44,
                "edge_over": 4.2,
                "edge_under": -3.1,
                "recommendation": "over",
                "confidence_tier": "B",
            },
            "quality_gate": {
                "applied": True,
                "aggregate_quality": 0.82,
                "data_completeness": {},
                "downgrades": [],
            },
        }
    # game
    return {
        "trace_id": trace_id,
        "model_version": "omega-lite-v1",
        "ran_at": "2026-05-14T19:23:11Z",
        "kind": "game",
        "input_snapshot": {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "league": "NBA",
            "odds": {"moneyline_home": -150, "moneyline_away": 130},
            "seed": 42,
        },
        "result": {
            "matchup": "Celtics @ Lakers",
            "league": "NBA",
            "status": "success",
            "simulation": {"home_win_prob": 58.0, "away_win_prob": 42.0},
            "edges": [],
            "best_bet": None,
        },
        "quality_gate": {
            "applied": True,
            "aggregate_quality": 0.85,
            "data_completeness": {},
            "downgrades": [],
        },
    }


def _make_export_block(trace_id: str = "sandbox-abc123", with_bet: bool = True) -> Dict[str, Any]:
    block = {
        "trace": _make_analyze_out(trace_id=trace_id, kind="prop"),
        "bet_record": None,
        "clv_capture_instructions": {
            "league": "NBA",
            "event_date": "2026-05-14",
            "matchup": "Boston Celtics @ Miami Heat",
            "market": "player_prop:pts",
            "selection_descriptor": "Tatum_over_27.5_pts",
            "line_at_decision": 27.5,
            "odds_at_decision": -115,
            "book_at_decision": "DraftKings",
        },
    }
    if with_bet:
        block["bet_record"] = {
            "book": "DraftKings",
            "market": "player_prop:pts",
            "selection": "Tatum Over 27.5 pts",
            "line_taken": 27.5,
            "odds_taken": -115,
            "stake_units": 1.0,
            "decision_timestamp": "2026-05-14T19:25:00Z",
        }
    return block


@pytest.fixture()
def workspace(tmp_path: Path):
    """Provide an isolated inbox + db pair."""
    inbox = tmp_path / "inbox" / "traces"
    inbox.mkdir(parents=True)
    db_path = tmp_path / "test_traces.db"
    yield inbox, db_path


def _write_file(inbox: Path, name: str, payload: Dict[str, Any]) -> Path:
    p = inbox / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Shape A: export block
# ---------------------------------------------------------------------------

class TestExportBlock:
    def test_round_trip_with_bet(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "sandbox-abc123.json", _make_export_block("sandbox-abc123"))

        monkeypatch.setattr(sys, "argv", [
            "ingest_traces.py", "--inbox", str(inbox), "--db", str(db_path),
        ])
        rc = ingest_traces.main()
        assert rc == 0

        store = TraceStore(db_path=str(db_path))
        retrieved = store.get_trace("sandbox-abc123")
        assert retrieved is not None
        assert retrieved["trace_id"] == "sandbox-abc123"
        assert retrieved["league"] == "NBA"
        assert retrieved["kind"] == "prop"

        bets = store.get_bet_records("sandbox-abc123")
        assert len(bets) == 1
        assert bets[0]["book"] == "DraftKings"
        assert bets[0]["selection_descriptor"] == "Tatum_over_27.5_pts"
        assert bets[0]["odds_taken"] == -115.0
        assert bets[0]["status"] == "pending"
        store.close()

        # File moved to processed/
        assert not (inbox / "sandbox-abc123.json").exists()
        assert (inbox / "processed" / "sandbox-abc123.json").exists()

    def test_round_trip_no_bet(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "sandbox-xyz.json", _make_export_block("sandbox-xyz", with_bet=False))

        monkeypatch.setattr(sys, "argv", [
            "ingest_traces.py", "--inbox", str(inbox), "--db", str(db_path),
        ])
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        assert store.get_trace("sandbox-xyz") is not None
        assert store.get_bet_records("sandbox-xyz") == []
        store.close()


# ---------------------------------------------------------------------------
# Shape B: raw analyze() output
# ---------------------------------------------------------------------------

class TestRawAnalyzeOutput:
    def test_raw_output_is_accepted(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "raw.json", _make_analyze_out("sandbox-raw1", kind="game"))

        monkeypatch.setattr(sys, "argv", [
            "ingest_traces.py", "--inbox", str(inbox), "--db", str(db_path),
        ])
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        trace = store.get_trace("sandbox-raw1")
        assert trace is not None
        assert trace["league"] == "NBA"
        assert trace["matchup"] == "Celtics @ Lakers"
        store.close()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

class TestFailedFiles:
    def test_malformed_json_moves_to_failed(self, workspace, monkeypatch):
        inbox, db_path = workspace
        bad = inbox / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")

        monkeypatch.setattr(sys, "argv", [
            "ingest_traces.py", "--inbox", str(inbox), "--db", str(db_path),
        ])
        assert ingest_traces.main() == 0  # main returns 0 even with file-level failures

        assert not bad.exists()
        assert (inbox / "failed" / "bad.json").exists()
        assert (inbox / "failed" / "bad.json.error.txt").exists()

    def test_missing_trace_id_moves_to_failed(self, workspace, monkeypatch):
        inbox, db_path = workspace
        bad_payload = _make_export_block("sandbox-broken")
        bad_payload["trace"].pop("trace_id")
        _write_file(inbox, "no_id.json", bad_payload)

        monkeypatch.setattr(sys, "argv", [
            "ingest_traces.py", "--inbox", str(inbox), "--db", str(db_path),
        ])
        assert ingest_traces.main() == 0

        assert (inbox / "failed" / "no_id.json").exists()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotent:
    def test_double_ingest_same_trace(self, workspace, monkeypatch):
        inbox, db_path = workspace
        _write_file(inbox, "first.json", _make_export_block("sandbox-dup"))

        monkeypatch.setattr(sys, "argv", [
            "ingest_traces.py", "--inbox", str(inbox), "--db", str(db_path),
        ])
        assert ingest_traces.main() == 0

        # Drop the same trace_id again as a new inbox file
        _write_file(inbox, "second.json", _make_export_block("sandbox-dup"))
        assert ingest_traces.main() == 0

        store = TraceStore(db_path=str(db_path))
        # Only one row for this trace_id
        rows = store.conn.execute(
            "SELECT COUNT(*) AS n FROM traces WHERE trace_id = ?",
            ("sandbox-dup",),
        ).fetchone()
        assert rows["n"] == 1
        # Only one bet (UNIQUE on trace_id+market+selection_descriptor)
        bets = store.get_bet_records("sandbox-dup")
        assert len(bets) == 1
        store.close()


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_does_not_persist_or_move(self, workspace, monkeypatch):
        inbox, db_path = workspace
        path = _write_file(inbox, "dry.json", _make_export_block("sandbox-dry"))

        monkeypatch.setattr(sys, "argv", [
            "ingest_traces.py", "--inbox", str(inbox), "--db", str(db_path), "--dry-run",
        ])
        assert ingest_traces.main() == 0

        assert path.exists()  # not moved
        store = TraceStore(db_path=str(db_path))
        assert store.get_trace("sandbox-dry") is None  # not persisted
        store.close()
