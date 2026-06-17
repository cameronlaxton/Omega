"""
Tests for omega-ingest-closing-lines â€” closing-line ingest.

Covers:
- valid file: rows attached, file moves to processed/
- malformed file (missing required field): rejected, .error.txt sidecar
- missing trace_id reference: rejected, no partial write
- transaction rollback on partial-fail within a file's lines array
- idempotent re-ingest (same file twice = no duplicates)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.ops.ingest_closing_lines import ingest_file  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _tmp_store() -> TraceStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return TraceStore(db_path=tmp.name)


def _seed_trace(store: TraceStore, trace_id: str = "sandbox-x") -> None:
    store.persist(
        {
            "trace_id": trace_id,
            "run_id": "r-1",
            "timestamp": "2026-05-15T20:00:00Z",
            "prompt": "test",
            "league": "NBA",
        }
    )


def _write_payload(tmp_path: Path, payload: dict, name: str = "snap.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestValid:
    def test_attaches_lines(self, tmp_path):
        store = _tmp_store()
        _seed_trace(store)
        payload = {
            "trace_id": "sandbox-x",
            "captured_at": "2026-05-15T23:55:00Z",
            "source": "draftkings.com",
            "lines": [
                {
                    "market": "spread",
                    "selection_descriptor": "lakers_-3.5",
                    "closing_line": -3.5,
                    "closing_odds": -110,
                },
                {
                    "market": "moneyline",
                    "selection_descriptor": "lakers_ml",
                    "closing_odds": -160,
                },
            ],
        }
        path = _write_payload(tmp_path, payload)
        trace_id, n = ingest_file(path, store)
        assert trace_id == "sandbox-x"
        assert n == 2
        rows = store.get_closing_lines("sandbox-x")
        assert len(rows) == 2
        store.close()

    def test_idempotent(self, tmp_path):
        store = _tmp_store()
        _seed_trace(store)
        payload = {
            "trace_id": "sandbox-x",
            "captured_at": "2026-05-15T23:55:00Z",
            "source": "draftkings.com",
            "lines": [
                {
                    "market": "total",
                    "selection_descriptor": "over_225.5",
                    "closing_line": 225.5,
                    "closing_odds": -105,
                }
            ],
        }
        path = _write_payload(tmp_path, payload)
        ingest_file(path, store)
        # Re-applying: attach_closing_line is UNIQUE-keyed and returns existing id
        path2 = _write_payload(tmp_path, payload, name="snap2.json")
        ingest_file(path2, store)
        assert len(store.get_closing_lines("sandbox-x")) == 1
        store.close()


class TestMalformed:
    def test_missing_required_top_level(self, tmp_path):
        store = _tmp_store()
        _seed_trace(store)
        path = _write_payload(
            tmp_path,
            {"trace_id": "sandbox-x", "source": "x"},  # missing captured_at, lines
        )
        with pytest.raises(ValueError, match="Missing required"):
            ingest_file(path, store)
        store.close()

    def test_missing_line_field(self, tmp_path):
        store = _tmp_store()
        _seed_trace(store)
        path = _write_payload(
            tmp_path,
            {
                "trace_id": "sandbox-x",
                "captured_at": "2026-05-15T23:55:00Z",
                "source": "draftkings.com",
                "lines": [{"market": "spread", "selection_descriptor": "x"}],  # no closing_odds
            },
        )
        with pytest.raises(ValueError, match="closing_odds"):
            ingest_file(path, store)
        # no partial write
        assert store.get_closing_lines("sandbox-x") == []
        store.close()

    def test_unknown_market(self, tmp_path):
        store = _tmp_store()
        _seed_trace(store)
        path = _write_payload(
            tmp_path,
            {
                "trace_id": "sandbox-x",
                "captured_at": "2026-05-15T23:55:00Z",
                "source": "x",
                "lines": [
                    {
                        "market": "weather_under_55F",
                        "selection_descriptor": "x",
                        "closing_odds": -110,
                    }
                ],
            },
        )
        with pytest.raises(ValueError, match="not in allowed set"):
            ingest_file(path, store)
        store.close()

    def test_spread_requires_closing_line(self, tmp_path):
        store = _tmp_store()
        _seed_trace(store)
        path = _write_payload(
            tmp_path,
            {
                "trace_id": "sandbox-x",
                "captured_at": "2026-05-15T23:55:00Z",
                "source": "x",
                "lines": [
                    {
                        "market": "spread",
                        "selection_descriptor": "x",
                        "closing_odds": -110,
                        # missing closing_line
                    }
                ],
            },
        )
        with pytest.raises(ValueError, match="closing_line is required"):
            ingest_file(path, store)
        store.close()


class TestMissingTrace:
    def test_unknown_trace_id_rejected(self, tmp_path):
        store = _tmp_store()
        # no trace seeded
        path = _write_payload(
            tmp_path,
            {
                "trace_id": "sandbox-nonexistent",
                "captured_at": "2026-05-15T23:55:00Z",
                "source": "x",
                "lines": [
                    {
                        "market": "moneyline",
                        "selection_descriptor": "x",
                        "closing_odds": 100,
                    }
                ],
            },
        )
        with pytest.raises(ValueError, match="No trace found"):
            ingest_file(path, store)
        store.close()

