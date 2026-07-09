"""ETL standard 2 — Pydantic validate-or-fail at the ingestion boundary.

A renamed/missing column must raise SourceSchemaDriftError, write a fail-status
data_provenance sidecar event, and write nothing downstream — never a silent
None coercion.

References:
  omega/integrations/_etl.py::validate_records
  docs/phase7/MULTI_SPORT_EXPANSION.md  (Part 5B standard 2; verification test 14)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from omega.integrations._etl import SourceSchemaDriftError, _emit_provenance_event, validate_records
from omega.trace.session_sidecar import SessionSidecar, bootstrap_payload


class _SackmannRow(BaseModel):
    player: str
    surface: str
    spw_pct: float


def _write_sidecar(tmp_path: Path) -> Path:
    payload = bootstrap_payload(
        "sess-etl-test",
        model_version="test",
        purpose="etl unit test",
        bankroll=1000.0,
    )
    path = tmp_path / "sess-etl-test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_records_pass():
    rows = [
        {"player": "Sinner", "surface": "hard", "spw_pct": 0.68},
        {"player": "Alcaraz", "surface": "clay", "spw_pct": 0.66},
    ]
    out = validate_records(rows, _SackmannRow, source="sackmann")
    assert len(out) == 2
    assert out[0].player == "Sinner"


def test_renamed_column_raises_and_does_not_coerce():
    # Upstream renamed "spw_pct" -> "serve_win" : must fail loud, not None-coerce.
    rows = [{"player": "Sinner", "surface": "hard", "serve_win": 0.68}]
    with pytest.raises(SourceSchemaDriftError) as exc:
        validate_records(rows, _SackmannRow, source="sackmann")
    assert exc.value.source == "sackmann"
    assert exc.value.record_index == 0


def test_drift_writes_fail_status_provenance_event(tmp_path):
    sidecar_path = _write_sidecar(tmp_path)
    rows = [{"player": "Sinner", "surface": "hard"}]  # missing spw_pct

    with pytest.raises(SourceSchemaDriftError):
        validate_records(rows, _SackmannRow, source="sackmann", session_path=sidecar_path)

    sidecar = SessionSidecar.from_path(sidecar_path)
    drift_events = [
        e for e in sidecar.audit_events if e.event_type == "data_provenance" and e.status == "fail"
    ]
    assert len(drift_events) == 1
    assert "sackmann" in drift_events[0].step


def test_provenance_write_failure_is_logged_not_silent(tmp_path, caplog):
    """F11 (SIDECAR_LOGGING_AUDIT_2026-06-07): a total failure of the
    provenance audit write (e.g. an unreadable/nonexistent sidecar) must not
    raise past this best-effort helper, but it must not be a silent
    `except: pass` either -- it needs to log so the lost QA signal is visible."""
    nonexistent = tmp_path / "does_not_exist" / "sess-ghost.json"

    with caplog.at_level("WARNING"):
        _emit_provenance_event(nonexistent, source="sackmann", status="fail", notes="test")

    assert any("data_provenance" in r.message for r in caplog.records)
    assert any(str(nonexistent) in r.message for r in caplog.records)
