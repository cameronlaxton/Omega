"""
Tests for scripts/backfill_trace_quality.py — repopulating calibration-eligibility
metadata onto trace_quality blocks written by an older producer.

Covers:
- a stale trace (trace_quality == {aggregate_quality}) with provided context +
  success + complete identity becomes calibration_eligible=True, with provenance
- a stale trace with missing context_source becomes eligible=False with the right
  exclusion reason
- aggregate_quality and other full_trace fields are preserved
- dry-run writes nothing
- idempotency: a second run is a no-op, and natively-correct traces are skipped
- parity: the backfill yields the same eligibility the live engine would compute
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import backfill_trace_quality as bf  # type: ignore  # noqa: E402

from omega.core.contracts.service import derive_calibration_eligibility  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _tmp_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _stale_trace(
    trace_id: str,
    *,
    context_source: str | None = "provided",
    status: str = "success",
    home_team: str = "Boston Celtics",
    away_team: str = "Miami Heat",
    kind: str = "game",
) -> dict[str, Any]:
    """A trace whose trace_quality lacks eligibility keys (old producer shape)."""
    result: dict[str, Any] = {"status": status}
    if context_source is not None:
        result["context_source"] = context_source
    return {
        "trace_id": trace_id,
        "run_id": trace_id,
        "timestamp": "2026-05-28T18:00:00Z",
        "prompt": "game",
        "league": "NBA",
        "matchup": f"{away_team} @ {home_team}",
        "execution_mode": "sandbox_game",
        "kind": kind,
        "aggregate_quality": 0.74,
        "predictions": {"home_win_prob": 55.0},
        "recommendations": None,
        "odds_snapshot": None,
        "downgrades": [],
        "input_snapshot": {"home_team": home_team, "away_team": away_team, "league": "NBA"},
        "result": result,
        "trace_quality": {"aggregate_quality": 0.74},
    }


def _read_tq(store: TraceStore, trace_id: str) -> dict[str, Any]:
    row = store.conn.execute(
        "SELECT full_trace FROM traces WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    return json.loads(row["full_trace"]).get("trace_quality") or {}


def test_provided_context_becomes_eligible():
    store = TraceStore(db_path=_tmp_db())
    store.persist(_stale_trace("sandbox-elig"))

    bf.run(db=store.db_path, league=None, dry_run=False)

    tq = _read_tq(store, "sandbox-elig")
    assert tq["calibration_eligible"] is True
    assert tq["context_source"] == "provided"
    assert tq["identity_status"] == "complete"
    assert tq["calibration_exclusion_reasons"] == []
    assert tq["eligibility_source"] == "backfill_v1"
    # preserved
    assert tq["aggregate_quality"] == 0.74


def test_missing_context_is_ineligible_with_reason():
    store = TraceStore(db_path=_tmp_db())
    store.persist(_stale_trace("sandbox-noctx", context_source=None))

    bf.run(db=store.db_path, league=None, dry_run=False)

    tq = _read_tq(store, "sandbox-noctx")
    assert tq["calibration_eligible"] is False
    assert "legacy_missing_context_source" in tq["calibration_exclusion_reasons"]


def test_dry_run_writes_nothing():
    store = TraceStore(db_path=_tmp_db())
    store.persist(_stale_trace("sandbox-dry"))

    summary = bf.run(db=store.db_path, league=None, dry_run=True)

    assert summary["updated"] == 0
    assert summary["needing_backfill"] == 1
    # unchanged on disk
    tq = _read_tq(store, "sandbox-dry")
    assert tq.get("calibration_eligible") is None


def test_idempotent_second_run_is_noop():
    store = TraceStore(db_path=_tmp_db())
    store.persist(_stale_trace("sandbox-idem"))

    first = bf.run(db=store.db_path, league=None, dry_run=False)
    second = bf.run(db=store.db_path, league=None, dry_run=False)

    assert first["updated"] == 1
    assert second["updated"] == 0
    assert second["skipped_already_present"] == 1


def test_league_filter():
    store = TraceStore(db_path=_tmp_db())
    store.persist(_stale_trace("sandbox-nba"))
    mlb = _stale_trace("sandbox-mlb")
    mlb["league"] = "MLB"
    store.persist(mlb)

    bf.run(db=store.db_path, league="NBA", dry_run=False)

    assert _read_tq(store, "sandbox-nba").get("calibration_eligible") is True
    # MLB untouched by the NBA-filtered run
    assert _read_tq(store, "sandbox-mlb").get("calibration_eligible") is None


def test_parity_with_engine_policy():
    """Backfill must produce the same eligibility the live engine helper would."""
    store = TraceStore(db_path=_tmp_db())
    store.persist(_stale_trace("sandbox-parity"))
    bf.run(db=store.db_path, league=None, dry_run=False)

    tq = _read_tq(store, "sandbox-parity")
    expected = derive_calibration_eligibility(
        status="success",
        context_source="provided",
        baseline_used=False,
        identity_status="complete",
        downgrades=[],
    )
    for key, val in expected.items():
        assert tq[key] == val
