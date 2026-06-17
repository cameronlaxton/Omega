from __future__ import annotations

import sys

import pytest

from omega.ops import (
    backfill_bets,
    backfill_closing_lines,
    backfill_evidence_signals,
    backfill_trace_quality,
    fetch_closing_lines,
    ingest_closing_lines,
    report_calibration,
)
from omega.strategy.anchor.tracker import AnchorBetTracker

POSTGRES_URL = "postgresql+psycopg://omega:omega@localhost:5432/omega"
MESSAGE = "not yet supported on Postgres backend; SQLite only in Phase 1"


def test_sqlite_only_ops_raise_clear_phase1_error(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)

    calls = [
        lambda: backfill_bets.main(["--dry-run"]),
        lambda: backfill_evidence_signals.main(["--dry-run"]),
        lambda: backfill_trace_quality.main(["--dry-run"]),
        lambda: AnchorBetTracker(),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match=MESSAGE):
            call()


@pytest.mark.parametrize(
    ("module_main", "argv"),
    [
        (backfill_closing_lines.main, ["cmd", "--dry-run"]),
        (fetch_closing_lines.main, ["cmd", "--dry-run"]),
        (ingest_closing_lines.main, ["cmd", "--dry-run"]),
        (report_calibration.main, ["cmd", "--league", "NBA"]),
    ],
)
def test_sqlite_only_no_argv_ops_raise_clear_phase1_error(
    monkeypatch, module_main, argv
):
    monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError, match=MESSAGE):
        module_main()
