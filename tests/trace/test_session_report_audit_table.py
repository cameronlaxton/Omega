"""Tests for the bet-level audit table in session reports."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest

from omega.trace.session_report.extractors import extract_intake_report
from omega.trace.session_report.markdown import _render_audit_table, render_intake_markdown
from omega.trace.session_report.models import AuditRow
from omega.trace.store import TraceStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_trace(trace_id: str, session_id: str, league: str = "MLB") -> dict:
    return {
        "trace_id": trace_id,
        "run_id": f"run-{trace_id}",
        "session_id": session_id,
        "timestamp": "2026-06-19T12:00:00Z",
        "kind": "game",
        "league": league,
        "matchup": "Yankees @ Red Sox",
        "result": {
            "status": "success",
            "best_bet": {"selection": "Yankees -1.5", "confidence_tier": "B", "edge_pct": 4.2},
        },
        "input_snapshot": {
            "bookmaker": "betmgm",
            "event_id": "evt-abc-123",
            "odds": {"moneyline_home": -150, "line": -1.5},
        },
        "trace_quality": {
            "calibration_eligible": True,
            "evidence_status": "present",
            "aggregate_quality": "grade_b",
            "output_mode": "actionable",
        },
    }


@pytest.fixture()
def store_with_traces(tmp_path):
    db = tmp_path / "omega_traces.db"
    s = TraceStore(db_path=str(db))
    # Write two traces
    for tid, league in [
        ("sandbox-mlb-001", "MLB"),
        ("sandbox-fifa-001", "FIFA_WORLD_CUP_2026"),
    ]:
        trace = _minimal_trace(tid, "sess-test-001", league)
        s.persist(trace)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# AuditRow model
# ---------------------------------------------------------------------------


def test_audit_row_construction():
    row = AuditRow(
        trace_id="sandbox-mlb-001",
        league="MLB",
        matchup="Yankees @ Red Sox",
        market_type="game",
        selection="Yankees -1.5",
        line="-1.5",
        odds="-150",
        bookmaker="betmgm",
        confidence_tier="B",
        calibration_eligible="True",
        aggregate_quality="grade_b",
        output_mode="actionable",
        evidence_count=3,
        ledger_status="no ledger row",
    )
    assert row.trace_id == "sandbox-mlb-001"
    assert row.evidence_count == 3
    assert row.resolver_warnings == []


def test_audit_row_defaults():
    row = AuditRow(trace_id="t1")
    assert row.evidence_count == 0
    assert row.resolver_warnings == []
    assert row.fallback_usage is None


# ---------------------------------------------------------------------------
# extract_intake_report populates audit_rows
# ---------------------------------------------------------------------------


def test_extract_intake_report_produces_audit_rows(store_with_traces):
    data = extract_intake_report(store_with_traces, session_id="sess-test-001")
    assert len(data.audit_rows) == 2


def test_audit_row_trace_ids_match_cards(store_with_traces):
    data = extract_intake_report(store_with_traces, session_id="sess-test-001")
    card_ids = {c.trace_id for c in data.cards}
    audit_ids = {r.trace_id for r in data.audit_rows}
    assert card_ids == audit_ids


def test_audit_row_extracts_league(store_with_traces):
    data = extract_intake_report(store_with_traces, session_id="sess-test-001")
    leagues = {r.league for r in data.audit_rows}
    assert "MLB" in leagues
    assert "FIFA_WORLD_CUP_2026" in leagues


def test_audit_row_extracts_event_id(store_with_traces):
    data = extract_intake_report(store_with_traces, session_id="sess-test-001")
    # Both traces have event_id in input_snapshot
    event_ids = {r.event_id for r in data.audit_rows if r.event_id}
    assert "evt-abc-123" in event_ids


def test_audit_row_extracts_calibration_eligible(store_with_traces):
    data = extract_intake_report(store_with_traces, session_id="sess-test-001")
    elig_values = {r.calibration_eligible for r in data.audit_rows}
    assert "True" in elig_values


def test_audit_row_present_with_no_ledger_rows(store_with_traces):
    """Traces with no ledger entry should still produce an audit row."""
    data = extract_intake_report(store_with_traces, session_id="sess-test-001")
    no_ledger = [r for r in data.audit_rows if r.ledger_status == "no ledger row"]
    assert len(no_ledger) > 0


# ---------------------------------------------------------------------------
# _render_audit_table
# ---------------------------------------------------------------------------


def test_render_audit_table_empty():
    lines = _render_audit_table([])
    rendered = "\n".join(lines)
    assert "Bet-Level Trust Audit" in rendered
    assert "not captured" in rendered


def test_render_audit_table_nonempty():
    rows = [
        AuditRow(
            trace_id="sandbox-mlb-001",
            league="MLB",
            matchup="Yankees @ Red Sox",
            market_type="game",
            selection="Yankees -1.5",
            bookmaker="betmgm",
            output_mode="actionable",
            confidence_tier="B",
            ledger_status="pending",
            evidence_count=2,
        )
    ]
    lines = _render_audit_table(rows)
    rendered = "\n".join(lines)
    assert "sandbox-mlb-001" in rendered
    assert "MLB" in rendered
    assert "actionable" in rendered


def test_render_audit_table_warnings_tag():
    rows = [
        AuditRow(
            trace_id="t1",
            resolver_warnings=["no exact BetMGM market match", "candidate_events=0"],
        )
    ]
    lines = _render_audit_table(rows)
    rendered = "\n".join(lines)
    assert "⚠" in rendered


def test_render_audit_table_no_fallback_shows_dash():
    rows = [AuditRow(trace_id="t1", fallback_usage=None)]
    lines = _render_audit_table(rows)
    rendered = "\n".join(lines)
    # Fallback column should show "—" when no fallback
    assert "—" in rendered


# ---------------------------------------------------------------------------
# render_intake_markdown includes audit table
# ---------------------------------------------------------------------------


def test_render_intake_markdown_contains_audit_section(store_with_traces):
    data = extract_intake_report(store_with_traces, session_id="sess-test-001")
    md = render_intake_markdown(data)
    assert "## Bet-Level Trust Audit" in md
    assert "trace_id" in md  # header row
    assert "sandbox-mlb-001" in md or "sandbox-fifa-001" in md


def test_render_intake_markdown_audit_table_before_appendix(store_with_traces):
    data = extract_intake_report(store_with_traces, session_id="sess-test-001")
    md = render_intake_markdown(data)
    audit_pos = md.index("## Bet-Level Trust Audit")
    appendix_pos = md.index("## Appendix")
    assert audit_pos < appendix_pos, "Audit table must appear before the Appendix section"
