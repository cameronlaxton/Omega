"""Zero-evidence session blocker (Issue 4 + remediation proof).

Proves that "no evidence" is not treated as harmless: a session dominated by
zero-evidence-empty-context traces fails the run summary with a diagnostic.
"""

from __future__ import annotations

from omega.trace.quality import (
    ZERO_EVIDENCE_EMPTY_CONTEXT_THRESHOLD,
    summarize_zero_evidence,
)


def _ze_trace(i: int) -> dict:
    return {
        "trace_id": f"sandbox-{i}",
        "trace_quality": {
            "quality_reasons": ["zero_evidence_empty_context"],
            "evidence_status": "empty",
            "context_source": None,
        },
    }


def _clean_trace(i: int) -> dict:
    return {
        "trace_id": f"ok-{i}",
        "trace_quality": {
            "quality_reasons": [],
            "evidence_status": "present",
            "context_source": "provided",
        },
    }


class TestSummarizeZeroEvidence:
    def test_836_zero_evidence_traces_block_the_run(self):
        traces = [_ze_trace(i) for i in range(836)]
        summary = summarize_zero_evidence(traces)
        assert summary.count == 836
        assert summary.blocked is True
        assert summary.diagnostic  # non-empty diagnostic
        assert "836" in summary.diagnostic

    def test_threshold_boundary(self):
        # Exactly the threshold is NOT blocked; one more is.
        at = [_ze_trace(i) for i in range(ZERO_EVIDENCE_EMPTY_CONTEXT_THRESHOLD)]
        over = [_ze_trace(i) for i in range(ZERO_EVIDENCE_EMPTY_CONTEXT_THRESHOLD + 1)]
        assert summarize_zero_evidence(at).blocked is False
        assert summarize_zero_evidence(over).blocked is True

    def test_clean_traces_are_not_counted(self):
        traces = [_clean_trace(i) for i in range(50)]
        summary = summarize_zero_evidence(traces)
        assert summary.count == 0
        assert summary.blocked is False
        assert summary.diagnostic == ""

    def test_legacy_fallback_derivation(self):
        # A legacy trace without quality_reasons is still classified from raw
        # evidence_status + context_source.
        legacy = [
            {
                "trace_id": f"legacy-{i}",
                "trace_quality": {"evidence_status": "empty", "context_source": "baseline"},
            }
            for i in range(15)
        ]
        summary = summarize_zero_evidence(legacy)
        assert summary.count == 15
        assert summary.blocked is True


class TestSessionReportBlocker:
    def test_report_renders_blocker_section(self):
        from omega.trace.session_report.markdown import render_intake_markdown
        from omega.trace.session_report.models import IntakeReportData

        traces = [_ze_trace(i) for i in range(836)]
        summary = summarize_zero_evidence(traces)
        data = IntakeReportData(
            generated_at="now",
            source_db_path="x",
            source_db_fingerprint="f",
            context_mode="persisted",
            trace_count=836,
            ledger_count=0,
            sidecar_status="ok",
            coverage=[],
            ledger_linkage=[],
            provenance_split=[],
            cards=[],
            zero_evidence_count=summary.count,
            zero_evidence_blocked=summary.blocked,
            zero_evidence_trace_ids=summary.trace_ids,
            zero_evidence_diagnostic=summary.diagnostic,
        )
        md = render_intake_markdown(data)
        assert "BLOCKER" in md
        assert "836 of 836" in md

    def test_clean_report_has_no_blocker_section(self):
        from omega.trace.session_report.markdown import render_intake_markdown
        from omega.trace.session_report.models import IntakeReportData

        data = IntakeReportData(
            generated_at="now",
            source_db_path="x",
            source_db_fingerprint="f",
            context_mode="persisted",
            trace_count=5,
            ledger_count=0,
            sidecar_status="ok",
            coverage=[],
            ledger_linkage=[],
            provenance_split=[],
            cards=[],
        )
        md = render_intake_markdown(data)
        assert "BLOCKER" not in md
