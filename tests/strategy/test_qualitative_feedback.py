"""Phase 6 (Issue #22) — qualitative-signal feedback gate.

Verifies the report distinguishes the required dimensions on enriched traces and
labels pre-enrichment traces ``insufficient`` (never folding them into a
signal's aggregates).
"""

from __future__ import annotations

from omega.strategy.qualitative_feedback import (
    INSUFFICIENT,
    NO_EVIDENCE,
    SUFFICIENT,
    build_report,
    classify_trace,
    render_report_markdown,
)


def _enriched_prop_trace() -> dict:
    return {
        "trace_id": "t-prop",
        "kind": "prop",
        "evidence_mode": "live",
        "trace_quality": {"calibration_eligible": True},
        "result": {"simulation": {"simulation_backend": "fast_score"}},
        "evidence_application": [
            {
                "signal_type": "usage_spike",
                "target": "mean",
                "applied": True,
                "factor": 1.14,
                "raw_factor": 1.2,
                "reliability_weight": 1.0,
                "per_signal_capped_factor": 1.2,
                "family_role": "singleton",
                "confidence": 0.7,
                "confidence_defaulted": False,
                "final_applied_factor": 1.14,
                "evidence_mode": "live",
            }
        ],
        "_prop_outcomes": [{"stat_value": 25.0, "line": 20.0}],
    }


def _enriched_markov_trace() -> dict:
    return {
        "trace_id": "t-game",
        "kind": "game",
        "evidence_mode": "markov_transition",
        "trace_quality": {"calibration_eligible": False},
        "result": {"simulation": {"simulation_backend": "markov_state_v1"}},
        "evidence_application": [
            {
                "signal_type": "rest_advantage",
                "target": "markov_transition",
                "applied": True,
                "factor": 1.04,
                "effective_scalar": 1.04,
                "confidence": 0.75,
                "confidence_defaulted": False,
                "evidence_mode": "markov_transition",
            }
        ],
        "_outcome": {"result": "home_win"},
    }


def _legacy_trace() -> dict:
    # Pre-enrichment: shallow application dict, none of the normalized fields.
    return {
        "trace_id": "t-old",
        "kind": "prop",
        "evidence_mode": "shadow",
        "evidence_application": [
            {
                "signal_type": "recent_form",
                "target": "mean",
                "applied": False,
                "factor": 1.05,
                "reason": "legacy",
                "policy_version": "adj_v1_seed",
                "evidence_mode": "shadow",
            }
        ],
    }


def _no_evidence_trace() -> dict:
    return {"trace_id": "t-none", "kind": "game", "evidence_application": []}


# ---------------------------------------------------------------------------
# classify_trace
# ---------------------------------------------------------------------------


class TestClassifyTrace:
    def test_enriched_prop_is_sufficient(self):
        c = classify_trace(_enriched_prop_trace())
        assert c.status == SUFFICIENT
        assert c.market_type == "prop"
        assert c.evidence_mode == "live"
        assert c.backend_path == "fast_score"
        assert c.calibration_eligible is True
        assert c.outcome_resolved is True
        assert len(c.signals) == 1
        sig = c.signals[0]
        assert sig.signal_type == "usage_spike"
        assert sig.applied is True
        assert sig.final_factor == 1.14
        assert sig.confidence_defaulted is False
        assert sig.family_role == "singleton"

    def test_markov_backend_and_unresolved_calibration(self):
        c = classify_trace(_enriched_markov_trace())
        assert c.status == SUFFICIENT
        assert c.backend_path == "markov_state_v1"
        assert c.market_type == "game"
        assert c.calibration_eligible is False
        assert c.outcome_resolved is True

    def test_legacy_trace_is_insufficient(self):
        c = classify_trace(_legacy_trace())
        assert c.status == INSUFFICIENT
        assert c.reason is not None
        assert c.signals == ()  # never exposes per-signal rows to scoring

    def test_empty_evidence_is_no_evidence(self):
        c = classify_trace(_no_evidence_trace())
        assert c.status == NO_EVIDENCE
        assert c.signals == ()

    def test_backend_path_falls_back_to_markov_mode(self):
        trace = {
            "trace_id": "t",
            "evidence_mode": "markov_transition",
            "evidence_application": [{"signal_type": "x", "applied": True, "effective_scalar": 1.0}],
        }
        assert classify_trace(trace).backend_path == "markov"

    def test_backend_path_falls_back_to_plane(self):
        trace = {
            "trace_id": "t",
            "evidence_mode": "live",
            "evidence_application": [{"signal_type": "x", "applied": True, "raw_factor": 1.0}],
        }
        assert classify_trace(trace).backend_path == "plane"


# ---------------------------------------------------------------------------
# build_report — aggregation + gate
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_status_counts(self):
        report = build_report(
            [_enriched_prop_trace(), _enriched_markov_trace(), _legacy_trace(), _no_evidence_trace()]
        )
        assert report.total_traces == 4
        assert report.sufficient == 2
        assert report.insufficient == 1
        assert report.no_evidence == 1
        assert report.insufficient_trace_ids == ("t-old",)

    def test_legacy_signals_excluded_from_summaries(self):
        report = build_report([_enriched_prop_trace(), _legacy_trace()])
        signal_types = {s.signal_type for s in report.signal_summaries}
        assert "usage_spike" in signal_types
        # recent_form lives only on the insufficient trace -> never aggregated.
        assert "recent_form" not in signal_types

    def test_per_signal_dimension_counts(self):
        report = build_report([_enriched_prop_trace(), _enriched_markov_trace()])
        by_type = {s.signal_type: s for s in report.signal_summaries}

        usage = by_type["usage_spike"]
        assert usage.present == 1
        assert usage.applied == 1
        assert usage.outcome_resolved_applied == 1
        assert usage.calibration_eligible_applied == 1
        assert usage.by_backend_path == {"fast_score": 1}
        assert usage.by_market_type == {"prop": 1}

        rest = by_type["rest_advantage"]
        assert rest.applied == 1
        assert rest.outcome_resolved_applied == 1
        # markov trace was not calibration-eligible
        assert rest.calibration_eligible_applied == 0
        assert rest.by_backend_path == {"markov_state_v1": 1}

    def test_unapplied_signal_counts_present_not_applied(self):
        trace = _enriched_prop_trace()
        trace["evidence_application"][0]["applied"] = False
        report = build_report([trace])
        summary = report.signal_summaries[0]
        assert summary.present == 1
        assert summary.applied == 0
        assert summary.outcome_resolved_applied == 0


def test_render_report_markdown_labels_insufficient():
    report = build_report(
        [_enriched_prop_trace(), _legacy_trace(), _no_evidence_trace()]
    )
    md = render_report_markdown(report)
    assert "# Qualitative Signal Feedback" in md
    assert "Insufficient" in md
    assert "t-old" in md  # the labeled, excluded trace id is surfaced
    assert "usage_spike" in md
