"""
Calibration/grading must be decoupled from wager logging (2026-05-29).

These tests pin the architectural contract that the calibration plane evaluates
*model-issued candidates* (any persisted trace with model predictions, a valid
trace_quality flag set, and an attached outcome) — never human-confirmed bets.

Specifically:
- a candidate with a trace_id and an outcome is graded/calibration-eligible
  without any bet_record;
- a static/fallback report path (no fitted profile) still surfaces unlogged
  graded candidates;
- a missing bet_record does not flip a valid prediction to ineligible;
- bet_taken (record_bet) only adds wager-tracking metadata and changes nothing
  about calibration eligibility or grading;
- a missing closing line (no CLV) does not block ordinary grading/calibration.
"""

from __future__ import annotations

import tempfile
from typing import Any

from omega.ops.output_modes import OutputMode, classify_output_mode
from omega.trace.bet_record import BetRecord
from omega.trace.store import TraceStore


def _tmp_store() -> TraceStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return TraceStore(db_path=tmp.name)


def _candidate_trace(trace_id: str = "sandbox-cand-001") -> dict[str, Any]:
    """A model-issued game candidate: predictions present, calibration-eligible,
    NO bet_record anywhere. This is the unit the calibration plane must evaluate."""
    return {
        "trace_id": trace_id,
        "run_id": "r-cand",
        "timestamp": "2026-05-28T18:00:00Z",
        "prompt": "Celtics vs Lakers NBA",
        "league": "NBA",
        "matchup": "Celtics @ Lakers",
        "execution_mode": "native_sim",
        "simulation_seed": 999,
        "aggregate_quality": 0.82,
        "predictions": {"home_win_prob": 0.61, "away_win_prob": 0.39},
        "recommendations": [{"side": "home", "edge_pct": 3.1, "units": 1.0}],
        "odds_snapshot": {"moneyline_home": -140, "moneyline_away": 120},
        "downgrades": [],
        "kind": "game",
        "result": {"status": "success", "context_source": "provided", "baseline_used": False},
        "trace_quality": {
            "calibration_eligible": True,
            "context_source": "provided",
            "baseline_used": False,
            "identity_status": "complete",
            "calibration_exclusion_reasons": [],
        },
    }


class TestCandidateGradedWithoutBet:
    def test_candidate_with_outcome_is_graded_without_bet_record(self):
        store = _tmp_store()
        try:
            store.persist(_candidate_trace())
            store.attach_outcome(trace_id="sandbox-cand-001", home_score=110, away_score=104)

            graded = store.get_graded_traces(league="NBA")
            ids = {t["trace_id"] for t in graded}
            assert "sandbox-cand-001" in ids
            # And it carries zero wager metadata.
            assert store.get_bet_records("sandbox-cand-001") == []
        finally:
            store.close()

    def test_unlogged_candidate_surfaces_in_fallback_query(self):
        """Static/fallback report path uses the same eligibility query; an
        unlogged-but-graded candidate must still be counted."""
        store = _tmp_store()
        try:
            store.persist(_candidate_trace("sandbox-cand-002"))
            store.attach_outcome(trace_id="sandbox-cand-002", home_score=99, away_score=101)

            eligible_graded = store.query_traces(
                league="NBA",
                has_outcome=True,
                calibration_eligible_only=True,
                limit=100,
            )
            assert any(t["trace_id"] == "sandbox-cand-002" for t in eligible_graded)
        finally:
            store.close()

    def test_missing_bet_record_does_not_make_prediction_ineligible(self):
        store = _tmp_store()
        try:
            store.persist(_candidate_trace("sandbox-cand-003"))
            store.attach_outcome(trace_id="sandbox-cand-003", home_score=88, away_score=80)

            eligible = store.query_traces(calibration_eligible_only=True, limit=100)
            assert any(t["trace_id"] == "sandbox-cand-003" for t in eligible)
        finally:
            store.close()

    def test_recording_a_bet_changes_nothing_about_eligibility(self):
        """bet_taken is pure wager-tracking: adding a BetRecord must not change
        whether the candidate is graded or calibration-eligible."""
        store = _tmp_store()
        try:
            store.persist(_candidate_trace("sandbox-cand-004"))
            store.attach_outcome(trace_id="sandbox-cand-004", home_score=120, away_score=118)

            before = {t["trace_id"] for t in store.get_graded_traces(league="NBA")}

            store.record_bet(
                BetRecord(
                    bet_id="bet-004",
                    trace_id="sandbox-cand-004",
                    book="BetMGM",
                    market="moneyline",
                    selection="Boston Celtics",
                    selection_descriptor="home_ml",
                    odds_taken=-140.0,
                    stake_units=1.0,
                    decision_timestamp="2026-05-28T18:30:00Z",
                )
            )

            after = {t["trace_id"] for t in store.get_graded_traces(league="NBA")}
            assert before == after
            assert "sandbox-cand-004" in after
            # The bet exists purely as wager metadata.
            assert len(store.get_bet_records("sandbox-cand-004")) == 1
        finally:
            store.close()

    def test_missing_closing_line_does_not_block_grading(self):
        """No closing_line (so CLV is unavailable) — grading/calibration still works."""
        store = _tmp_store()
        try:
            store.persist(_candidate_trace("sandbox-cand-005"))
            store.attach_outcome(trace_id="sandbox-cand-005", home_score=95, away_score=97)

            graded = store.get_graded_traces(league="NBA")
            assert any(t["trace_id"] == "sandbox-cand-005" for t in graded)
            # No closing line was ever attached; CLV is simply unavailable.
            row = store.conn.execute(
                "SELECT COUNT(*) FROM closing_lines WHERE trace_id = ?",
                ("sandbox-cand-005",),
            ).fetchone()[0]
            assert row == 0
        finally:
            store.close()


class TestOutputModeIgnoresBetLogging:
    def test_actionable_without_any_bet_record(self):
        mode = classify_output_mode(
            calibration_profile="nba-fitted-v3",
            trace_count=40,
            sidecar_valid=True,
        )
        assert mode is OutputMode.ACTIONABLE


class TestDecouplingPreservedAfterLifecycleHardening:
    """Guardrail: the lifecycle-reliability work must not re-couple calibration
    eligibility to bet logging. A valid candidate with an outcome and no
    bet_record stays calibration-eligible and graded."""

    def test_eligibility_independent_of_bet_record(self):
        store = _tmp_store()
        try:
            store.persist(_candidate_trace("sandbox-decouple-1"))
            store.attach_outcome(trace_id="sandbox-decouple-1", home_score=101, away_score=99)
            graded = store.get_graded_traces(league="NBA")
            row = next(t for t in graded if t["trace_id"] == "sandbox-decouple-1")
            assert row["trace_quality"]["calibration_eligible"] is True
            assert store.get_bet_records("sandbox-decouple-1") == []
        finally:
            store.close()
