"""Tests for omega.trace.eligibility — the single source of truth for
calibration/evidence eligibility.

Locks two invariants that have caused real defects:
  1. Empty evidence does NOT block probability calibration (it only blocks
     evidence-signal learning).
  2. A failed trace-scoped QA verdict makes a trace calibration-ineligible.

Also a characterization test proving the simplified query_traces SQL filter
(gate on the canonical calibration_eligible flag) returns the SAME set as the
old multi-clause filter for traces produced by the real analyze() path.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.core.contracts.service import analyze  # noqa: E402
from omega.trace.eligibility import (  # noqa: E402
    EV_ELIGIBLE_ORIGINAL,
    EV_ELIGIBLE_RECOVERED,
    EV_INELIGIBLE_EMPTY,
    EV_INELIGIBLE_QA_FAILED,
    REASON_BASELINE_CONTEXT,
    REASON_ENGINE_SKIPPED,
    REASON_LEGACY_MISSING_IDENTITY,
    REASON_QA_FAILED,
    STATUS_ELIGIBLE,
    STATUS_INELIGIBLE_MISSING_PREDICTION,
    STATUS_INELIGIBLE_QA_FAILED,
    STATUS_INELIGIBLE_TRACE_QUALITY,
    calibration_exclusion_reasons,
    evidence_learning_eligibility,
    probability_calibration_eligibility,
)
from omega.trace.persistable import PersistableTrace  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

_CLEAN_TQ = {
    "calibration_eligible": True,
    "context_source": "provided",
    "identity_status": "complete",
    "evidence_status": "empty",
}


class TestProbabilityCalibration:
    def test_empty_evidence_is_still_calibration_eligible(self):
        # The headline invariant: evidence_status="empty" must NOT block calibration.
        result = probability_calibration_eligibility(
            predictions={"home_win_prob": 58.0}, trace_quality=dict(_CLEAN_TQ)
        )
        assert result.eligible is True
        assert result.status == STATUS_ELIGIBLE

    def test_missing_predictions_is_ineligible(self):
        result = probability_calibration_eligibility(
            predictions=None, trace_quality=dict(_CLEAN_TQ)
        )
        assert result.eligible is False
        assert result.status == STATUS_INELIGIBLE_MISSING_PREDICTION

    def test_failed_qa_verdict_is_ineligible(self):
        result = probability_calibration_eligibility(
            predictions={"home_win_prob": 58.0},
            trace_quality=dict(_CLEAN_TQ),
            qa_verdict="fail",
        )
        assert result.eligible is False
        assert result.status == STATUS_INELIGIBLE_QA_FAILED

    def test_flag_false_is_ineligible(self):
        tq = {**_CLEAN_TQ, "calibration_eligible": False}
        result = probability_calibration_eligibility(
            predictions={"home_win_prob": 58.0}, trace_quality=tq
        )
        assert result.eligible is False
        assert result.status == STATUS_INELIGIBLE_TRACE_QUALITY


class TestExclusionReasons:
    def test_clean_inputs_have_no_reasons(self):
        assert calibration_exclusion_reasons(
            result_status="success",
            context_source="provided",
            baseline_used=False,
            identity_status="complete",
        ) == []

    def test_reason_strings_are_preserved(self):
        reasons = calibration_exclusion_reasons(
            result_status="skipped",
            context_source="league_default",
            baseline_used=True,
            identity_status="missing",
        )
        assert REASON_ENGINE_SKIPPED in reasons
        assert REASON_BASELINE_CONTEXT in reasons
        assert REASON_LEGACY_MISSING_IDENTITY in reasons
        # sorted + deduped
        assert reasons == sorted(set(reasons))

    def test_qa_fail_appends_qa_reason(self):
        reasons = calibration_exclusion_reasons(
            result_status="success",
            context_source="provided",
            baseline_used=False,
            identity_status="complete",
            qa_verdict="fail",
        )
        assert reasons == [REASON_QA_FAILED]


class TestEvidenceLearning:
    def test_present_evidence_is_eligible_original(self):
        result = evidence_learning_eligibility(trace_quality={"evidence_status": "present"})
        assert result.eligible is True
        assert result.status == EV_ELIGIBLE_ORIGINAL

    def test_recovered_evidence_is_counted_separately(self):
        result = evidence_learning_eligibility(
            trace_quality={"evidence_status": "recovered_predecision"}
        )
        assert result.eligible is True
        assert result.status == EV_ELIGIBLE_RECOVERED

    def test_empty_evidence_is_learning_ineligible(self):
        result = evidence_learning_eligibility(trace_quality={"evidence_status": "empty"})
        assert result.eligible is False
        assert result.status == EV_INELIGIBLE_EMPTY

    def test_qa_fail_blocks_evidence_learning(self):
        result = evidence_learning_eligibility(
            trace_quality={"evidence_status": "present"}, qa_verdict="fail"
        )
        assert result.eligible is False
        assert result.status == EV_INELIGIBLE_QA_FAILED


class TestPersistableDelegates:
    def test_empty_evidence_trace_is_probability_gradeable(self):
        # End-to-end: a real analyze() trace with no evidence is still
        # probability-calibration eligible via the diagnostic.
        out = analyze(
            {
                "home_team": "Boston Celtics",
                "away_team": "Indiana Pacers",
                "league": "NBA",
                "n_iterations": 100,
                "seed": 42,
                "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
                "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
                "game_context": {"is_playoff": False, "rest_days": 2},
                "odds": {"moneyline_home": -150, "moneyline_away": 130},
            },
            session_id="sess-elig-pers",
            bankroll=2500.0,
        )
        trace = PersistableTrace.from_analyze_output(out)
        assert trace.trace_quality["evidence_status"] == "empty"
        elig = trace.calibration_eligibility()
        assert elig["probability_calibration"] is True
        assert elig["evidence_scoring"] is False


def _tmp_store() -> TraceStore:
    fd, path = tempfile.mkstemp(suffix=".db")
    import os

    os.close(fd)
    return TraceStore(db_path=path)


# Old (pre-centralization) calibration-eligible WHERE clauses, kept here only to
# prove the simplified filter is equivalent on real data.
_OLD_ELIGIBLE_SQL = """
    SELECT t.trace_id FROM traces t
    WHERE t.predictions IS NOT NULL
      AND json_extract(t.full_trace, '$.result.status') = 'success'
      AND json_extract(t.full_trace, '$.trace_quality.calibration_eligible') = 1
      AND json_extract(t.full_trace, '$.trace_quality.context_source') = 'provided'
      AND json_extract(t.full_trace, '$.trace_quality.identity_status') = 'complete'
"""


class TestQueryFilterCharacterization:
    def test_simplified_filter_matches_old_filter_on_real_traces(self):
        store = _tmp_store()

        # Eligible: provided context.
        elig = analyze(
            {
                "home_team": "Boston Celtics",
                "away_team": "Indiana Pacers",
                "league": "NBA",
                "n_iterations": 100,
                "seed": 7,
                "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
                "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
                "game_context": {"is_playoff": False, "rest_days": 2},
                "odds": {"moneyline_home": -150, "moneyline_away": 130},
            },
            session_id="sess-char-elig",
            bankroll=2500.0,
        )
        # Ineligible: baseline default context.
        baseline = analyze(
            {
                "home_team": "Boston Celtics",
                "away_team": "Indiana Pacers",
                "league": "NBA",
                "n_iterations": 100,
                "seed": 9,
                "allow_baseline": True,
                "game_context": {"is_playoff": False, "rest_days": 2},
                "odds": {"moneyline_home": -150, "moneyline_away": 130},
            },
            session_id="sess-char-baseline",
            bankroll=2500.0,
        )
        for out in (elig, baseline):
            rec = PersistableTrace.from_analyze_output(out).to_store_record()
            store.persist(rec)
            store.attach_outcome(rec["trace_id"], home_score=110, away_score=104)

        new_ids = {t["trace_id"] for t in store.query_traces(calibration_eligible_only=True, limit=1000)}
        old_ids = {row[0] for row in store.conn.execute(_OLD_ELIGIBLE_SQL).fetchall()}
        assert new_ids == old_ids
        # And the eligible trace is actually present.
        assert elig["trace_id"] in new_ids
        assert baseline["trace_id"] not in new_ids
        store.close()
