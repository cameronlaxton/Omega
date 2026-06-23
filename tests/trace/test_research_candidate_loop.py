"""
Integration test for the cold-start calibration loop (W-B).

Ensures that:
1. In RESEARCH_CANDIDATE output mode, analyze() can still run successfully,
   minting a `sandbox-` trace_id and populating calibration-relevant data.
2. The trace is successfully persisted to the DB using the PersistableTrace adapter.
3. User-facing betting numbers (edge%, EV%, Kelly, units, tier, trace_id)
   can be withheld by the presentation layer (the agent) based on OutputMode.
4. After outcome grading is attached, the trace is calibration-eligible
   (decoupled from bet logging or closing lines).
"""

from __future__ import annotations

import tempfile

from omega.core.contracts.service import analyze
from omega.ops.output_modes import OutputMode, classify_output_mode, contains_blocked_phrase
from omega.trace.persistable import PersistableTrace
from omega.trace.store import TraceStore


def _tmp_store() -> TraceStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return TraceStore(db_path=tmp.name)


class TestResearchCandidateLoop:
    def test_research_candidate_loop_eligibility_and_withholding(self):
        # 1. Determine output mode - no profile fitted yet -> RESEARCH_CANDIDATE
        mode = classify_output_mode(
            calibration_profile=None,  # static fallback
            trace_count=0,
            sidecar_valid=True,
        )
        assert mode == OutputMode.RESEARCH_CANDIDATE

        # 2. Call analyze() - engine available, it must still run, mint sandbox- id
        trace_envelope = analyze(
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
            session_id="sess-20260531-test-loop",
            bankroll=1000.0,
        )

        assert trace_envelope["trace_id"].startswith("sandbox-")
        assert trace_envelope["result"]["status"] == "success"

        # 3. Honest output in RESEARCH_CANDIDATE mode.
        # The engine still runs and produces internal edge data (which the agent
        # withholds), but with no fitted profile (static_identity) and no
        # structured evidence it must NOT manufacture an actionable bet: edges
        # are capped (never A) and there is no headline best_bet.
        result = trace_envelope["result"]
        assert result["edges"], "engine should still produce internal edge data"
        tiers = {e["confidence_tier"] for e in result["edges"]}
        assert "A" not in tiers, "uncalibrated, evidence-free trace must not show Tier A"
        assert result["best_bet"] is None, "no actionable headline bet without calibration"
        # The honesty signals are present and real.
        tq = trace_envelope["trace_quality"]
        assert isinstance(tq["aggregate_quality"], int)
        assert tq["confidence_cap"] in ("B", "C", "Pass")
        assert tq["calibration_path"] == "static_identity"

        # But the agent must NOT show these to the user.
        # We check the contains_blocked_phrase helper to audit that blocked terms can be scanned:
        sample_blocked_text = "Here is the best bet with Tier A confidence from the engine."
        assert len(contains_blocked_phrase(sample_blocked_text)) > 0

        # 4. Persistence and Calibration Eligibility
        store = _tmp_store()
        try:
            # Adapt the trace using the canonical PersistableTrace adapter
            persistable = PersistableTrace.from_analyze_output(trace_envelope)

            # Persist trace
            store.persist(persistable)

            # Grade outcome
            store.attach_outcome(
                trace_id=trace_envelope["trace_id"],
                home_score=110,
                away_score=100,
            )

            # Query graded traces and assert it's eligible
            graded_traces = store.get_graded_traces(league="NBA")
            matching = [t for t in graded_traces if t["trace_id"] == trace_envelope["trace_id"]]
            assert len(matching) == 1
            graded_trace = matching[0]

            # The trace must have correct quality attributes
            assert graded_trace["trace_quality"]["calibration_eligible"] is True
            assert graded_trace["trace_quality"]["context_source"] == "provided"
            assert len(graded_trace["trace_quality"]["calibration_exclusion_reasons"]) == 0

        finally:
            store.close()
