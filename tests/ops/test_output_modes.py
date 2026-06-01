"""Tests for omega.ops.output_modes."""

from __future__ import annotations

from omega.ops.output_modes import (
    RESEARCH_CANDIDATE_DISCLAIMER,
    RESEARCH_CANDIDATE_HEADER,
    OutputMode,
    cap_stake_for_research,
    classify_output_mode,
    contains_blocked_phrase,
    format_research_candidate_block,
)


class TestClassifyOutputMode:
    def test_static_fallback_is_research_candidate(self):
        mode = classify_output_mode(calibration_profile=None, trace_count=5, sidecar_valid=True)
        assert mode == OutputMode.RESEARCH_CANDIDATE

    def test_zero_traces_is_research_candidate(self):
        mode = classify_output_mode(calibration_profile="v2", trace_count=0, sidecar_valid=True)
        assert mode == OutputMode.RESEARCH_CANDIDATE

    def test_invalid_sidecar_is_research_candidate(self):
        mode = classify_output_mode(calibration_profile="v2", trace_count=3, sidecar_valid=False)
        assert mode == OutputMode.RESEARCH_CANDIDATE

    def test_all_conditions_met_is_actionable(self):
        mode = classify_output_mode(calibration_profile="v2", trace_count=3, sidecar_valid=True)
        assert mode == OutputMode.ACTIONABLE


class TestCapStakeForResearch:
    def test_above_1u_capped(self):
        assert cap_stake_for_research(3.0) == 1.0

    def test_exactly_1u_unchanged(self):
        assert cap_stake_for_research(1.0) == 1.0

    def test_below_1u_unchanged(self):
        assert cap_stake_for_research(0.5) == 0.5

    def test_zero_unchanged(self):
        assert cap_stake_for_research(0.0) == 0.0


class TestContainsBlockedPhrase:
    def test_best_bet_phrase_blocked(self):
        found = contains_blocked_phrase("This is our best bet for tonight.")
        assert "best bet" in found

    def test_clean_research_text_passes(self):
        found = contains_blocked_phrase(
            "Matchup analysis: OKC at San Antonio. Wembanyama averages 27.5 pts/game."
        )
        assert found == []

    def test_research_candidate_header_is_clean(self):
        found = contains_blocked_phrase(RESEARCH_CANDIDATE_HEADER)
        assert found == []


class TestFormatResearchCandidateBlock:
    def test_header_present(self):
        result = format_research_candidate_block("Some analysis.")
        assert RESEARCH_CANDIDATE_HEADER in result

    def test_disclaimer_present(self):
        result = format_research_candidate_block("Some analysis.")
        assert RESEARCH_CANDIDATE_DISCLAIMER in result
