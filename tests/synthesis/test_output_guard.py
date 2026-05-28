"""Tests for omega.synthesis.output_guard — Research Candidate classification."""

from __future__ import annotations

from omega.synthesis.output_guard import (
    OutputMode,
    cap_stake_for_research,
    classify_output_mode,
    contains_blocked_phrase,
    format_research_candidate_block,
    RESEARCH_CANDIDATE_DISCLAIMER,
    RESEARCH_CANDIDATE_HEADER,
)


class TestClassifyOutputMode:
    def test_static_fallback_is_research_candidate(self):
        mode = classify_output_mode(
            calibration_profile=None,  # static fallback
            trace_count=5,
            sidecar_valid=True,
            has_bet_record=True,
        )
        assert mode == OutputMode.RESEARCH_CANDIDATE

    def test_zero_traces_is_research_candidate(self):
        mode = classify_output_mode(
            calibration_profile="v2",
            trace_count=0,
            sidecar_valid=True,
            has_bet_record=True,
        )
        assert mode == OutputMode.RESEARCH_CANDIDATE

    def test_invalid_sidecar_is_research_candidate(self):
        mode = classify_output_mode(
            calibration_profile="v2",
            trace_count=3,
            sidecar_valid=False,
            has_bet_record=True,
        )
        assert mode == OutputMode.RESEARCH_CANDIDATE

    def test_no_bet_record_is_research_candidate(self):
        mode = classify_output_mode(
            calibration_profile="v2",
            trace_count=3,
            sidecar_valid=True,
            has_bet_record=False,
        )
        assert mode == OutputMode.RESEARCH_CANDIDATE

    def test_all_conditions_met_is_actionable(self):
        mode = classify_output_mode(
            calibration_profile="v2",
            trace_count=3,
            sidecar_valid=True,
            has_bet_record=True,
        )
        assert mode == OutputMode.ACTIONABLE

    def test_missing_bet_record_cannot_be_labeled_actionable(self):
        """Explicit contract: no bet_record → never actionable."""
        mode = classify_output_mode(
            calibration_profile="v2-fitted",
            trace_count=10,
            sidecar_valid=True,
            has_bet_record=False,
        )
        assert mode is not OutputMode.ACTIONABLE


class TestCapStakeForResearch:
    def test_above_1u_capped(self):
        assert cap_stake_for_research(3.0) == 1.0

    def test_exactly_1u_unchanged(self):
        assert cap_stake_for_research(1.0) == 1.0

    def test_below_1u_unchanged(self):
        assert cap_stake_for_research(0.5) == 0.5

    def test_zero_unchanged(self):
        assert cap_stake_for_research(0.0) == 0.0

    def test_static_fallback_stake_never_exceeds_1u(self):
        """Integration: static fallback must produce ≤ 1u."""
        from omega.synthesis.staking import calculate_stake
        units, _ = calculate_stake(bankroll=1000.0, unit_pct=0.01, is_static_fallback=True)
        capped = cap_stake_for_research(units)
        assert capped <= 1.0


class TestContainsBlockedPhrase:
    def test_best_bet_phrase_blocked(self):
        found = contains_blocked_phrase("This is our best bet for tonight.")
        assert "best bet" in found

    def test_tier_a_blocked(self):
        found = contains_blocked_phrase("Tier A confidence on this play.")
        assert "Tier A" in found

    def test_tier_b_blocked(self):
        found = contains_blocked_phrase("Tier B rating for this market.")
        assert "Tier B" in found

    def test_engine_confirmed_blocked(self):
        found = contains_blocked_phrase("engine-confirmed edge of 8%.")
        assert "engine-confirmed" in found

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

    def test_content_preserved(self):
        result = format_research_candidate_block("OKC at SA — qualitative lean.")
        assert "OKC at SA" in result
