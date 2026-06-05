"""Tests for omega.ops.output_modes."""

from __future__ import annotations

from omega.ops.output_modes import (
    MAX_ECE_FOR_ACTIONABLE,
    MIN_SAMPLES_FOR_ACTIONABLE,
    RESEARCH_CANDIDATE_DISCLAIMER,
    RESEARCH_CANDIDATE_HEADER,
    OutputMode,
    cap_stake_for_research,
    classify_market_output_mode,
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


class TestClassifyMarketOutputMode:
    """Per-market authorization with the calibration-quality floor."""

    def _good(self, **overrides):
        kwargs = dict(
            profile_id="iso_good",
            sample_size=MIN_SAMPLES_FOR_ACTIONABLE + 50,
            calibration_error=MAX_ECE_FOR_ACTIONABLE / 2,
            trace_count=20,
            sidecar_valid=True,
        )
        kwargs.update(overrides)
        return classify_market_output_mode(**kwargs)

    def test_profile_clears_floor_is_actionable(self):
        mode, reasons = self._good()
        assert mode is OutputMode.ACTIONABLE
        assert reasons == []

    def test_no_profile_is_research(self):
        mode, reasons = self._good(profile_id=None, sample_size=None, calibration_error=None)
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("No fitted calibration profile" in r for r in reasons)

    def test_under_sampled_profile_is_research(self):
        mode, reasons = self._good(sample_size=MIN_SAMPLES_FOR_ACTIONABLE - 1)
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("sample_size" in r for r in reasons)

    def test_poor_ece_profile_is_research(self):
        mode, reasons = self._good(calibration_error=MAX_ECE_FOR_ACTIONABLE + 0.01)
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("ECE" in r for r in reasons)

    def test_missing_ece_is_research(self):
        mode, reasons = self._good(calibration_error=None)
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("calibration_error missing" in r for r in reasons)

    def test_zero_coverage_is_research(self):
        mode, reasons = self._good(trace_count=0)
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("0 calibration-eligible" in r for r in reasons)

    def test_invalid_sidecar_is_research(self):
        mode, reasons = self._good(sidecar_valid=False)
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("sidecar" in r.lower() for r in reasons)

    def test_real_nba_prop_profile_stays_research(self):
        # iso_nba_prop_v1_7c8018680da72efe: production but n=48, ECE=0.2876 —
        # fails BOTH floor checks, so props must NOT unlock formal output.
        mode, reasons = classify_market_output_mode(
            profile_id="iso_nba_prop_v1_7c8018680da72efe",
            sample_size=48,
            calibration_error=0.287589,
            trace_count=12,
            sidecar_valid=True,
        )
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("sample_size" in r for r in reasons)
        assert any("ECE" in r for r in reasons)


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
