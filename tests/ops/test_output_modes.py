"""Tests for omega.ops.output_modes."""

from __future__ import annotations

from omega.ops.output_modes import (
    MAX_ECE_FOR_ACTIONABLE,
    MIN_SAMPLES_FOR_ACTIONABLE,
    RESEARCH_CANDIDATE_DISCLAIMER,
    RESEARCH_CANDIDATE_HEADER,
    RESEARCH_PLUS_DISCLAIMER,
    RESEARCH_PLUS_HEADER,
    OutputMode,
    cap_stake_for_research,
    cap_stake_for_research_plus,
    classify_market_output_mode,
    classify_output_mode,
    contains_blocked_phrase,
    contains_blocked_phrase_research_plus,
    format_research_candidate_block,
    format_research_plus_block,
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

    def test_under_sampled_profile_is_research_plus(self):
        # A real-but-thin profile now SHOWS its numbers under research+ guardrails
        # (capped stake) instead of being fully suppressed.
        mode, reasons = self._good(sample_size=MIN_SAMPLES_FOR_ACTIONABLE - 1)
        assert mode is OutputMode.RESEARCH_PLUS
        assert any("sample_size" in r for r in reasons)

    def test_poor_ece_profile_is_research_plus(self):
        mode, reasons = self._good(calibration_error=MAX_ECE_FOR_ACTIONABLE + 0.01)
        assert mode is OutputMode.RESEARCH_PLUS
        assert any("ECE" in r for r in reasons)

    def test_missing_ece_is_research_plus(self):
        mode, reasons = self._good(calibration_error=None)
        assert mode is OutputMode.RESEARCH_PLUS
        assert any("calibration_error missing" in r for r in reasons)

    def test_zero_coverage_is_research(self):
        mode, reasons = self._good(trace_count=0)
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("0 calibration-eligible" in r for r in reasons)

    def test_invalid_sidecar_is_research(self):
        mode, reasons = self._good(sidecar_valid=False)
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("sidecar" in r.lower() for r in reasons)

    def test_real_nba_prop_profile_is_research_plus(self):
        # iso_nba_prop_v1_7c8018680da72efe: production but n=48, ECE=0.2876 —
        # fails BOTH floor checks. It is a real profile, so research+ surfaces its
        # numbers (with the failing ECE in the honesty block) under a capped stake
        # rather than hiding them; it never unlocks full ACTIONABLE output.
        mode, reasons = classify_market_output_mode(
            profile_id="iso_nba_prop_v1_7c8018680da72efe",
            sample_size=48,
            calibration_error=0.287589,
            trace_count=12,
            sidecar_valid=True,
            maturity="production",
        )
        assert mode is OutputMode.RESEARCH_PLUS
        assert any("sample_size" in r for r in reasons)
        assert any("ECE" in r for r in reasons)


class TestMaturityTiers:
    """Maturity drives the split: none/retired hide; provisional/probation show."""

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

    def test_production_maturity_clears_floor_is_actionable(self):
        mode, reasons = self._good(maturity="production")
        assert mode is OutputMode.ACTIONABLE
        assert reasons == []

    def test_provisional_maturity_is_research_plus(self):
        mode, reasons = self._good(maturity="provisional")
        assert mode is OutputMode.RESEARCH_PLUS
        assert any("provisional" in r for r in reasons)

    def test_probation_maturity_is_research_plus(self):
        mode, _ = self._good(maturity="probation")
        assert mode is OutputMode.RESEARCH_PLUS

    def test_none_maturity_is_research_candidate(self):
        # 'none' maturity = not trusted to apply -> genuinely uncalibrated, hidden.
        mode, reasons = self._good(maturity="none")
        assert mode is OutputMode.RESEARCH_CANDIDATE
        assert any("not trusted to apply" in r for r in reasons)

    def test_retired_maturity_is_research_candidate(self):
        mode, _ = self._good(maturity="retired")
        assert mode is OutputMode.RESEARCH_CANDIDATE


class TestCapStakeForResearchPlus:
    def test_provisional_capped_half_unit(self):
        assert cap_stake_for_research_plus(3.0, "provisional") == 0.5

    def test_probation_capped_one_unit(self):
        assert cap_stake_for_research_plus(3.0, "probation") == 1.0

    def test_unknown_maturity_uses_default_ceiling(self):
        assert cap_stake_for_research_plus(3.0, "production") == 0.5
        assert cap_stake_for_research_plus(3.0, None) == 0.5

    def test_below_ceiling_unchanged(self):
        assert cap_stake_for_research_plus(0.25, "probation") == 0.25


class TestResearchPlusBlockedPhrases:
    def test_hype_blocked_but_tier_allowed(self):
        assert "best bet" in contains_blocked_phrase_research_plus("our best bet tonight")
        # Tier labels are permitted in research+ (confidence is shown, capped <= B).
        assert contains_blocked_phrase_research_plus("Confidence: Tier B") == []

    def test_clean_text_passes(self):
        assert contains_blocked_phrase_research_plus("edge +3.2%, EV +5.1%") == []

    def test_matching_is_case_insensitive(self):
        # Casing must not let an overclaiming phrase slip past the guardrail.
        assert "best bet" in contains_blocked_phrase_research_plus("Our BEST BET tonight")
        assert "engine-confirmed" in contains_blocked_phrase_research_plus("ENGINE-CONFIRMED edge")


class TestFormatResearchPlusBlock:
    def test_header_and_disclaimer_present(self):
        result = format_research_plus_block("Edge +3.2%.")
        assert RESEARCH_PLUS_HEADER in result
        assert RESEARCH_PLUS_DISCLAIMER in result


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

    def test_matching_is_case_insensitive(self):
        assert "best bet" in contains_blocked_phrase("This is our BEST BET tonight.")
        assert "tier a" in contains_blocked_phrase("Graded Tier A play")


class TestFormatResearchCandidateBlock:
    def test_header_present(self):
        result = format_research_candidate_block("Some analysis.")
        assert RESEARCH_CANDIDATE_HEADER in result

    def test_disclaimer_present(self):
        result = format_research_candidate_block("Some analysis.")
        assert RESEARCH_CANDIDATE_DISCLAIMER in result
