"""Phase 0 contract + adapter tests for the decision-support presentation layer.

Verification-plan coverage (design §17, contract tests):
- mode defaults and invalid values fail closed;
- EventIdentityV1 validation + event_key derivation;
- OutcomeCase / DecisionSupportPresentationV1 structural + language validation;
- trace v1/v2 dual-read through PersistableTrace;
- the adapter's symmetric-outcome, output-mode-intersection, legacy-compat,
  grouping, ordering, and denied-key guarantees.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from omega.core.contracts.schemas import (
    DecisionSupportPresentationV1,
    EventIdentityV1,
    OutcomeCase,
    coerce_engine_auto_ledger_mode,
    coerce_presentation_mode,
)
from omega.trace.decision_support import (
    DENYLIST_KEYS,
    DecisionSupportViolation,
    assert_no_denied_keys,
    build_market_view,
    build_matchup_brief,
    group_key_for_trace,
    group_traces_into_briefs,
)
from omega.trace.persistable import PersistableTrace

EVENT_IDENTITY = {
    "schema_version": 1,
    "provider": "the-odds-api",
    "provider_event_id": "ev-001",
    "event_key": "MLB::the-odds-api::ev-001",
    "league": "MLB",
    "home_team": "Yankees",
    "away_team": "Red Sox",
    "game_date": "2026-07-16",
}

PRODUCTION_AUDIT = [
    {
        "profile_id": "iso_mlb_game_v8",
        "profile_maturity": "production",
        "sample_size": 400,
        "ece": 0.03,
    }
]


def _game_trace(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "trace_id": "t-game",
        "kind": "game",
        "league": "MLB",
        "matchup": "Red Sox @ Yankees",
        "input_snapshot": {
            "league": "MLB",
            "home_team": "Yankees",
            "away_team": "Red Sox",
        },
        "result": {
            "status": "success",
            "simulation": {"home_win_prob": 58.0, "away_win_prob": 42.0, "draw_prob": None},
            "edges": [
                {
                    "side": "home",
                    "team": "Yankees",
                    "market": "moneyline",
                    "calibrated_prob": 0.58,
                    "market_implied": 0.55,
                    "edge_pct": 3.0,
                    "ev_pct": 5.0,
                    "confidence_tier": "B",
                    "recommended_units": 1.0,
                    "market_odds": -120,
                },
                {
                    "side": "away",
                    "team": "Red Sox",
                    "market": "moneyline",
                    "calibrated_prob": 0.42,
                    "market_implied": 0.45,
                    "edge_pct": -3.0,
                    "ev_pct": -5.0,
                    "confidence_tier": "Pass",
                    "recommended_units": 0.0,
                    "market_odds": 100,
                },
            ],
            "best_bet": {
                "selection": "Yankees ML",
                "odds": -120,
                "edge_pct": 3.0,
                "ev_pct": 5.0,
                "confidence_tier": "B",
                "recommended_units": 1.0,
                "kelly_fraction": 0.02,
            },
            "simulation_distributions": [
                {
                    "target": "home_score",
                    "distribution_type": "poisson",
                    "sample_mean": 4.8,
                    "sample_std": 2.1,
                    "p10": 2.0,
                    "p50": 5.0,
                    "p90": 8.0,
                    "n_iterations": 10000,
                }
            ],
        },
        "calibration_audit": list(PRODUCTION_AUDIT),
        "odds_snapshot": {"moneyline_home": -120, "moneyline_away": 100},
        "event_identity": dict(EVENT_IDENTITY),
        "presentation_mode": "decision_support",
    }
    base.update(overrides)
    return base


def _prop_trace(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "trace_id": "t-prop",
        "kind": "prop",
        "league": "MLB",
        "matchup": "Red Sox @ Yankees",
        "input_snapshot": {
            "league": "MLB",
            "player_name": "Aaron Judge",
            "prop_type": "hits",
            "line": 1.5,
            "game_date": "2026-07-16",
        },
        "result": {
            "status": "success",
            "over_prob": 0.41,
            "under_prob": 0.59,
            "recommendation": "under",
            "confidence_tier": "C",
            "edge_over": -2.0,
            "edge_under": 4.0,
        },
        "calibration_audit": list(PRODUCTION_AUDIT),
        "event_identity": dict(EVENT_IDENTITY),
    }
    base.update(overrides)
    return base


# -- mode coercion -------------------------------------------------------------


class TestModeCoercion:
    @pytest.mark.parametrize("bad", [None, "", "BOGUS", 3, {"x": 1}, "Decision_Support"])
    def test_presentation_mode_fails_closed(self, bad):
        assert coerce_presentation_mode(bad) == "decision_support"

    @pytest.mark.parametrize("bad", [None, "", "on", 1, [], "Shadow"])
    def test_ledger_mode_fails_closed(self, bad):
        assert coerce_engine_auto_ledger_mode(bad) == "disabled"

    def test_valid_values_pass_through(self):
        assert coerce_presentation_mode("recommendation_lab") == "recommendation_lab"
        assert coerce_engine_auto_ledger_mode("shadow") == "shadow"


# -- EventIdentityV1 -----------------------------------------------------------


class TestEventIdentity:
    def test_derive_event_key_is_stable(self):
        key = EventIdentityV1.derive_event_key("mlb ", "the-odds-api", " ev-001 ")
        assert key == "MLB::the-odds-api::ev-001"

    def test_rejects_empty_fields(self):
        with pytest.raises(Exception):
            EventIdentityV1(**{**EVENT_IDENTITY, "provider_event_id": ""})

    def test_rejects_extra_keys(self):
        with pytest.raises(Exception):
            EventIdentityV1(**{**EVENT_IDENTITY, "edge_pct": 3.0})


# -- decision-support presentation contract ------------------------------------


class TestDecisionSupportPresentation:
    def test_complete_case_requires_both_sides(self):
        with pytest.raises(ValueError, match="complete"):
            OutcomeCase(
                market_key="moneyline",
                outcome_key="home",
                label="Yankees win",
                supporting=["rotation edge"],
                challenging=[],
                data_status="complete",
            )

    def test_partial_case_may_be_one_sided(self):
        case = OutcomeCase(
            market_key="moneyline",
            outcome_key="home",
            label="Yankees win",
            supporting=["rotation edge"],
            challenging=[],
            data_status="partial",
        )
        assert case.data_status == "partial"

    def test_duplicate_outcome_keys_rejected(self):
        case = dict(
            market_key="moneyline",
            outcome_key="home",
            label="Yankees win",
            supporting=["a"],
            challenging=["b"],
            data_status="complete",
        )
        with pytest.raises(ValueError, match="Duplicate"):
            DecisionSupportPresentationV1(
                matchup_summary="s",
                market_context="c",
                outcome_cases=[OutcomeCase(**case), OutcomeCase(**case)],
            )

    @pytest.mark.parametrize(
        "text", ["Our best bet tonight", "This is a LOCK", "smash the over", "Tier A play"]
    )
    def test_blocked_language_rejected(self, text):
        with pytest.raises(ValueError, match="blocked language"):
            DecisionSupportPresentationV1(matchup_summary=text, market_context="x")

    def test_blocked_language_in_case_prose_rejected(self):
        with pytest.raises(ValueError, match="blocked language"):
            OutcomeCase(
                market_key="moneyline",
                outcome_key="home",
                label="Yankees win",
                supporting=["engine-confirmed edge"],
                challenging=["bullpen"],
                data_status="complete",
            )

    def test_extra_keys_rejected(self):
        with pytest.raises(Exception):
            DecisionSupportPresentationV1(
                matchup_summary="s", market_context="c", edge_pct=3.0
            )


# -- PersistableTrace v2 dual-read ----------------------------------------------


class TestPersistableV2:
    def test_v1_analyze_output_reads_with_fail_closed_defaults(self):
        rec = PersistableTrace.from_analyze_output(
            {
                "trace_id": "t1",
                "ran_at": "2026-07-16T00:00:00Z",
                "kind": "game",
                "input_snapshot": {"league": "MLB", "home_team": "A", "away_team": "B"},
                "result": {},
            }
        ).to_store_record()
        assert rec["schema_version"] == 2
        assert rec["presentation_mode"] == "decision_support"
        assert rec["engine_auto_ledger_mode"] == "disabled"
        assert rec["event_identity"] is None
        assert rec["decision_support_presentation"] is None

    def test_v2_round_trip_preserves_stamps(self):
        out = {
            "trace_id": "t2",
            "ran_at": "2026-07-16T00:00:00Z",
            "kind": "game",
            "input_snapshot": {"league": "MLB", "home_team": "A", "away_team": "B"},
            "result": {},
            "presentation_mode": "recommendation_lab",
            "engine_auto_ledger_mode": "shadow",
            "event_identity": dict(EVENT_IDENTITY),
        }
        rec = PersistableTrace.from_analyze_output(out).to_store_record()
        again = PersistableTrace.from_analyze_output(rec).to_store_record()
        assert again["presentation_mode"] == "recommendation_lab"
        assert again["engine_auto_ledger_mode"] == "shadow"
        assert again["event_identity"]["event_key"] == EVENT_IDENTITY["event_key"]

    def test_malformed_identity_and_mode_fail_closed(self):
        rec = PersistableTrace.from_analyze_output(
            {
                "trace_id": "t3",
                "ran_at": "x",
                "kind": "game",
                "input_snapshot": {},
                "result": {},
                "event_identity": {"garbage": True},
                "engine_auto_ledger_mode": "YOLO",
                "presentation_mode": 42,
            }
        ).to_store_record()
        assert rec["event_identity"] is None
        assert rec["engine_auto_ledger_mode"] == "disabled"
        assert rec["presentation_mode"] == "decision_support"


# -- adapter: probability symmetry + output-mode intersection --------------------


def _set_by_market(view) -> dict[str, Any]:
    return {s.market_key: s for s in view.probability_sets}


class TestSymmetricProbabilities:
    def test_authorized_game_set_is_complete_and_identity_ordered(self):
        view = build_market_view(_game_trace())
        probs = _set_by_market(view)["moneyline"]
        assert probs.disclosure == "shown"
        assert [o.outcome_key for o in probs.outcomes] == ["home", "away"]
        assert probs.outcomes[0].model_estimate == 0.58
        assert probs.outcomes[1].model_estimate == 0.42
        assert probs.outcomes[0].market_implied == 0.55
        assert "not a recommendation" in probs.estimate_label

    def test_incomplete_game_set_is_withheld_entirely(self):
        trace = _game_trace()
        trace["result"]["edges"] = trace["result"]["edges"][:1]  # home side only
        probs = _set_by_market(build_market_view(trace))["moneyline"]
        assert probs.disclosure == "withheld"
        assert probs.withheld_reason == "incomplete_outcome_set"
        assert probs.outcomes == []

    def test_three_way_market_requires_draw(self):
        trace = _game_trace()
        trace["result"]["simulation"]["draw_prob"] = 22.0  # 3-way market
        probs = _set_by_market(build_market_view(trace))["moneyline"]
        assert probs.disclosure == "withheld"  # no draw edge present
        trace["result"]["edges"].append(
            {
                "side": "draw",
                "team": "Draw",
                "market": "draw",
                "calibrated_prob": 0.22,
                "market_implied": 0.24,
                "market_odds": 310,
            }
        )
        probs = _set_by_market(build_market_view(trace))["moneyline"]
        assert probs.disclosure == "shown"
        assert [o.outcome_key for o in probs.outcomes] == ["home", "away", "draw"]

    def test_spread_and_total_sets_render_symmetrically(self):
        trace = _game_trace()
        trace["result"]["edges"].extend(
            [
                {
                    "side": "home", "team": "Yankees", "market": "spread", "line": -1.5,
                    "calibrated_prob": 0.44, "market_implied": 0.48, "market_odds": 130,
                },
                {
                    "side": "away", "team": "Red Sox", "market": "spread", "line": 1.5,
                    "calibrated_prob": 0.56, "market_implied": 0.55, "market_odds": -155,
                },
                {
                    "side": "over", "team": "Over 8.5", "market": "total", "line": 8.5,
                    "calibrated_prob": 0.52, "market_implied": 0.5, "market_odds": -110,
                },
                {
                    "side": "under", "team": "Under 8.5", "market": "total", "line": 8.5,
                    "calibrated_prob": 0.48, "market_implied": 0.5, "market_odds": -110,
                },
            ]
        )
        view = build_market_view(trace)
        # Stable market-identity order, never edge-ordered.
        assert [s.market_key for s in view.probability_sets] == [
            "moneyline", "spread", "total",
        ]
        by_market = _set_by_market(view)
        spread = by_market["spread"]
        assert spread.disclosure == "shown"
        assert [o.outcome_key for o in spread.outcomes] == ["home", "away"]
        assert spread.outcomes[0].label == "Yankees -1.5"
        assert spread.outcomes[1].label == "Red Sox +1.5"
        total = by_market["total"]
        assert [o.outcome_key for o in total.outcomes] == ["over", "under"]
        assert total.outcomes[0].model_estimate == 0.52

    def test_partial_spread_market_is_withheld(self):
        trace = _game_trace()
        trace["result"]["edges"].append(
            {
                "side": "home", "team": "Yankees", "market": "spread", "line": -1.5,
                "calibrated_prob": 0.44, "market_implied": 0.48, "market_odds": 130,
            }
        )
        spread = _set_by_market(build_market_view(trace))["spread"]
        assert spread.disclosure == "withheld"
        assert spread.withheld_reason == "incomplete_outcome_set"

    def test_prop_set_requires_both_sides(self):
        view = build_market_view(_prop_trace())
        probs = view.probability_sets[0]
        assert probs.disclosure == "shown"
        assert [o.outcome_key for o in probs.outcomes] == ["over", "under"]
        trace = _prop_trace()
        trace["result"].pop("under_prob")
        probs = build_market_view(trace).probability_sets[0]
        assert probs.disclosure == "withheld"
        assert probs.withheld_reason == "incomplete_outcome_set"

    def test_research_candidate_withholds_probabilities_keeps_distributions(self):
        trace = _game_trace(calibration_audit=[])  # no profile -> research_candidate
        view = build_market_view(trace)
        assert view.output_mode == "research_candidate"
        for probs in view.probability_sets:
            assert probs.disclosure == "withheld"
            assert probs.withheld_reason == "research_candidate_output_mode"
        # Raw simulation summaries stay visible with the simulation label.
        assert len(view.distributions) == 1
        assert "not a recommendation" in view.distributions[0].simulation_label
        # Listed market lines stay visible as context.
        assert view.market_lines == {"moneyline_home": -120, "moneyline_away": 100}

    def test_retired_maturity_withholds(self):
        trace = _game_trace(
            calibration_audit=[
                {
                    "profile_id": "iso_x",
                    "profile_maturity": "retired",
                    "sample_size": 400,
                    "ece": 0.03,
                }
            ]
        )
        view = build_market_view(trace)
        assert view.output_mode == "research_candidate"
        assert view.probability_sets[0].disclosure == "withheld"

    def test_sensitivity_is_explicitly_unavailable(self):
        view = build_market_view(_game_trace())
        assert view.sensitivity.status == "unavailable"
        assert "not available" in (view.sensitivity.reason or "")
        assert view.sensitivity.scenarios == []

    def test_engine_persisted_sensitivity_is_rendered_not_computed(self):
        trace = _game_trace()
        trace["result"]["sensitivity"] = [
            {"input": "home_off_rating", "range": [4.5, 5.5], "seed": 123}
        ]
        view = build_market_view(trace)
        assert view.sensitivity.status == "available"
        assert view.sensitivity.scenarios[0]["input"] == "home_off_rating"


# -- adapter: legacy compatibility ------------------------------------------------


class TestLegacyCompatibility:
    def test_legacy_prose_maps_and_verdict_stays_lab_only(self):
        trace = _game_trace(
            reasoning_presentation={
                "thesis": "Yankees rotation edge",
                "market_read": "Line moved home overnight",
                "why": "Their bats are better",
                "risks": "Bullpen fatigue",
                "verdict": "BET home side",
            }
        )
        legacy = build_market_view(trace).legacy_presentation
        assert legacy is not None
        assert legacy.summary == "Yankees rotation edge"
        assert legacy.market_context == "Line moved home overnight"
        assert legacy.uncertainties == ["Bullpen fatigue"]
        assert legacy.one_sided_case == "Their bats are better"
        assert legacy.incomplete is True
        dumped = legacy.model_dump(mode="json")
        assert "verdict" not in json.dumps(dumped)

    def test_blocked_legacy_prose_is_withheld_with_warning(self):
        trace = _game_trace(
            reasoning_presentation={
                "thesis": "This is our best bet tonight",
                "risks": "Bullpen fatigue",
            }
        )
        view = build_market_view(trace)
        assert view.legacy_presentation.summary is None
        assert view.legacy_presentation.uncertainties == ["Bullpen fatigue"]
        assert any("blocked language" in n for n in view.data_quality)


class TestSourceProvenance:
    def test_rsvg_summaries_carry_provenance_status(self):
        trace = _game_trace(
            trace_quality={
                "rsvg": {
                    "status": "pass",
                    "source_summaries": [
                        {
                            "source": "mlb.com",
                            "summary": "Lineups confirmed.",
                            "source_title": "Yankees lineup notes",
                            "source_url": "https://mlb.com/x",
                            "retrieved_at": "2026-07-16T15:00:00Z",
                        },
                        {
                            "source": "beat-writer",
                            "summary": "Bullpen usage note.",
                            "source_url": "https://x.com/y",
                        },
                        {"source": "clubhouse rumor", "summary": "Vague chatter."},
                    ],
                }
            },
            reasoning_inputs={"sources": ["mlb.com", "espn.com"]},
        )
        views = build_market_view(trace).sources
        by_source = {v.source: v for v in views}
        assert by_source["mlb.com"].provenance_status == "ok"
        assert by_source["mlb.com"].source_title == "Yankees lineup notes"
        assert by_source["beat-writer"].provenance_status == "partial"
        assert by_source["clubhouse rumor"].provenance_status == "missing_provenance"
        # Flat reasoning source not already covered by an RSVG summary appears
        # once, labeled missing provenance; the duplicate label is not repeated.
        assert by_source["espn.com"].provenance_status == "missing_provenance"
        assert len(views) == 4

    def test_legacy_flat_sources_still_render(self):
        trace = _game_trace(reasoning_inputs={"sources": ["https://espn.com/recap"]})
        views = build_market_view(trace).sources
        assert views[0].source_url == "https://espn.com/recap"
        assert views[0].provenance_status == "partial"


# -- adapter: grouping, ordering, denial sweep ------------------------------------


class TestGroupingAndSafety:
    def test_shared_event_key_groups_game_and_prop(self):
        briefs = group_traces_into_briefs([_prop_trace(), _game_trace()])
        assert len(briefs) == 1
        brief = briefs[0]
        assert brief.event_key == EVENT_IDENTITY["event_key"]
        assert not brief.identity_warning
        # Stable market-identity order: game first, then props by player/market.
        assert [m.market_group for m in brief.markets] == ["game", "Aaron Judge hits"]

    def test_legacy_traces_stay_singleton_with_identity_warning(self):
        old = _game_trace(trace_id="t-old", event_identity=None)
        briefs = group_traces_into_briefs([old, _game_trace()])
        keys = {b.group_key for b in briefs}
        assert keys == {"trace:t-old", EVENT_IDENTITY["event_key"]}
        legacy = next(b for b in briefs if b.group_key == "trace:t-old")
        assert legacy.identity_warning
        assert any(
            "identity_warning:no_provider_event_identity" in n
            for m in legacy.markets
            for n in m.data_quality
        )

    def test_mixed_identities_cannot_share_a_brief(self):
        other = _game_trace(
            trace_id="t-other",
            event_identity={**EVENT_IDENTITY, "provider_event_id": "ev-002",
                            "event_key": "MLB::the-odds-api::ev-002"},
        )
        with pytest.raises(ValueError, match="different event identities"):
            build_matchup_brief([_game_trace(), other])

    def test_group_key_for_trace(self):
        assert group_key_for_trace(_game_trace()) == (
            EVENT_IDENTITY["event_key"],
            EVENT_IDENTITY["event_key"],
            False,
        )
        assert group_key_for_trace({"trace_id": "x"}) == ("trace:x", None, True)

    def test_no_recommendation_trace_is_included(self):
        trace = _game_trace()
        trace["result"]["best_bet"] = None
        trace["result"]["edges"] = []
        briefs = group_traces_into_briefs([trace])
        assert len(briefs) == 1
        assert briefs[0].markets[0].probability_sets[0].disclosure == "withheld"

    def test_serialized_brief_contains_no_denied_keys_or_phrases(self):
        brief = build_matchup_brief([_game_trace(), _prop_trace()])
        dumped = json.dumps(brief.model_dump(mode="json"))
        for key in sorted(DENYLIST_KEYS):
            assert f'"{key}"' not in dumped, key
        from omega.core.contracts.language import blocked_language

        assert blocked_language(dumped) == []

    def test_denied_key_sweep_raises(self):
        with pytest.raises(DecisionSupportViolation):
            assert_no_denied_keys({"nested": [{"kelly_fraction": 0.02}]})

    def test_briefs_sorted_by_stable_identity(self):
        early = _game_trace(
            trace_id="t-early",
            event_identity={**EVENT_IDENTITY, "provider_event_id": "ev-0",
                            "event_key": "MLB::the-odds-api::ev-0",
                            "game_date": "2026-07-01"},
            input_snapshot={"league": "MLB", "home_team": "A", "away_team": "B",
                            "game_date": "2026-07-01"},
        )
        briefs = group_traces_into_briefs([_game_trace(), early])
        assert [b.game_date for b in briefs] == ["2026-07-01", "2026-07-16"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
