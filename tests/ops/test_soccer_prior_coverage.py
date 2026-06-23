"""Tests for omega.ops.soccer_prior_coverage — FIFA/soccer prior-coverage gate."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest

from omega.ops.soccer_prior_coverage import (
    TIER_MODERATE,
    TIER_NONE,
    TIER_STRONG,
    TIER_WEAK,
    build_coverage_report,
    gate_output_mode,
)
from omega.trace.priors import (
    DC_STATUS_CANDIDATE,
    DC_STATUS_PRODUCTION,
    DixonColesProfile,
    XgPrior,
    upsert_dixon_coles_profile,
    upsert_xg_prior,
)
from omega.trace.store import TraceStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "omega_traces.db"
    s = TraceStore(db_path=str(db))
    yield s
    s.close()


def _seed_dc_production(store: TraceStore, profile_id: str = "fifa_intl_v1") -> None:
    upsert_dixon_coles_profile(
        store,
        DixonColesProfile(
            profile_id=profile_id,
            rho=-0.07,
            n_matches=380,
            fit_loss=0.042,
            as_of_date="2026-06-10",
            status=DC_STATUS_PRODUCTION,
            source="statsbomb",
        ),
    )


def _seed_dc_candidate(store: TraceStore, profile_id: str = "fifa_intl_v1") -> None:
    upsert_dixon_coles_profile(
        store,
        DixonColesProfile(
            profile_id=profile_id,
            rho=-0.05,
            n_matches=200,
            fit_loss=0.055,
            as_of_date="2026-06-01",
            status=DC_STATUS_CANDIDATE,
            source="statsbomb",
        ),
    )


def _seed_xg(store: TraceStore, team: str, competition: str = "FIFA_WORLD_CUP_2026") -> None:
    upsert_xg_prior(
        store,
        XgPrior(
            team=team,
            competition=competition,
            season="2025/2026",
            xg_for=1.42,
            xg_against=0.95,
            matches=6,
            source="statsbomb",
            as_of_date="2026-06-10",
        ),
    )


# ---------------------------------------------------------------------------
# Tier assignment tests
# ---------------------------------------------------------------------------


def test_strong_tier_with_dc_and_xg(store):
    _seed_dc_production(store)
    _seed_xg(store, "Brazil")
    _seed_xg(store, "Argentina")

    report = build_coverage_report(
        "FIFA_WORLD_CUP_2026",
        home_team="Brazil",
        away_team="Argentina",
        season="2025/2026",
        store=store,
    )
    assert report.confidence_tier == TIER_STRONG
    assert report.recommended_output_mode == "actionable"
    assert report.dc_profile is not None
    assert report.dc_profile.rho == pytest.approx(-0.07)
    assert report.home_team is not None and report.home_team.has_xg
    assert report.away_team is not None and report.away_team.has_xg


def test_moderate_tier_dc_but_no_team_xg(store):
    _seed_dc_production(store)
    # No xG rows seeded

    report = build_coverage_report(
        "FIFA_WORLD_CUP_2026",
        home_team="Brazil",
        away_team="Argentina",
        store=store,
    )
    assert report.confidence_tier == TIER_MODERATE
    assert report.recommended_output_mode == "low_confidence_actionable"
    assert report.home_team is not None and not report.home_team.has_xg
    assert len(report.fallback_usage) > 0


def test_moderate_tier_dc_no_teams_requested(store):
    _seed_dc_production(store, profile_id="epl_v1")

    report = build_coverage_report("EPL", store=store)
    assert report.confidence_tier == TIER_MODERATE
    assert report.home_team is None


def test_weak_tier_candidate_only(store):
    _seed_dc_candidate(store)

    report = build_coverage_report("FIFA_WORLD_CUP_2026", store=store)
    assert report.confidence_tier == TIER_WEAK
    assert report.recommended_output_mode == "research_candidate"
    assert report.dc_profile is None
    assert report.dc_candidate_exists is True
    assert any("candidate" in w.lower() or "promote" in w.lower() for w in report.warnings)


def test_none_tier_no_dc_rows(store):
    report = build_coverage_report("FIFA_WORLD_CUP_2026", store=store)
    assert report.confidence_tier == TIER_NONE
    assert report.recommended_output_mode == "research_candidate"
    assert report.dc_profile is None
    assert report.dc_candidate_exists is False
    assert any("No Dixon-Coles rows" in w for w in report.warnings)


def test_none_tier_non_soccer_league(store):
    """A league without a rho_fit_profile (MLB, NBA) returns tier=none."""
    report = build_coverage_report("MLB", store=store)
    assert report.confidence_tier == TIER_NONE
    assert report.competition_profile_id is None


# ---------------------------------------------------------------------------
# Gate: weak/none coverage CANNOT become actionable
# ---------------------------------------------------------------------------


def test_gate_blocks_actionable_when_tier_is_none(store):
    report = build_coverage_report("FIFA_WORLD_CUP_2026", store=store)
    assert report.confidence_tier == TIER_NONE
    gated = gate_output_mode(report, "actionable")
    assert gated == "research_candidate"
    assert gated != "actionable"


def test_gate_blocks_actionable_when_tier_is_weak(store):
    _seed_dc_candidate(store)
    report = build_coverage_report("FIFA_WORLD_CUP_2026", store=store)
    assert report.confidence_tier == TIER_WEAK
    gated = gate_output_mode(report, "actionable")
    assert gated == "research_candidate"


def test_gate_downgrades_actionable_to_low_confidence_when_moderate(store):
    _seed_dc_production(store)
    report = build_coverage_report("FIFA_WORLD_CUP_2026", store=store)
    assert report.confidence_tier == TIER_MODERATE
    gated = gate_output_mode(report, "actionable")
    assert gated == "low_confidence_actionable"


def test_gate_passes_through_research_candidate_unchanged(store):
    report = build_coverage_report("FIFA_WORLD_CUP_2026", store=store)
    gated = gate_output_mode(report, "research_candidate")
    assert gated == "research_candidate"


def test_gate_strong_allows_actionable(store):
    _seed_dc_production(store)
    _seed_xg(store, "Germany")
    _seed_xg(store, "Spain")
    report = build_coverage_report(
        "FIFA_WORLD_CUP_2026",
        home_team="Germany",
        away_team="Spain",
        season="2025/2026",
        store=store,
    )
    assert report.confidence_tier == TIER_STRONG
    gated = gate_output_mode(report, "actionable")
    assert gated == "actionable"


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------


def test_to_dict_is_serialisable(store):
    import json

    _seed_dc_production(store)
    _seed_xg(store, "France")
    report = build_coverage_report(
        "FIFA_WORLD_CUP_2026",
        home_team="France",
        season="2025/2026",
        store=store,
    )
    d = report.to_dict()
    json.dumps(d)  # must not raise
    assert d["league"] == "FIFA_WORLD_CUP_2026"
    assert d["dc_profile"]["rho"] == pytest.approx(-0.07)
