"""M2 acceptance: soccer bivariate-Poisson replay determinism + rho fail-closed.

Design gates (docs/phase7/MULTI_SPORT_EXPANSION.md Milestone 2):
  * 10 historical EPL/UCL-style matches replay bit-identically with non-zero
    draw_prob through the canonical analyze_game path;
  * an analysis without a Dixon-Coles prior returns status="skipped" with
    missing_requirements=["rho_prior"] end-to-end;
  * the gatherer injects rho + provenance from the production profile and the
    session sidecar records a data_provenance event;
  * swapping competition profiles (different rho) measurably changes draw_prob
    for identical xG inputs.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze_game
from omega.trace.priors import (
    DixonColesProfile,
    build_game_prior_payload,
    inject_game_priors,
    promote_dixon_coles_profile,
    upsert_dixon_coles_profile,
)
from omega.trace.session_sidecar import append_audit_events, bootstrap_payload
from omega.trace.store import TraceStore

_TEST_RHO = -0.13

# Ten EPL/UCL fixtures with season-realistic xG attack/defense rates.
_FIXTURES = [
    ("EPL", "Manchester City", "Liverpool", 2.1, 1.0, 1.8, 1.1),
    ("EPL", "Arsenal", "Chelsea", 1.9, 0.9, 1.4, 1.3),
    ("EPL", "Tottenham", "Manchester United", 1.6, 1.4, 1.4, 1.4),
    ("EPL", "Newcastle", "Aston Villa", 1.7, 1.2, 1.6, 1.5),
    ("EPL", "Brighton", "West Ham", 1.5, 1.3, 1.2, 1.6),
    ("CHAMPIONS_LEAGUE", "Real Madrid", "Bayern Munich", 2.0, 1.1, 1.9, 1.2),
    ("CHAMPIONS_LEAGUE", "Barcelona", "PSG", 1.8, 1.2, 1.9, 1.3),
    ("CHAMPIONS_LEAGUE", "Inter Milan", "Borussia Dortmund", 1.6, 1.0, 1.5, 1.4),
    ("CHAMPIONS_LEAGUE", "Atletico Madrid", "Juventus", 1.3, 0.9, 1.2, 1.0),
    ("CHAMPIONS_LEAGUE", "Benfica", "Ajax", 1.5, 1.3, 1.6, 1.5),
]


def _request(league, home, away, hxg, hxga, axg, axga, *, seed, prior=None, odds=None):
    return GameAnalysisRequest(
        home_team=home,
        away_team=away,
        league=league,
        n_iterations=4000,
        seed=seed,
        simulation_backend="soccer_bivariate_poisson_dc",
        home_context={"xg_for": hxg, "xg_against": hxga},
        away_context={"xg_for": axg, "xg_against": axga},
        game_context={"is_playoff": False, "rest_days": 3},
        prior_payload=prior if prior is not None else {"rho": _TEST_RHO},
        odds=odds,
    )


def _tmp_store() -> TraceStore:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return TraceStore(db_path=f.name)


def _promote_profile(store, profile_id="fifa_intl_v1", rho=-0.11, as_of="2026-06-10"):
    upsert_dixon_coles_profile(
        store,
        DixonColesProfile(
            profile_id=profile_id,
            rho=rho,
            n_matches=900,
            as_of_date=as_of,
            source="statsbomb_open_data",
        ),
    )
    promote_dixon_coles_profile(store, profile_id, as_of)


# ---------------------------------------------------------------------------
# Gate 1 — replay determinism with non-zero draw_prob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "league,home,away,hxg,hxga,axg,axga",
    _FIXTURES,
    ids=[f"{f[1]}_v_{f[2]}".replace(" ", "_") for f in _FIXTURES],
)
def test_replay_is_bit_identical_with_nonzero_draw_prob(league, home, away, hxg, hxga, axg, axga):
    seed = int.from_bytes(hashlib.sha256(f"{home}|{away}".encode()).digest()[:4], "big") % 100_000
    req = _request(league, home, away, hxg, hxga, axg, axga, seed=seed)
    first = analyze_game(req)
    second = analyze_game(req)

    assert first.status == "success"
    assert first.simulation is not None and second.simulation is not None
    assert first.simulation.simulation_backend == "soccer_bivariate_poisson_dc"
    assert first.simulation.component_version == "soccer_bvp_dc_v1"
    assert first.simulation.draw_prob is not None and first.simulation.draw_prob > 0.0

    for field in (
        "home_win_prob",
        "away_win_prob",
        "draw_prob",
        "predicted_home_score",
        "predicted_away_score",
        "predicted_spread",
        "predicted_total",
    ):
        assert getattr(first.simulation, field) == getattr(second.simulation, field)


def test_three_way_edges_priced_when_draw_odds_supplied():
    league, home, away, hxg, hxga, axg, axga = _FIXTURES[0]
    odds = {
        "moneyline_home": -120,
        "moneyline_away": +320,
        "moneyline_draw": +270,
        "over_under": 2.5,
    }
    resp = analyze_game(_request(league, home, away, hxg, hxga, axg, axga, seed=11, odds=odds))
    assert resp.status == "success"
    sides = {e.side for e in resp.edges}
    assert "draw" in sides  # 3-way moneyline evaluated, not just home/away


# ---------------------------------------------------------------------------
# Gate 2 — fail-closed end-to-end when the rho prior is missing
# ---------------------------------------------------------------------------


def test_world_cup_league_defaults_to_dc_backend_and_skips_without_rho():
    req = GameAnalysisRequest(
        home_team="France",
        away_team="Brazil",
        league="FIFA_WORLD_CUP_2026",
        n_iterations=500,
        seed=7,
        # simulation_backend left at default -> league default_game_backend
        home_context={"xg_for": 1.8, "xg_against": 0.9},
        away_context={"xg_for": 1.7, "xg_against": 1.0},
        game_context={"is_playoff": True, "rest_days": 4},
    )
    resp = analyze_game(req)
    assert resp.status == "skipped"
    assert resp.missing_requirements == ["rho_prior"]


# ---------------------------------------------------------------------------
# Gate 3 — gatherer injection + sidecar provenance
# ---------------------------------------------------------------------------


def test_injection_merges_production_rho_with_provenance():
    store = _tmp_store()
    try:
        _promote_profile(store, rho=-0.11, as_of="2026-06-10")
        payload = {
            "league": "FIFA_WORLD_CUP_2026",
            "home_team": "France",
            "away_team": "Brazil",
        }
        merged, event = inject_game_priors(payload, store=store)
        prior = merged["prior_payload"]
        assert prior["rho"] == pytest.approx(-0.11)
        assert prior["rho_profile_id"] == "fifa_intl_v1"
        assert prior["rho_as_of_date"] == "2026-06-10"
        assert event is not None and event["status"] == "ok"
        assert event["outputs"]["rho_profile_id"] == "fifa_intl_v1"
        assert event["outputs"]["rho_as_of_date"] == "2026-06-10"
    finally:
        store.close()


def test_injection_warns_and_leaves_payload_when_no_production_profile():
    store = _tmp_store()
    try:
        payload = {"league": "FIFA_WORLD_CUP_2026"}
        merged, event = inject_game_priors(payload, store=store)
        assert "prior_payload" not in merged or merged["prior_payload"] is None
        assert event is not None and event["status"] == "warn"
        assert "fail closed" in event["notes"]
    finally:
        store.close()


def test_injection_preserves_caller_supplied_rho_for_replay():
    store = _tmp_store()
    try:
        _promote_profile(store, rho=-0.11)
        recorded = {"rho": -0.2, "rho_profile_id": "recorded_run"}
        merged, event = build_game_prior_payload("FIFA_WORLD_CUP_2026", recorded, store)
        assert merged == recorded  # replay never re-reads live table state
        assert event is None
    finally:
        store.close()


def test_injection_is_noop_for_leagues_without_profile():
    payload = {"league": "NBA", "home_team": "Lakers", "away_team": "Celtics"}
    merged, event = inject_game_priors(payload)
    assert merged == payload
    assert event is None


def test_provenance_event_lands_in_session_sidecar(tmp_path):
    store = _tmp_store()
    try:
        _promote_profile(store)
        _, event = inject_game_priors({"league": "FIFA_WORLD_CUP_2026"}, store=store)
        sidecar = tmp_path / "test-session.json"
        sidecar.write_text(
            json.dumps(
                bootstrap_payload(
                    "test-session",
                    model_version="test",
                    purpose="m2 acceptance",
                    bankroll=1000.0,
                )
            ),
            encoding="utf-8",
        )
        append_audit_events(sidecar, [event])
        saved = json.loads(Path(sidecar).read_text(encoding="utf-8"))
        events = [e for e in saved["audit_events"] if e["event_type"] == "data_provenance"]
        assert len(events) == 1
        assert events[0]["outputs"]["rho_profile_id"] == "fifa_intl_v1"
        assert events[0]["outputs"]["rho_as_of_date"] == "2026-06-10"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Gate 4 — differential rho profiles change draw_prob for identical xG
# ---------------------------------------------------------------------------


def test_profile_rho_measurably_shifts_draw_prob():
    league, home, away, hxg, hxga, axg, axga = _FIXTURES[2]  # near-even matchup
    flat = analyze_game(
        _request(
            league,
            home,
            away,
            hxg,
            hxga,
            axg,
            axga,
            seed=99,
            prior={"rho": 0.0, "rho_profile_id": "flat_test"},
        )
    )
    corrected = analyze_game(
        _request(
            league,
            home,
            away,
            hxg,
            hxga,
            axg,
            axga,
            seed=99,
            prior={"rho": -0.2, "rho_profile_id": "intl_test"},
        )
    )
    assert flat.status == corrected.status == "success"
    assert corrected.simulation.draw_prob > flat.simulation.draw_prob
