"""Gatherer tennis-prior injection tests (Phase 7 M3 PR-T4).

The injection joins priors_tennis rates into missing serve/return context keys
and the latest pressure deltas (player rows, else __group__ fallback) into
prior_payload, with per-side provenance — and never re-reads live tables for a
recorded request.
"""

from __future__ import annotations

import tempfile

import pytest

from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze_game
from omega.trace.priors import (
    PRESSURE_GROUP_PLAYER_KEY,
    PRESSURE_SOURCE_GROUP,
    PRESSURE_SOURCE_PLAYER,
    TennisPressureDelta,
    TennisPrior,
    inject_game_priors,
    upsert_pressure_deltas,
    upsert_tennis_prior,
)
from omega.trace.store import TraceStore


def _store() -> TraceStore:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return TraceStore(db_path=f.name)


def _seed_store(store: TraceStore) -> None:
    for player, spw, rpw in (("Jannik Sinner", 0.67, 0.40), ("Novak Djokovic", 0.65, 0.39)):
        upsert_tennis_prior(
            store,
            TennisPrior(
                player=player, tour="ATP", surface="grass",
                spw_pct=spw, rpw_pct=rpw, n_matches=20, as_of_date="2026-06-10",
            ),
        )
    upsert_pressure_deltas(
        store,
        [
            TennisPressureDelta(
                player="Jannik Sinner", tour="ATP", surface="grass",
                state="break_point_against", delta=-0.021, n_points=900,
                source=PRESSURE_SOURCE_PLAYER, as_of_date="2026-06-10",
            ),
            TennisPressureDelta(
                player=PRESSURE_GROUP_PLAYER_KEY, tour="ATP", surface="grass",
                state="break_point_against", delta=-0.012, n_points=50_000,
                source=PRESSURE_SOURCE_GROUP, as_of_date="2026-06-10",
            ),
        ],
    )


def _payload(**overrides):
    base = {
        "league": "ATP",
        "home_team": "Jannik Sinner",
        "away_team": "Novak Djokovic",
        "game_context": {"surface": "Grass", "is_playoff": False, "rest_days": 2},
    }
    base.update(overrides)
    return base


def test_injection_fills_rates_and_joins_pressure_with_provenance():
    store = _store()
    try:
        _seed_store(store)
        merged, event = inject_game_priors(_payload(), store=store)

        assert merged["home_context"]["serve_win_pct"] == pytest.approx(0.67)
        assert merged["away_context"]["return_win_pct"] == pytest.approx(0.39)

        prior = merged["prior_payload"]
        assert prior["pressure_coefficients"]["home"]["break_point_against"] == pytest.approx(-0.021)
        # Djokovic has no player rows -> __group__ fallback, never zeros.
        assert prior["pressure_coefficients"]["away"]["break_point_against"] == pytest.approx(-0.012)
        assert prior["pressure_coefficient_source"] == {
            "home": PRESSURE_SOURCE_PLAYER,
            "away": PRESSURE_SOURCE_GROUP,
        }
        assert event["status"] == "ok"
        assert event["outputs"]["pressure_coefficient_source"]["away"] == PRESSURE_SOURCE_GROUP
    finally:
        store.close()


def test_caller_supplied_rates_win_over_table_rates():
    store = _store()
    try:
        _seed_store(store)
        payload = _payload(home_context={"serve_win_pct": 0.70, "return_win_pct": 0.42})
        merged, _ = inject_game_priors(payload, store=store)
        assert merged["home_context"]["serve_win_pct"] == pytest.approx(0.70)
    finally:
        store.close()


def test_recorded_request_with_coefficients_is_untouched():
    store = _store()
    try:
        _seed_store(store)
        recorded = _payload(
            prior_payload={"pressure_coefficients": {"home": {"tiebreak": -0.5}}}
        )
        merged, event = inject_game_priors(recorded, store=store)
        assert merged["prior_payload"]["pressure_coefficients"] == {
            "home": {"tiebreak": -0.5}
        }
        assert event is None
    finally:
        store.close()


def test_missing_surface_warns_and_passes_through():
    store = _store()
    try:
        _seed_store(store)
        merged, event = inject_game_priors(
            _payload(game_context={"is_playoff": False, "rest_days": 2}), store=store
        )
        assert "prior_payload" not in merged or merged.get("prior_payload") is None
        assert event["status"] == "warn"
        assert "surface" in event["notes"]
    finally:
        store.close()


def test_empty_tables_warn():
    store = _store()
    try:
        _, event = inject_game_priors(_payload(), store=store)
        assert event["status"] == "warn"
        assert "omega-refresh-sackmann" in event["notes"]
    finally:
        store.close()


def test_injected_payload_analyzes_end_to_end():
    store = _store()
    try:
        _seed_store(store)
        merged, _ = inject_game_priors(_payload(), store=store)
    finally:
        store.close()
    merged.setdefault("n_iterations", 1000)
    merged.setdefault("seed", 9)
    resp = analyze_game(GameAnalysisRequest(**merged))
    assert resp.status == "success"
    assert resp.simulation.simulation_backend == "tennis_markov_iid"
