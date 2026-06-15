"""Tests for NFL NB-dispersion prop-prior injection (Phase 7 M4 gatherer wiring).

inject_prop_priors merges the fitted/shrunk dispersion k from
priors_nfl_dispersion into a prop request's player_context at the gatherer layer
(replay-safe), and analyze_player_prop prefers it over the per-request
std-derived k.
"""

from __future__ import annotations

import math
import tempfile

import pytest

from omega.core.contracts.schemas import PlayerPropRequest
from omega.core.contracts.service import analyze_player_prop
from omega.trace.priors import (
    NflDispersionPrior,
    inject_prop_priors,
    upsert_nfl_dispersion,
)
from omega.trace.store import TraceStore


def _tmp_store() -> TraceStore:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return TraceStore(db_path=f.name)


def _seed(
    store,
    entity="Saquon Barkley",
    stat="rushing_yards",
    k=3.5,
    source="player",
    as_of_date="2026-06-15",
):
    upsert_nfl_dispersion(
        store,
        NflDispersionPrior(
            entity=entity,
            stat_type=stat,
            season="2025",
            position_group="RB",
            nb_dispersion_k=k,
            nb_k_shrinkage_weight=0.8,
            nb_k_source=source,
            n_observations=200,
            as_of_date=as_of_date,
        ),
    )


def _payload(**kw):
    base = {
        "league": "NFL",
        "player_name": "Saquon Barkley",
        "prop_type": "rushing_yards",
        "player_context": {"rushing_yards_mean": 90.0, "rushing_yards_std": 35.0},
    }
    base.update(kw)
    return base


def test_injects_fitted_dispersion_with_provenance():
    store = _tmp_store()
    try:
        _seed(store, k=3.5, source="position_group")
        out, event = inject_prop_priors(_payload(), store=store)
        ctx = out["player_context"]
        assert ctx["nb_dispersion_k"] == pytest.approx(3.5)
        assert ctx["nb_k_source"] == "position_group"
        assert "nb_k_shrinkage_weight" in ctx
        assert event is not None and event["status"] == "ok"
        assert event["outputs"]["nb_k_source"] == "position_group"
    finally:
        store.close()


def test_injects_with_player_and_stat_aliases():
    store = _tmp_store()
    try:
        _seed(store, entity="Patrick Mahomes", stat="passing_yards", k=6.25)
        out, event = inject_prop_priors(
            {
                "league": "NFL",
                "player_name": "Pat Mahomes",
                "prop_type": "pass_yds",
                "player_context": {"pass_yds_mean": 285.0, "pass_yds_std": 42.0},
            },
            store=store,
        )
        ctx = out["player_context"]
        assert ctx["nb_dispersion_k"] == pytest.approx(6.25)
        assert event is not None and event["status"] == "ok"
        assert "Patrick Mahomes passing_yards" in event["notes"]
    finally:
        store.close()


def test_warns_when_no_fitted_row():
    store = _tmp_store()
    try:
        out, event = inject_prop_priors(_payload(), store=store)
        assert "nb_dispersion_k" not in (out.get("player_context") or {})
        assert event is not None and event["status"] == "warn"
        assert "omega-fit-nfl-dispersion" in event["notes"]
    finally:
        store.close()


def test_injection_uses_prior_knowable_by_game_date():
    store = _tmp_store()
    try:
        _seed(store, k=3.5, as_of_date="2026-05-01")
        _seed(store, k=9.9, as_of_date="2026-09-01")
        out, event = inject_prop_priors(_payload(game_date="2026-06-01"), store=store)
        assert out["player_context"]["nb_dispersion_k"] == pytest.approx(3.5)
        assert event is not None and event["status"] == "ok"
    finally:
        store.close()


def test_noop_for_non_nb_league_stat():
    # NBA points is not NB-routed -> no injection, no event, no store touched.
    out, event = inject_prop_priors(
        {
            "league": "NBA",
            "player_name": "Nikola Jokic",
            "prop_type": "pts",
            "player_context": {"pts_mean": 27.0},
        }
    )
    assert event is None
    assert "nb_dispersion_k" not in out["player_context"]


def test_caller_supplied_k_is_preserved_for_replay():
    store = _tmp_store()
    try:
        _seed(store, k=3.5)  # a different live value exists
        payload = _payload()
        payload["player_context"]["nb_dispersion_k"] = 9.9  # recorded request
        out, event = inject_prop_priors(payload, store=store)
        assert out["player_context"]["nb_dispersion_k"] == 9.9  # not overwritten
        assert event is None
    finally:
        store.close()


def test_service_prefers_injected_k_over_std_derived():
    injected = PlayerPropRequest(
        player_name="Saquon Barkley",
        league="NFL",
        prop_type="rushing_yards",
        line=82.5,
        game_date="2026-09-10",
        home_team="Eagles",
        away_team="Cowboys",
        n_iterations=4000,
        seed=7,
        game_context={"is_playoff": False, "rest_days": 7},
        player_context={
            "rushing_yards_mean": 90.0,
            "rushing_yards_std": 35.0,
            "nb_dispersion_k": 2.0,
            "nb_k_source": "position_group",
        },
    )
    resp = analyze_player_prop(injected)
    assert resp.status == "success"
    params = resp.simulation_distributions[0]["distribution_params"]
    assert params["k"] == pytest.approx(2.0)  # fitted k used, not std-derived
    assert "nb_k_source:position_group" in resp.notes

    # Without an injected k, the backend derives a different k from the std.
    std_derived = injected.model_copy(
        update={"player_context": {"rushing_yards_mean": 90.0, "rushing_yards_std": 35.0}}
    )
    resp2 = analyze_player_prop(std_derived)
    assert resp2.simulation_distributions[0]["distribution_params"]["k"] != pytest.approx(2.0)


@pytest.mark.parametrize("bad_k", [0, -1, math.inf, "not-a-number"])
def test_service_ignores_invalid_injected_k_and_derives_from_projection_std(bad_k):
    req = PlayerPropRequest(
        player_name="Saquon Barkley",
        league="NFL",
        prop_type="rushing_yards",
        line=82.5,
        game_date="2026-09-10",
        home_team="Eagles",
        away_team="Cowboys",
        n_iterations=500,
        seed=7,
        game_context={"is_playoff": False, "rest_days": 7},
        player_context={
            "rushing_yards_mean": 90.0,
            "rushing_yards_std": 30.0,
            "nb_dispersion_k": bad_k,
        },
    )
    resp = analyze_player_prop(req)
    assert resp.status == "success"
    params = resp.simulation_distributions[0]["distribution_params"]
    assert params["k"] == pytest.approx(10.0)
    assert not any(note.startswith("nb_k_source:") for note in resp.notes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nb_dispersion_k": 0.0},
        {"nb_dispersion_k": math.inf},
        {"nb_k_shrinkage_weight": -0.01},
        {"nb_k_shrinkage_weight": 1.01},
    ],
)
def test_nfl_dispersion_prior_validates_numeric_domains(kwargs):
    data = {
        "entity": "Saquon Barkley",
        "stat_type": "rushing_yards",
        "season": "2025",
        "position_group": "RB",
        "nb_dispersion_k": 3.5,
        "nb_k_shrinkage_weight": 0.8,
        "nb_k_source": "player",
        "n_observations": 200,
        "as_of_date": "2026-06-15",
    }
    data.update(kwargs)
    with pytest.raises(ValueError):
        NflDispersionPrior(**data)
