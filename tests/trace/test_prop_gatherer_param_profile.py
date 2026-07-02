"""P8.5 fast-follow — the prop gatherer merges a promoted prop backend parameter profile.

The prop mirror of test_gatherer_param_profile.py: inject_prop_priors now also
injects the production BackendParameterProfile's structural knobs (nb_k_scale)
+ parameter_profile_ref into player_context via _merge_prop_parameter_profile,
bucketed per (league, stat) by resolve_prop_calibration_bucket
(prop_neg_binom / NFL__RUSHING_YARDS). When no prop profile is promoted (the
current live state) the payload is unchanged — purely additive, bit-identical
until one is promoted. Replay-safe: embedded refs / OMEGA_REPLAY_MODE are never
overridden, and analyze_player_prop forwards the embedded knobs from
player_context into the backend's prior_payload so a recorded request replays
byte-faithful.
"""

from __future__ import annotations

import tempfile

import pytest

from omega.core.calibration.league_buckets import resolve_prop_calibration_bucket
from omega.core.contracts.schemas import PlayerPropRequest
from omega.core.contracts.service import analyze_player_prop
from omega.core.simulation.parameter_profile import (
    BackendParameterProfile,
    make_parameter_profile_id,
)
from omega.trace.parameter_profiles import (
    promote_parameter_profile,
    register_parameter_profile,
)
from omega.trace.priors import (
    NflDispersionPrior,
    inject_prop_priors,
    upsert_nfl_dispersion,
)
from omega.trace.store import TraceStore

_BACKEND = "prop_neg_binom"
_CONFIRMS = {
    "confirm_backtest_parity": True,
    "confirm_clv_non_regression": True,
    "parity_evidence": {"state": "PASS"},
    "clv_evidence": {"verdict": "PASS"},
}


def _store() -> TraceStore:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return TraceStore(db_path=f.name)


def _seed_dispersion(store: TraceStore, k: float = 3.5) -> None:
    upsert_nfl_dispersion(
        store,
        NflDispersionPrior(
            entity="Saquon Barkley",
            stat_type="rushing_yards",
            season="2025",
            position_group="RB",
            nb_dispersion_k=k,
            nb_k_shrinkage_weight=0.8,
            nb_k_source="player",
            n_observations=200,
            as_of_date="2026-06-15",
        ),
    )


def _promote_prop_profile(store: TraceStore, params: dict) -> BackendParameterProfile:
    bucket = resolve_prop_calibration_bucket("NFL", "rushing_yards")
    prof = BackendParameterProfile(
        profile_id=make_parameter_profile_id(_BACKEND, bucket, 2, params),
        version=2,
        backend_name=_BACKEND,
        backend_component_version="prop_nb_v1",
        competition_bucket=bucket,
        params=params,
        dataset_hash="h",
        sample_size=150,
        metrics={"brier_score": 0.20, "calibration_error": 0.03, "log_loss": 0.60},
    )
    register_parameter_profile(store, prof)
    promote_parameter_profile(store, prof.profile_id, **_CONFIRMS)
    return prof


def _payload(**kw):
    base = {
        "league": "NFL",
        "player_name": "Saquon Barkley",
        "prop_type": "rushing_yards",
        "player_context": {"rushing_yards_mean": 90.0, "rushing_yards_std": 35.0},
    }
    base.update(kw)
    return base


def test_prop_gatherer_merges_promoted_profile(monkeypatch):
    monkeypatch.delenv("OMEGA_REPLAY_MODE", raising=False)
    store = _store()
    try:
        _seed_dispersion(store, k=3.5)
        prof = _promote_prop_profile(store, {"nb_k_scale": 1.5})
        out, event = inject_prop_priors(_payload(), store=store)
        ctx = out["player_context"]
        assert ctx["nb_dispersion_k"] == pytest.approx(3.5)  # dispersion still injected
        assert ctx["nb_k_scale"] == 1.5  # structural knob from the profile
        assert ctx["parameter_profile_ref"]["param_profile_id"] == prof.profile_id
        assert ctx["parameter_profile_ref"]["competition_bucket"] == "NFL__RUSHING_YARDS"
        assert event["status"] == "ok"
        assert prof.profile_id in event["notes"]
        assert event["outputs"]["parameter_profile_ref"]["param_profile_id"] == prof.profile_id
    finally:
        store.close()


def test_prop_gatherer_no_profile_is_bit_identical():
    """Only the dispersion prior exists -> player_context carries the fitted k but
    NO structural knobs and NO parameter_profile_ref. The backend then runs at its
    identity defaults (bit-identical until a prop profile is promoted)."""
    store = _store()
    try:
        _seed_dispersion(store, k=3.5)
        out, event = inject_prop_priors(_payload(), store=store)
        ctx = out["player_context"]
        assert ctx["nb_dispersion_k"] == pytest.approx(3.5)
        assert "nb_k_scale" not in ctx
        assert "parameter_profile_ref" not in ctx
        assert event["status"] == "ok"
        assert "outputs" in event and "parameter_profile_ref" not in event["outputs"]
    finally:
        store.close()


def test_prop_gatherer_merges_profile_even_when_k_pre_supplied(monkeypatch):
    """Mirror of the game path's pre-supplied-rho fix: a live request that
    pre-supplies nb_dispersion_k still receives the promoted structural knobs."""
    monkeypatch.delenv("OMEGA_REPLAY_MODE", raising=False)
    store = _store()
    try:
        prof = _promote_prop_profile(store, {"nb_k_scale": 1.5})
        payload = _payload()
        payload["player_context"]["nb_dispersion_k"] = 9.9
        out, event = inject_prop_priors(payload, store=store)
        ctx = out["player_context"]
        assert ctx["nb_dispersion_k"] == 9.9  # caller's k preserved
        assert ctx["nb_k_scale"] == 1.5  # param profile still merged
        assert ctx["parameter_profile_ref"]["param_profile_id"] == prof.profile_id
        assert event["status"] == "ok"
        assert event["step"] == "prop_parameter_profile:inject"
    finally:
        store.close()


def test_prop_gatherer_skips_profile_ref_when_structural_knob_overridden(monkeypatch):
    """Do not stamp a production profile ref when caller-supplied structural
    knobs mean the backend will price a different raw substrate."""
    monkeypatch.delenv("OMEGA_REPLAY_MODE", raising=False)
    store = _store()
    try:
        _promote_prop_profile(store, {"nb_k_scale": 1.5})
        payload = _payload()
        payload["player_context"].update(nb_dispersion_k=9.9, nb_k_scale=0.9)
        out, event = inject_prop_priors(payload, store=store)
        ctx = out["player_context"]
        assert ctx["nb_dispersion_k"] == 9.9
        assert ctx["nb_k_scale"] == 0.9
        assert "parameter_profile_ref" not in ctx
        assert event is None
    finally:
        store.close()


def test_prop_gatherer_replay_mode_skips_live_lookup(monkeypatch):
    """Under OMEGA_REPLAY_MODE the live parameter_profiles table is never read, so
    a post-hoc promotion cannot leak into a historical replay."""
    store = _store()
    try:
        _promote_prop_profile(store, {"nb_k_scale": 1.5})
        monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
        payload = _payload()
        payload["player_context"]["nb_dispersion_k"] = 9.9  # recorded request
        out, event = inject_prop_priors(payload, store=store)
        assert "nb_k_scale" not in out["player_context"]
        assert "parameter_profile_ref" not in out["player_context"]
        assert event is None
    finally:
        store.close()


def test_prop_gatherer_does_not_overwrite_pinned_ref(monkeypatch):
    """A recorded request that already carries a parameter_profile_ref is left
    untouched — replay must not re-read the live table or its knobs."""
    monkeypatch.delenv("OMEGA_REPLAY_MODE", raising=False)
    store = _store()
    try:
        _promote_prop_profile(store, {"nb_k_scale": 1.5})
        payload = _payload()
        payload["player_context"].update(
            nb_dispersion_k=9.9,
            nb_k_scale=0.9,
            parameter_profile_ref={"param_profile_id": "pinned"},
        )
        out, event = inject_prop_priors(payload, store=store)
        ctx = out["player_context"]
        assert ctx["parameter_profile_ref"]["param_profile_id"] == "pinned"
        assert ctx["nb_k_scale"] == 0.9
        assert event is None
    finally:
        store.close()


def test_prop_gatherer_router_pair_untouched_no_store():
    """A router-backed (ungoverned) pair returns unchanged without opening any
    store — no profile can govern the distribution-router dispatcher."""
    out, event = inject_prop_priors(
        {
            "league": "NBA",
            "player_name": "Nikola Jokic",
            "prop_type": "pts",
            "player_context": {"pts_mean": 27.0},
        }
    )
    assert event is None
    assert "nb_k_scale" not in out["player_context"]
    assert "parameter_profile_ref" not in out["player_context"]


def _prop_request(player_context: dict) -> PlayerPropRequest:
    return PlayerPropRequest(
        player_name="Saquon Barkley",
        league="NFL",
        prop_type="rushing_yards",
        line=82.5,
        game_date="2026-09-10",
        home_team="Eagles",
        away_team="Cowboys",
        n_iterations=2000,
        seed=7,
        game_context={"is_playoff": False, "rest_days": 7},
        player_context=player_context,
    )


def test_service_applies_embedded_profile_knobs_and_echoes_provenance():
    """End-to-end consumer seam: the knobs the gatherer embedded in player_context
    reach the NB backend's prior_payload — distribution_params.k is the POST-scale
    k, the applied nb_k_scale is echoed next to it (so the frozen-artifact builder
    can divide the base back out), and the ref lands on the response for the V20
    traces.parameter_profile_ref column."""
    ref = {"param_profile_id": "pp-test", "backend_name": "prop_neg_binom"}
    resp = analyze_player_prop(
        _prop_request(
            {
                "rushing_yards_mean": 90.0,
                "rushing_yards_std": 35.0,
                "nb_dispersion_k": 2.0,
                "nb_k_scale": 1.5,
                "parameter_profile_ref": ref,
            }
        )
    )
    assert resp.status == "success"
    params = resp.simulation_distributions[0]["distribution_params"]
    assert params["k"] == pytest.approx(3.0)  # 2.0 base * 1.5 profile scale
    assert params["nb_k_scale"] == pytest.approx(1.5)
    assert resp.parameter_profile_ref == ref


def test_service_without_profile_knobs_is_bit_identical():
    resp = analyze_player_prop(
        _prop_request(
            {
                "rushing_yards_mean": 90.0,
                "rushing_yards_std": 35.0,
                "nb_dispersion_k": 2.0,
            }
        )
    )
    assert resp.status == "success"
    params = resp.simulation_distributions[0]["distribution_params"]
    assert params["k"] == pytest.approx(2.0)  # identity: base k untouched
    assert "nb_k_scale" not in params
    assert resp.parameter_profile_ref is None
