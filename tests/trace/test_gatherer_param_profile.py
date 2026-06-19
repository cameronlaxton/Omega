"""P8.2 — the gatherer merges a promoted backend parameter profile (soccer pilot).

build_game_prior_payload now also injects the production BackendParameterProfile's
structural knobs (+ stamps parameter_profile_ref) alongside the Dixon-Coles rho.
When no parameter profile is promoted (the current live state) the payload is
unchanged beyond rho — purely additive, bit-identical until a profile is promoted.
Replay-safe: a caller-supplied ref/param is never overwritten.
"""

from __future__ import annotations

import tempfile

from omega.core.calibration.league_buckets import resolve_calibration_bucket
from omega.core.simulation.parameter_profile import (
    BackendParameterProfile,
    make_parameter_profile_id,
)
from omega.trace.parameter_profiles import (
    promote_parameter_profile,
    register_parameter_profile,
)
from omega.trace.priors import (
    DixonColesProfile,
    build_game_prior_payload,
    promote_dixon_coles_profile,
    upsert_dixon_coles_profile,
)
from omega.trace.store import TraceStore

_BACKEND = "soccer_bivariate_poisson_dc"
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


def _promote_rho(store: TraceStore) -> None:
    upsert_dixon_coles_profile(
        store,
        DixonColesProfile(profile_id="epl_v1", rho=-0.05, n_matches=900, as_of_date="2026-06-01"),
    )
    promote_dixon_coles_profile(store, "epl_v1", "2026-06-01")


def _promote_param_profile(store: TraceStore, params: dict) -> BackendParameterProfile:
    bucket = resolve_calibration_bucket("EPL")
    prof = BackendParameterProfile(
        profile_id=make_parameter_profile_id(_BACKEND, bucket, 2, params),
        version=2,
        backend_name=_BACKEND,
        backend_component_version="soccer_bvp_dc_v1",
        competition_bucket=bucket,
        params=params,
        dataset_hash="h",
        sample_size=150,
        metrics={"brier_score": 0.20, "calibration_error": 0.03, "log_loss": 0.60},
    )
    register_parameter_profile(store, prof)
    promote_parameter_profile(store, prof.profile_id, **_CONFIRMS)
    return prof


def test_gatherer_merges_promoted_param_profile():
    store = _store()
    try:
        _promote_rho(store)
        prof = _promote_param_profile(store, {"home_advantage": 0.5, "lambda_scale": 1.1})
        merged, event = build_game_prior_payload("EPL", None, store)
        assert merged["rho"] == -0.05  # DC rho still injected
        assert merged["home_advantage"] == 0.5  # structural knobs from the profile
        assert merged["lambda_scale"] == 1.1
        assert merged["parameter_profile_ref"]["param_profile_id"] == prof.profile_id
        assert event["status"] == "ok"
        assert "parameter_profile_ref" in event["outputs"]
    finally:
        store.close()


def test_gatherer_no_param_profile_is_additive():
    """Only rho promoted -> payload carries rho but NO structural knobs and NO
    parameter_profile_ref. The backend then uses its historical-constant defaults
    (bit-identical until a profile is promoted)."""
    store = _store()
    try:
        _promote_rho(store)
        merged, event = build_game_prior_payload("EPL", None, store)
        assert merged["rho"] == -0.05
        assert "home_advantage" not in merged
        assert "lambda_scale" not in merged
        assert "parameter_profile_ref" not in merged
    finally:
        store.close()


def test_gatherer_merges_param_profile_even_when_rho_pre_supplied(monkeypatch):
    """Fix for the rho-coupling finding: a live request that pre-supplies rho still
    receives the promoted structural knobs, not only freshly-fetched-rho requests."""
    monkeypatch.delenv("OMEGA_REPLAY_MODE", raising=False)
    store = _store()
    try:
        _promote_rho(store)
        prof = _promote_param_profile(store, {"home_advantage": 0.5, "lambda_scale": 1.1})
        merged, event = build_game_prior_payload("EPL", {"rho": -0.05}, store)
        assert merged["rho"] == -0.05  # caller's rho preserved
        assert merged["home_advantage"] == 0.5  # param profile still merged
        assert merged["lambda_scale"] == 1.1
        assert merged["parameter_profile_ref"]["param_profile_id"] == prof.profile_id
        assert event is None  # caller-supplied-rho path stays quiet
    finally:
        store.close()


def test_gatherer_replay_mode_skips_live_param_lookup(monkeypatch):
    """Under OMEGA_REPLAY_MODE the live parameter_profiles table is never read, so a
    post-hoc promotion cannot leak into a historical replay."""
    store = _store()
    try:
        _promote_rho(store)
        _promote_param_profile(store, {"home_advantage": 0.5})
        monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
        merged, _ = build_game_prior_payload("EPL", {"rho": -0.05}, store)
        assert merged["rho"] == -0.05
        assert "home_advantage" not in merged  # live lookup skipped in replay
        assert "parameter_profile_ref" not in merged
    finally:
        store.close()


def test_gatherer_replay_safe_does_not_overwrite_pinned_ref():
    """A recorded request that already carries a parameter_profile_ref is left
    untouched — replay must not re-read the live table."""
    store = _store()
    try:
        _promote_rho(store)
        _promote_param_profile(store, {"home_advantage": 0.5, "lambda_scale": 1.1})
        existing = {"parameter_profile_ref": {"param_profile_id": "pinned"}, "lambda_scale": 0.9}
        merged, _ = build_game_prior_payload("EPL", existing, store)
        # rho is still injected (caller did not supply it), but the pinned ref and
        # the caller's structural knob win over the live production profile.
        assert merged["rho"] == -0.05
        assert merged["parameter_profile_ref"]["param_profile_id"] == "pinned"
        assert merged["lambda_scale"] == 0.9
    finally:
        store.close()
