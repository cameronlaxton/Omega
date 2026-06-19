"""Schema V19: parameter_profiles table + fail-closed backend-parameter promotion.

The governance rail for backend structural parameters. Promotion composes the
SAME shared gate engine as calibration (omega.core.governance.promotion_gates),
so a backend parameter profile reaches PRODUCTION only when every gate passes on
its RAW (pre-calibration) metrics, and exactly one profile per
(backend_name, competition_bucket) is production at a time.
"""

from __future__ import annotations

import tempfile

import pytest

from omega.core.governance.promotion_gates import PromotionGateError
from omega.core.simulation.parameter_profile import (
    BackendParameterProfile,
    ParameterProfileStatus,
    make_parameter_profile_id,
)
from omega.trace.parameter_profiles import (
    extract_parameter_profile_ref,
    get_parameter_profile,
    get_parameter_profile_ref,
    get_production_parameter_profile,
    list_parameter_profiles,
    parameter_pin_status,
    promote_parameter_profile,
    register_parameter_profile,
    reject_parameter_profile,
)
from omega.trace.store import TraceStore

_CONFIRMS = {
    "confirm_backtest_parity": True,
    "confirm_clv_non_regression": True,
    "parity_evidence": {"state": "PASS"},
    "clv_evidence": {"verdict": "PASS"},
}


def _tmp_db() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def _store() -> TraceStore:
    return TraceStore(db_path=_tmp_db())


def _profile(version: int = 1, **overrides) -> BackendParameterProfile:
    params = overrides.pop("params", {"rho": -0.11, "home_advantage": 0.0})
    backend = overrides.pop("backend_name", "soccer_bivariate_poisson_dc")
    bucket = overrides.pop("competition_bucket", "FIFA_INTL")
    defaults = {
        "profile_id": make_parameter_profile_id(backend, bucket, version, params),
        "version": version,
        "backend_name": backend,
        "backend_component_version": "soccer_bvp_dc_v1",
        "competition_bucket": bucket,
        "params": params,
        "dataset_hash": "abc123",
        "sample_size": 200,
        "metrics": {"brier_score": 0.21, "calibration_error": 0.04, "log_loss": 0.62},
    }
    defaults.update(overrides)
    return BackendParameterProfile(**defaults)


def test_fresh_db_has_v19_table_and_version():
    store = _store()
    try:
        tables = {
            row[0]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "parameter_profiles" in tables
        assert store.schema_version() == 20
    finally:
        store.close()


def test_register_writes_candidate_and_roundtrips():
    store = _store()
    try:
        prof = _profile()
        register_parameter_profile(store, prof)
        got = get_parameter_profile(store, prof.profile_id)
        assert got is not None
        assert got.status == ParameterProfileStatus.CANDIDATE
        assert got.params == {"rho": -0.11, "home_advantage": 0.0}
        assert got.metrics["calibration_error"] == 0.04
        # No production profile exists yet -> gatherer would fail closed.
        assert get_production_parameter_profile(store, prof.backend_name, "FIFA_INTL") is None
    finally:
        store.close()


def test_register_rejects_non_candidate_status():
    store = _store()
    try:
        with pytest.raises(ValueError, match="candidates only"):
            register_parameter_profile(
                store, _profile(status=ParameterProfileStatus.PRODUCTION)
            )
    finally:
        store.close()


def test_promote_fails_closed_without_evidence():
    store = _store()
    try:
        prof = _profile()
        register_parameter_profile(store, prof)
        with pytest.raises(PromotionGateError) as exc:
            promote_parameter_profile(store, prof.profile_id)  # no confirmations
        assert "BACKTEST_PARITY" in exc.value.report.failed_gates
        assert "CLV_NON_REG" in exc.value.report.failed_gates
        # Unchanged: still a candidate, still no production.
        assert get_parameter_profile(store, prof.profile_id).status == ParameterProfileStatus.CANDIDATE
        assert get_production_parameter_profile(store, prof.backend_name, "FIFA_INTL") is None
    finally:
        store.close()


def test_promote_fails_closed_on_high_raw_ece():
    """A backend whose RAW held-out ECE is over the floor cannot promote — the
    whole point of the rail: structure must clear the floor, not calibration."""
    store = _store()
    try:
        prof = _profile(metrics={"brier_score": 0.21, "calibration_error": 0.12, "log_loss": 0.62})
        register_parameter_profile(store, prof)
        with pytest.raises(PromotionGateError) as exc:
            promote_parameter_profile(store, prof.profile_id, **_CONFIRMS)
        assert "ECE_FLOOR" in exc.value.report.failed_gates
    finally:
        store.close()


def test_promote_fails_closed_on_undersample():
    store = _store()
    try:
        prof = _profile(sample_size=20)
        register_parameter_profile(store, prof)
        with pytest.raises(PromotionGateError) as exc:
            promote_parameter_profile(store, prof.profile_id, **_CONFIRMS)
        assert "SAMPLE_SIZE" in exc.value.report.failed_gates
    finally:
        store.close()


def test_first_promotion_succeeds_and_records_report():
    store = _store()
    try:
        prof = _profile()
        register_parameter_profile(store, prof)
        report = promote_parameter_profile(store, prof.profile_id, **_CONFIRMS)
        assert report.passed
        prod = get_production_parameter_profile(store, prof.backend_name, "FIFA_INTL")
        assert prod is not None
        assert prod.profile_id == prof.profile_id
        assert prod.status == ParameterProfileStatus.PRODUCTION
        assert prod.promoted_at is not None
        assert prod.promotion_gate_report["passed"] is True
    finally:
        store.close()


def test_second_promotion_requires_brier_improvement_and_archives_incumbent():
    store = _store()
    try:
        v1 = _profile(version=1)
        register_parameter_profile(store, v1)
        promote_parameter_profile(store, v1.profile_id, **_CONFIRMS)

        # Identical brier -> no improvement -> blocked; incumbent untouched.
        v2_flat = _profile(version=2, params={"rho": -0.10, "home_advantage": 0.0})
        register_parameter_profile(store, v2_flat)
        with pytest.raises(PromotionGateError) as exc:
            promote_parameter_profile(store, v2_flat.profile_id, **_CONFIRMS)
        assert "BRIER_IMPROVES" in exc.value.report.failed_gates
        assert get_production_parameter_profile(store, v1.backend_name, "FIFA_INTL").profile_id == v1.profile_id

        # A genuinely better candidate archives the incumbent and takes over.
        v2_better = _profile(
            version=3,
            params={"rho": -0.09, "home_advantage": 0.0},
            metrics={"brier_score": 0.19, "calibration_error": 0.03, "log_loss": 0.60},
        )
        register_parameter_profile(store, v2_better)
        report = promote_parameter_profile(store, v2_better.profile_id, **_CONFIRMS)
        assert report.passed
        assert report.incumbent_id == v1.profile_id
        assert get_parameter_profile(store, v1.profile_id).status == ParameterProfileStatus.ARCHIVED
        prod = get_production_parameter_profile(store, v1.backend_name, "FIFA_INTL")
        assert prod.profile_id == v2_better.profile_id
        # Single-production invariant: exactly one production row for the key.
        n_prod = store.conn.execute(
            "SELECT COUNT(*) FROM parameter_profiles WHERE backend_name=? AND "
            "competition_bucket=? AND status='production'",
            (v1.backend_name, "FIFA_INTL"),
        ).fetchone()[0]
        assert n_prod == 1
    finally:
        store.close()


def test_promotion_is_backend_and_bucket_scoped():
    """Promoting one (backend, bucket) never touches another's production row."""
    store = _store()
    try:
        fifa = _profile(competition_bucket="FIFA_INTL")
        epl = _profile(competition_bucket="EPL")
        register_parameter_profile(store, fifa)
        register_parameter_profile(store, epl)
        promote_parameter_profile(store, fifa.profile_id, **_CONFIRMS)
        promote_parameter_profile(store, epl.profile_id, **_CONFIRMS)
        assert get_production_parameter_profile(store, fifa.backend_name, "FIFA_INTL") is not None
        assert get_production_parameter_profile(store, epl.backend_name, "EPL") is not None
    finally:
        store.close()


def test_cannot_overwrite_promoted_profile_via_register():
    store = _store()
    try:
        prof = _profile()
        register_parameter_profile(store, prof)
        promote_parameter_profile(store, prof.profile_id, **_CONFIRMS)
        # The model carries CANDIDATE by default; registering the same id again
        # must refuse because the stored row is now PRODUCTION (immutable).
        with pytest.raises(ValueError, match="immutable"):
            register_parameter_profile(store, _profile())
    finally:
        store.close()


def test_reject_moves_candidate_to_rejected():
    store = _store()
    try:
        prof = _profile()
        register_parameter_profile(store, prof)
        rejected = reject_parameter_profile(store, prof.profile_id, "raw ECE over floor")
        assert rejected.status == ParameterProfileStatus.REJECTED
        assert rejected.reject_reason == "raw ECE over floor"
        assert list_parameter_profiles(store, status=ParameterProfileStatus.REJECTED)
    finally:
        store.close()


def test_trace_ref_shape():
    prof = _profile()
    ref = prof.trace_ref()
    assert ref["backend_name"] == "soccer_bivariate_poisson_dc"
    assert ref["backend_component_version"] == "soccer_bvp_dc_v1"
    assert ref["param_profile_id"] == prof.profile_id
    assert ref["competition_bucket"] == "FIFA_INTL"
    assert "dataset_hash" in ref


# ---------------------------------------------------------------------------
# P8.0.3 — trace parameter_profile_ref provenance + unpinned audit
# ---------------------------------------------------------------------------


def _persist_trace(store: TraceStore, trace_id: str, **extra) -> None:
    blob = {
        "trace_id": trace_id,
        "run_id": "r-" + trace_id,
        "timestamp": "2026-06-18T12:00:00Z",
        "prompt": "x",
        "league": "FIFA_WORLD_CUP_2026",
        "matchup": "France @ Brazil",
        "execution_mode": "native_sim",
        "kind": "game",
    }
    blob.update(extra)
    store.persist(blob)


def test_persist_stamps_explicit_parameter_profile_ref():
    store = _store()
    try:
        ref = _profile().trace_ref()
        _persist_trace(store, "t-explicit", parameter_profile_ref=ref)
        got = get_parameter_profile_ref(store, "t-explicit")
        assert got is not None
        assert got["param_profile_id"] == ref["param_profile_id"]
        assert got["backend_name"] == "soccer_bivariate_poisson_dc"
    finally:
        store.close()


def test_persist_synthesizes_legacy_soccer_rho_ref():
    """Today's soccer rho provenance (echoed onto the sim result) becomes a
    queryable ref with no backend change."""
    store = _store()
    try:
        _persist_trace(
            store,
            "t-soccer",
            result={
                "backend_name": "soccer_bivariate_poisson_dc",
                "component_version": "soccer_bvp_dc_v1",
                "rho_profile_id": "fifa_intl_v1",
                "rho_as_of_date": "2026-06-10",
                "dc_rho": -0.11,
            },
        )
        got = get_parameter_profile_ref(store, "t-soccer")
        assert got is not None
        assert got["param_profile_id"] == "fifa_intl_v1"
        assert got["priors_as_of_date"] == "2026-06-10"
        assert got["source"] == "legacy_dc_rho"
    finally:
        store.close()


def test_trace_without_provenance_is_unpinned():
    store = _store()
    try:
        _persist_trace(store, "t-bare", result={"status": "success"})
        assert get_parameter_profile_ref(store, "t-bare") is None
        ref, event = parameter_pin_status(store, "t-bare")
        assert ref is None
        assert event["status"] == "warn"
        assert event["outputs"]["freshness"] == "unpinned"
        assert event["event_type"] == "data_provenance"
    finally:
        store.close()


def test_parameter_pin_status_ok_for_pinned_trace():
    store = _store()
    try:
        ref = _profile().trace_ref()
        _persist_trace(store, "t-pinned", parameter_profile_ref=ref)
        got, event = parameter_pin_status(store, "t-pinned")
        assert got is not None
        assert event["status"] == "ok"
        assert event["outputs"]["parameter_profile_ref"]["param_profile_id"] == ref["param_profile_id"]
    finally:
        store.close()


def test_extract_parameter_profile_ref_pure():
    """The extractor is a pure dict function (no store) — its three resolution
    tiers, exercised directly."""
    assert extract_parameter_profile_ref({}) is None
    assert extract_parameter_profile_ref({"result": {"status": "success"}}) is None
    explicit = {"param_profile_id": "p1", "backend_name": "b"}
    assert extract_parameter_profile_ref({"parameter_profile_ref": explicit}) == explicit
    legacy = extract_parameter_profile_ref(
        {"result": {"rho_profile_id": "epl_v1", "component_version": "soccer_bvp_dc_v1"}}
    )
    assert legacy["param_profile_id"] == "epl_v1"
    assert legacy["source"] == "legacy_dc_rho"


def test_reject_is_idempotent():
    store = _store()
    try:
        prof = _profile()
        register_parameter_profile(store, prof)
        first = reject_parameter_profile(store, prof.profile_id, "reason A")
        again = reject_parameter_profile(store, prof.profile_id, "reason B")
        # The original rejection record is preserved, not clobbered by the re-reject.
        assert first.reject_reason == "reason A"
        assert again.reject_reason == "reason A"
        assert again.rejected_at == first.rejected_at
    finally:
        store.close()


def test_register_clears_stale_lifecycle_stamps():
    store = _store()
    try:
        prof = _profile(promoted_at="2026-01-01T00:00:00Z", reject_reason="stale")
        register_parameter_profile(store, prof)
        got = get_parameter_profile(store, prof.profile_id)
        assert got.promoted_at is None
        assert got.rejected_at is None
        assert got.reject_reason is None
    finally:
        store.close()
