"""P8.3 — omega-fit-calibration records the backend binding, fail-closed on mixes.

New calibration candidates declare the raw-probability substrate of their source
traces (backend name/component version + governed param_profile_id, read
strictly from recorded trace provenance). A dataset mixing substrates is
refused — there is no legacy bypass. Prop-plane profiles produced by the P8.5
prop structural sweep (``BackendParameterProfile.trace_ref()`` echoed onto prop
traces) bind and are consumed by the runtime match.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fit_calibration  # type: ignore  # noqa: E402

from omega.core.calibration.fitter import CalibrationFitter  # noqa: E402
from omega.core.calibration.probability import (  # noqa: E402
    apply_calibration_audited,
    calibration_registry_override,
)
from omega.core.calibration.profiles import (  # noqa: E402
    BindingStatus,
    ProfileMaturity,
    ProfileStatus,
)
from omega.core.calibration.registry import CalibrationRegistry  # noqa: E402
from omega.core.simulation.parameter_profile import BackendParameterProfile  # noqa: E402

_PROP_REF = {
    "backend_name": "prop_neg_binom",
    "backend_component_version": "prop_nb_v1",
    "param_profile_id": "prop_neg_binom__NFL__RUSHING_YARDS__v2",
    "competition_bucket": "NFL__RUSHING_YARDS",
}


def _tmp_registry_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    os.unlink(tmp.name)
    return tmp.name


def _game_trace(prob: float, result: str, version: str = "fs_v1", ref: dict | None = None):
    trace = {
        "predictions": {"home_win_prob": prob},
        "_outcome": {"result": result},
        "result": {"simulation_backend": "fast_score", "component_version": version},
    }
    if ref is not None:
        trace["result"]["parameter_profile_ref"] = ref
    return trace


def _prop_trace(over_prob: float, results: list[str], ref: dict | None = None):
    trace = {
        "predictions": {"over_prob": over_prob, "under_prob": 1.0 - over_prob},
        "_prop_outcomes": [{"side": "over", "result": r} for r in results],
        "result": {},
    }
    if ref is not None:
        trace["result"]["parameter_profile_ref"] = ref
    return trace


class TestBindingForTraces:
    def test_homogeneous_game_traces_bind(self):
        traces = [_game_trace(0.6, "home_win"), _game_trace(0.4, "away_win")]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "game")
        assert err is None
        assert binding is not None
        assert binding.backend_name == "fast_score"
        assert binding.backend_component_version == "fs_v1"
        assert binding.param_profile_id is None

    def test_game_traces_bind_request_snapshot_parameter_profile_ref(self):
        trace = _game_trace(0.6, "home_win", ref=None)
        trace["input_snapshot"] = {"prior_payload": {"parameter_profile_ref": _PROP_REF}}
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), [trace], "game")
        assert err is None
        assert binding is not None
        assert binding.backend_name == "fast_score"
        assert binding.backend_component_version == "fs_v1"
        assert binding.param_profile_id == _PROP_REF["param_profile_id"]

    def test_homogeneous_pinned_prop_traces_bind_to_param_profile(self):
        traces = [
            _prop_trace(0.6, ["win", "loss"], ref=_PROP_REF),
            _prop_trace(0.55, ["win"], ref=_PROP_REF),
        ]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "prop")
        assert err is None
        assert binding is not None
        assert binding.backend_name == "prop_neg_binom"
        assert binding.backend_component_version == "prop_nb_v1"
        assert binding.param_profile_id == _PROP_REF["param_profile_id"]

    def test_mixed_param_profile_ids_fail_closed(self):
        other = {**_PROP_REF, "param_profile_id": "prop_neg_binom__NFL__RUSHING_YARDS__v3"}
        traces = [
            _prop_trace(0.6, ["win"], ref=_PROP_REF),
            _prop_trace(0.5, ["loss"], ref=other),
        ]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "prop")
        assert binding is None
        assert err is not None and "mixes 2 raw-probability substrates" in err

    def test_mixed_component_versions_fail_closed(self):
        traces = [
            _game_trace(0.6, "home_win", version="fs_v1"),
            _game_trace(0.4, "away_win", version="fs_v2"),
        ]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "game")
        assert binding is None
        assert err is not None

    def test_pinned_plus_unpinned_mix_fails_closed(self):
        traces = [
            _prop_trace(0.6, ["win"], ref=_PROP_REF),
            _prop_trace(0.5, ["loss"]),  # no provenance at all
        ]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "prop")
        assert binding is None
        assert err is not None

    def test_all_unpinned_is_explicit_unpinned_binding(self):
        traces = [_prop_trace(0.6, ["win"]), _prop_trace(0.5, ["loss"])]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "prop")
        assert err is None
        assert binding is not None
        assert not binding.is_pinned()  # declared unpinned, not legacy

    def test_non_contributing_traces_constrain_nothing(self):
        # A game trace (different substrate) contributes no prop pairs, so a
        # prop-plane fit ignores it instead of failing closed on a phantom mix.
        traces = [
            _prop_trace(0.6, ["win"], ref=_PROP_REF),
            _game_trace(0.5, "home_win", version="fs_v9"),
        ]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "prop")
        assert err is None
        assert binding is not None
        assert binding.param_profile_id == _PROP_REF["param_profile_id"]


class TestFitRegistersBinding:
    def _fit(self, registry: CalibrationRegistry, binding):
        train_p = [round(0.2 + (i % 7) * 0.1, 2) for i in range(40)]
        train_o = [1 if p >= 0.5 else 0 for p in train_p]
        return fit_calibration.fit_and_register(
            CalibrationFitter(),
            registry,
            "NFL",
            "shrinkage",
            train_p,
            train_o,
            [0.35, 0.6, 0.5, 0.7, 0.45],
            [0, 1, 1, 1, 0],
            dry_run=False,
            market="prop",
            backend_binding=binding,
        )

    def test_candidate_carries_binding_and_round_trips(self):
        traces = [_prop_trace(0.6, ["win", "loss"], ref=_PROP_REF)]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "prop")
        assert err is None
        path = _tmp_registry_path()
        registry = CalibrationRegistry(path=path)
        profile = self._fit(registry, binding)
        assert profile.binding_status() == BindingStatus.BOUND
        loaded = CalibrationRegistry(path=path).get_profile(profile.profile_id)
        assert loaded is not None
        assert loaded.backend_binding is not None
        assert loaded.backend_binding.param_profile_id == _PROP_REF["param_profile_id"]

    def test_candidate_without_binding_is_legacy(self):
        path = _tmp_registry_path()
        profile = self._fit(CalibrationRegistry(path=path), None)
        assert profile.binding_status() == BindingStatus.LEGACY


class TestPropSweepProfilesBindAndConsume:
    """End-to-end P8.5 -> P8.3: a governed prop BackendParameterProfile's
    trace_ref, echoed onto graded prop traces, binds the calibration fit; the
    runtime then applies the profile only on the matching live substrate."""

    def _param_profile(self, version: int = 2) -> BackendParameterProfile:
        return BackendParameterProfile(
            profile_id=f"prop_neg_binom__NFL__RUSHING_YARDS__v{version}",
            version=version,
            backend_name="prop_neg_binom",
            backend_component_version="prop_nb_v1",
            competition_bucket="NFL__RUSHING_YARDS",
            params={"nb_k_scale": 1.5},
            dataset_hash="h",
            sample_size=150,
            metrics={"calibration_error": 0.03},
        )

    def test_sweep_profile_ref_binds_and_runtime_consumes(self):
        prof = self._param_profile(version=2)
        ref = prof.trace_ref()
        traces = [_prop_trace(0.55 + 0.01 * i, ["win", "loss"], ref=ref) for i in range(25)]
        binding, err = fit_calibration.binding_for_traces(CalibrationFitter(), traces, "prop")
        assert err is None
        assert binding is not None
        assert binding.param_profile_id == prof.profile_id

        # Register the bound calibration profile as production (isolated registry).
        path = _tmp_registry_path()
        registry = CalibrationRegistry(path=path)
        cal_profile = fit_calibration.fit_and_register(
            CalibrationFitter(),
            registry,
            "NFL",
            "shrinkage",
            [round(0.2 + (i % 7) * 0.1, 2) for i in range(40)],
            [1 if round(0.2 + (i % 7) * 0.1, 2) >= 0.5 else 0 for i in range(40)],
            [0.35, 0.6, 0.5, 0.7, 0.45],
            [0, 1, 1, 1, 0],
            dry_run=False,
            market="prop",
            backend_binding=binding,
        )
        data = registry._load()
        for p in data["profiles"]:
            if p["profile_id"] == cal_profile.profile_id:
                p["status"] = ProfileStatus.PRODUCTION.value
                p["maturity"] = ProfileMaturity.PRODUCTION.value
        registry._save(data)

        # Live substrate exactly as analyze_player_prop builds it: the resolved
        # backend's identity + the param profile ref the backend echoed.
        live = {
            "backend_name": "prop_neg_binom",
            "backend_component_version": "prop_nb_v1",
            "param_profile_id": prof.profile_id,
        }
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(
                0.80, league="NFL", market="prop", substrate_ref=live
            )
        assert audit["profile_id"] == cal_profile.profile_id
        assert audit["binding_status"] == "bound"

        # A NEWLY promoted param profile (v3) changes the substrate: the stale
        # calibration profile is skipped, never silently applied.
        live_v3 = {**live, "param_profile_id": self._param_profile(version=3).profile_id}
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(
                0.80, league="NFL", market="prop", substrate_ref=live_v3
            )
        assert audit["profile_id"] is None
        assert cal_profile.profile_id in audit["binding_mismatch"]


class TestMainFailsClosedOnMixedSubstrate:
    def _run_main(self, monkeypatch, traces) -> tuple[int, CalibrationRegistry]:
        path = _tmp_registry_path()

        def _registry_factory(*args, **kwargs):
            return CalibrationRegistry(path=path)

        monkeypatch.setattr(fit_calibration, "CalibrationRegistry", _registry_factory)
        monkeypatch.setattr(fit_calibration, "_load_graded_traces", lambda args: traces)
        rc = fit_calibration.main(
            ["--league", "NFL", "--plane", "prop", "--method", "shrinkage"]
        )
        return rc, CalibrationRegistry(path=path)

    def test_mixed_dataset_registers_nothing_and_exits_nonzero(self, monkeypatch):
        other = {**_PROP_REF, "param_profile_id": "prop_neg_binom__NFL__RUSHING_YARDS__v3"}
        traces = [
            _prop_trace(0.4 + 0.01 * (i % 30), ["win" if i % 2 else "loss"], ref=_PROP_REF)
            for i in range(25)
        ] + [
            _prop_trace(0.4 + 0.01 * (i % 30), ["win" if i % 2 else "loss"], ref=other)
            for i in range(25)
        ]
        rc, registry = self._run_main(monkeypatch, traces)
        assert rc == 1
        assert registry.list_profiles(league="NFL") == []

    def test_homogeneous_dataset_registers_bound_candidate(self, monkeypatch):
        traces = [
            _prop_trace(0.4 + 0.01 * (i % 30), ["win" if i % 2 else "loss"], ref=_PROP_REF)
            for i in range(50)
        ]
        rc, registry = self._run_main(monkeypatch, traces)
        assert rc == 0
        profiles = registry.list_profiles(league="NFL")
        assert len(profiles) == 1
        assert profiles[0].binding_status() == BindingStatus.BOUND
        assert profiles[0].backend_binding.param_profile_id == _PROP_REF["param_profile_id"]
