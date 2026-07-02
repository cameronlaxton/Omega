"""P8.3 — calibration/backend binding at runtime selection.

A CalibrationProfile now declares the raw-probability substrate it was fit
against (backend name + component version + governed param_profile_id). The
shared selection walk applies a BOUND profile only when the live substrate
matches, and fails closed (skip -> next rung -> static policy, with the
mismatch surfaced in the audit) when it does not — a profile fit on one
backend parameter profile is never silently applied to probabilities produced
by another. Legacy (pre-P8.3) and explicitly-unpinned profiles keep their
historical apply-anywhere behavior, but are flagged in the audit.
"""

from __future__ import annotations

import os
import tempfile

from omega.core.calibration.probability import (
    apply_calibration_audited,
    calibration_registry_override,
)
from omega.core.calibration.profiles import (
    BindingStatus,
    CalibrationBackendBinding,
    CalibrationProfile,
    ProfileMaturity,
    ProfileStatus,
)
from omega.core.calibration.registry import CalibrationRegistry

_SUBSTRATE = {
    "backend_name": "prop_neg_binom",
    "backend_component_version": "prop_nb_v1",
    "param_profile_id": "prop_neg_binom__NFL__RUSHING_YARDS__v2",
}


def _tmp_registry_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    os.unlink(tmp.name)
    return tmp.name


def _profile(
    profile_id: str,
    league: str,
    market: str = "game",
    binding: CalibrationBackendBinding | None = None,
) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=profile_id,
        version=1,
        method="shrinkage",
        league=league,
        market=market,
        params={"shrink_factor": 0.5},
        training_window="2026",
        sample_size=500,
        dataset_hash="hash",
        metrics={"calibration_error": 0.03, "brier_score": 0.20},
        status=ProfileStatus.PRODUCTION,
        maturity=ProfileMaturity.PRODUCTION,
        backend_binding=binding,
    )


class TestBindingModel:
    def test_legacy_profile_has_legacy_status(self):
        assert _profile("p", "NBA").binding_status() == BindingStatus.LEGACY

    def test_all_none_binding_is_unpinned(self):
        p = _profile("p", "NBA", binding=CalibrationBackendBinding())
        assert p.binding_status() == BindingStatus.UNPINNED

    def test_any_field_makes_it_bound(self):
        p = _profile(
            "p", "NBA", binding=CalibrationBackendBinding(backend_name="fast_score")
        )
        assert p.binding_status() == BindingStatus.BOUND

    def test_unpinned_binding_matches_anything(self):
        b = CalibrationBackendBinding()
        assert b.mismatch_reason(None) is None
        assert b.mismatch_reason(_SUBSTRATE) is None

    def test_bound_binding_fails_closed_on_unknown_substrate(self):
        b = CalibrationBackendBinding(**_SUBSTRATE)
        assert b.mismatch_reason(None) == "substrate_unknown"

    def test_bound_binding_fails_closed_on_missing_field(self):
        b = CalibrationBackendBinding(**_SUBSTRATE)
        live = {**_SUBSTRATE, "param_profile_id": None}
        assert b.mismatch_reason(live) == "substrate_missing:param_profile_id"

    def test_bound_binding_rejects_different_value(self):
        b = CalibrationBackendBinding(**_SUBSTRATE)
        live = {**_SUBSTRATE, "param_profile_id": "prop_neg_binom__NFL__RUSHING_YARDS__v3"}
        reason = b.mismatch_reason(live)
        assert reason is not None and reason.startswith("param_profile_id_mismatch")

    def test_bound_binding_accepts_exact_match(self):
        b = CalibrationBackendBinding(**_SUBSTRATE)
        assert b.mismatch_reason(dict(_SUBSTRATE)) is None

    def test_none_binding_fields_constrain_nothing(self):
        # Bound only by backend_name; a different param profile is irrelevant.
        b = CalibrationBackendBinding(backend_name="prop_neg_binom")
        live = {**_SUBSTRATE, "param_profile_id": "something_else"}
        assert b.mismatch_reason(live) is None

    def test_binding_round_trips_through_registry(self):
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        reg.register(_profile("rt", "NBA", binding=CalibrationBackendBinding(**_SUBSTRATE)))
        loaded = CalibrationRegistry(path=path).get_profile("rt")
        assert loaded is not None
        assert loaded.binding_status() == BindingStatus.BOUND
        assert loaded.backend_binding is not None
        assert loaded.backend_binding.param_profile_id == _SUBSTRATE["param_profile_id"]


class TestRuntimeSelection:
    def test_matching_substrate_applies_bound_profile(self):
        path = _tmp_registry_path()
        CalibrationRegistry(path=path).register(
            _profile("bound_nfl", "NFL", market="prop",
                     binding=CalibrationBackendBinding(**_SUBSTRATE))
        )
        with calibration_registry_override(path):
            cal, audit = apply_calibration_audited(
                0.80, league="NFL", market="prop", substrate_ref=dict(_SUBSTRATE)
            )
        assert audit["path"] == "profile"
        assert audit["profile_id"] == "bound_nfl"
        assert audit["binding_status"] == "bound"
        assert audit["binding_mismatch"] is None
        assert cal != 0.80  # the correction actually applied

    def test_mismatched_substrate_skips_bound_profile(self):
        path = _tmp_registry_path()
        CalibrationRegistry(path=path).register(
            _profile("bound_nfl", "NFL", market="prop",
                     binding=CalibrationBackendBinding(**_SUBSTRATE))
        )
        live = {**_SUBSTRATE, "param_profile_id": "prop_neg_binom__NFL__RUSHING_YARDS__v9"}
        with calibration_registry_override(path):
            cal, audit = apply_calibration_audited(
                0.80, league="NFL", market="prop", substrate_ref=live
            )
        # Fail closed: no profile applied, static policy took over, and the
        # audit says exactly which profile was skipped and why.
        assert audit["profile_id"] is None
        assert audit["path"] in ("static_identity", "static_calibrated")
        assert audit["binding_mismatch"] is not None
        assert "bound_nfl" in audit["binding_mismatch"]
        assert "param_profile_id_mismatch" in audit["binding_mismatch"]

    def test_unknown_substrate_skips_bound_profile(self):
        path = _tmp_registry_path()
        CalibrationRegistry(path=path).register(
            _profile("bound_nfl", "NFL", market="prop",
                     binding=CalibrationBackendBinding(**_SUBSTRATE))
        )
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(
                0.80, league="NFL", market="prop", substrate_ref=None
            )
        assert audit["profile_id"] is None
        assert "substrate_unknown" in (audit["binding_mismatch"] or "")

    def test_legacy_profile_applies_and_is_flagged(self):
        path = _tmp_registry_path()
        CalibrationRegistry(path=path).register(_profile("legacy_nba", "NBA"))
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(
                0.80, league="NBA", market="game", substrate_ref=dict(_SUBSTRATE)
            )
        assert audit["profile_id"] == "legacy_nba"
        assert audit["binding_status"] == "legacy"
        assert audit["binding_mismatch"] is None

    def test_unpinned_profile_applies_and_is_flagged(self):
        path = _tmp_registry_path()
        CalibrationRegistry(path=path).register(
            _profile("unpinned_nba", "NBA", binding=CalibrationBackendBinding())
        )
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(
                0.80, league="NBA", market="game", substrate_ref=dict(_SUBSTRATE)
            )
        assert audit["profile_id"] == "unpinned_nba"
        assert audit["binding_status"] == "unpinned"

    def test_walk_continues_past_mismatched_league_profile(self):
        # League profile is bound to another substrate; the sport-family profile
        # is legacy — the walk skips the league rung (recording why) and applies
        # the family profile, exactly like the profile-absent fallback.
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        reg.register(
            _profile("bound_nba", "NBA",
                     binding=CalibrationBackendBinding(param_profile_id="other_params"))
        )
        reg.register(_profile("bball", "BASKETBALL"))
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(
                0.80, league="NBA", market="game", substrate_ref=dict(_SUBSTRATE)
            )
        assert audit["profile_id"] == "bball"
        assert audit["fallback_level"] == "sport_family"
        assert "bound_nba" in audit["binding_mismatch"]

    def test_service_threads_prop_substrate_match(self):
        # analyze_player_prop builds the live substrate from the resolved prop
        # backend's identity + the governed ref the backend echoed; a bound prop
        # profile applies and the audit says so.
        from omega.core.contracts.schemas import PlayerPropRequest
        from omega.core.contracts.service import analyze_player_prop

        path = _tmp_registry_path()
        CalibrationRegistry(path=path).register(
            _profile("bound_prop_nfl", "NFL", market="prop",
                     binding=CalibrationBackendBinding(**_SUBSTRATE))
        )
        request = PlayerPropRequest(
            player_name="Saquon Barkley",
            league="NFL",
            prop_type="rushing_yards",
            line=82.5,
            game_date="2026-09-10",
            home_team="Eagles",
            away_team="Cowboys",
            n_iterations=2000,
            seed=7,
            odds_over=-110,
            odds_under=-110,
            game_context={"is_playoff": False, "rest_days": 7},
            player_context={
                "rushing_yards_mean": 90.0,
                "rushing_yards_std": 35.0,
                "nb_dispersion_k": 2.0,
                "parameter_profile_ref": {
                    "param_profile_id": _SUBSTRATE["param_profile_id"],
                    "backend_name": "prop_neg_binom",
                },
            },
        )
        with calibration_registry_override(path):
            resp = analyze_player_prop(request)
        assert resp.status == "success"
        assert resp.over_calibration_audit is not None
        assert resp.over_calibration_audit.profile_id == "bound_prop_nfl"
        assert resp.over_calibration_audit.binding_status == "bound"

        # Same request but the backend echoes a DIFFERENT param profile: the
        # bound profile is skipped and the mismatch is on the audit surface.
        request_v3 = request.model_copy(deep=True)
        request_v3.player_context["parameter_profile_ref"] = {
            "param_profile_id": "prop_neg_binom__NFL__RUSHING_YARDS__v9",
            "backend_name": "prop_neg_binom",
        }
        with calibration_registry_override(path):
            resp = analyze_player_prop(request_v3)
        assert resp.status == "success"
        assert resp.over_calibration_audit.profile_id is None
        assert "bound_prop_nfl" in (resp.over_calibration_audit.binding_mismatch or "")

    def test_service_threads_game_substrate_mismatch(self):
        # analyze_game passes the live sim's backend identity; a game profile
        # bound to a different backend never calibrates these edges.
        from omega.core.contracts.schemas import GameAnalysisRequest, OddsInput
        from omega.core.contracts.service import analyze_game

        path = _tmp_registry_path()
        CalibrationRegistry(path=path).register(
            _profile("bound_other_backend", "NBA",
                     binding=CalibrationBackendBinding(backend_name="not_the_live_backend"))
        )
        request = GameAnalysisRequest(
            home_team="Celtics",
            away_team="Pacers",
            league="NBA",
            n_iterations=500,
            odds=OddsInput(moneyline_home=-150, moneyline_away=130),
            game_context={"is_playoff": False, "rest_days": 2},
            home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            away_context={"off_rating": 112.0, "def_rating": 112.0, "pace": 98.0},
        )
        with calibration_registry_override(path):
            resp = analyze_game(request)
        assert resp.status == "success"
        assert resp.edges
        audit = resp.edges[0].calibration_audit
        assert audit.profile_id is None
        assert "bound_other_backend" in (audit.binding_mismatch or "")
        assert "backend_name_mismatch" in audit.binding_mismatch

    def test_markets_stay_isolated_under_binding_mismatch(self):
        # The game-market profile mismatches the live substrate. The walk must
        # fall to the static policy — never to the same league's prop or draw
        # profile (no cross-market authorization is introduced by P8.3).
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        reg.register(
            _profile("bound_game", "NFL", market="game",
                     binding=CalibrationBackendBinding(param_profile_id="other_params"))
        )
        reg.register(_profile("prop_nfl", "NFL", market="prop"))
        reg.register(_profile("draw_nfl", "NFL", market="draw"))
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(
                0.80, league="NFL", market="game", substrate_ref=dict(_SUBSTRATE)
            )
        assert audit["profile_id"] is None
        assert audit["path"] in ("static_identity", "static_calibrated")
        assert "bound_game" in audit["binding_mismatch"]
