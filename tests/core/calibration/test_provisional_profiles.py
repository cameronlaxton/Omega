"""Provisional calibration + hierarchical fallback (Issue 6 + remediation proofs).

Proves the all-or-nothing trap is gone: a thin profile applies a small, capped
correction at provisional maturity (and caps confidence), and selection walks a
hierarchical fallback.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from omega.core.calibration.probability import (
    apply_calibration_audited,
    calibration_registry_override,
)
from omega.core.calibration.profiles import (
    CalibrationProfile,
    ProfileMaturity,
    ProfileStatus,
)
from omega.core.calibration.registry import CalibrationRegistry


def _tmp_registry_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    os.unlink(tmp.name)
    return tmp.name


def _profile(profile_id, league, maturity, **kw) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=profile_id,
        version=1,
        method="shrinkage",
        league=league,
        market="game",
        params={"shrink_factor": 0.5},
        training_window="2025",
        sample_size=kw.get("sample_size", 60),
        dataset_hash="hash",
        metrics={"calibration_error": kw.get("ece", 0.08), "brier_score": 0.20},
        status=ProfileStatus.PRODUCTION,
        maturity=maturity,
    )


class TestProvisionalDamping:
    def test_provisional_correction_is_bounded(self):
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        reg.register(_profile("prov_nba", "NBA", ProfileMaturity.PROVISIONAL))
        with calibration_registry_override(path):
            cal, audit = apply_calibration_audited(0.80, league="NBA", market="game")
        # The probability moved (escaped static_identity)...
        assert cal != 0.80
        # ...but only by the small provisional bound (<= 0.03).
        assert abs(cal - 0.80) <= 0.03 + 1e-9
        assert audit["maturity"] == "provisional"
        assert audit["maturity_damped"] is True

    def test_production_correction_is_full(self):
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        reg.register(_profile("prod_nba", "NBA", ProfileMaturity.PRODUCTION, sample_size=500))
        with calibration_registry_override(path):
            cal, audit = apply_calibration_audited(0.80, league="NBA", market="game")
        # Full shrink toward 0.5 -> a large move, undamped.
        assert abs(cal - 0.80) > 0.03
        assert audit["maturity"] == "production"
        assert audit["maturity_damped"] is False


class TestHierarchicalFallback:
    def test_league_level_wins(self):
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        reg.register(_profile("nba", "NBA", ProfileMaturity.PROVISIONAL))
        reg.register(_profile("bball", "BASKETBALL", ProfileMaturity.PRODUCTION))
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(0.80, league="NBA", market="game")
        assert audit["fallback_level"] == "league"
        assert audit["profile_id"] == "nba"

    def test_sport_family_fallback(self):
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        reg.register(_profile("bball", "BASKETBALL", ProfileMaturity.PROBATION))
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(0.80, league="NBA", market="game")
        assert audit["fallback_level"] == "sport_family"
        assert audit["profile_id"] == "bball"

    def test_global_fallback(self):
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        reg.register(_profile("glob", "GLOBAL", ProfileMaturity.PROVISIONAL))
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(0.80, league="NBA", market="game")
        assert audit["fallback_level"] == "global"

    def test_no_profile_is_static(self):
        path = _tmp_registry_path()
        CalibrationRegistry(path=path)  # empty registry
        with calibration_registry_override(path):
            _cal, audit = apply_calibration_audited(0.95, league="NBA", market="game")
        assert audit["fallback_level"] is None
        assert audit["path"] in ("static_identity", "static_calibrated")


class TestLegacyMaturityDerivation:
    def test_legacy_production_profile_is_production_maturity(self):
        p = _profile("legacy", "NBA", maturity=None)  # field omitted
        p = p.model_copy(update={"maturity": None})
        assert p.effective_maturity() == ProfileMaturity.PRODUCTION


class TestProvisionalPromotion:
    def test_promote_provisional_activates_with_capped_trust(self):
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        cand = _profile("c1", "NBA", maturity=None)
        cand = cand.model_copy(update={"status": ProfileStatus.CANDIDATE, "maturity": None})
        reg.register(cand)
        reg.promote_provisional("c1")
        prod = reg.get_production("NBA")
        assert prod is not None
        assert prod.status == ProfileStatus.PRODUCTION
        assert prod.effective_maturity() == ProfileMaturity.PROVISIONAL

    def test_under_sampled_candidate_rejected(self):
        path = _tmp_registry_path()
        reg = CalibrationRegistry(path=path)
        cand = _profile("c2", "NHL", maturity=None, sample_size=10)
        cand = cand.model_copy(update={"status": ProfileStatus.CANDIDATE, "maturity": None})
        reg.register(cand)
        with pytest.raises(ValueError):
            reg.promote_provisional("c2")
