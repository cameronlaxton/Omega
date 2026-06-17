"""Tests for calibration promotion market awareness (gating-incumbent lookup)."""

from __future__ import annotations

import os
import tempfile

from omega.core.calibration.profiles import CalibrationProfile
from omega.core.calibration.registry import CalibrationRegistry


def _profile(profile_id: str, market: str, **overrides) -> CalibrationProfile:
    defaults = {
        "profile_id": profile_id,
        "version": 1,
        "method": "shrinkage",
        "league": "NBA",
        "market": market,
        "params": {"shrink_factor": 0.6},
        "training_window": "2025-01-01/2025-06-30",
        "sample_size": 200,
        "dataset_hash": "abc123",
        "metrics": {"brier_score": 0.22, "calibration_error": 0.04, "log_loss": 0.65},
    }
    defaults.update(overrides)
    return CalibrationProfile(**defaults)


def test_exact_incumbent_does_not_cross_markets():
    """A prop candidate must be gated against a prop incumbent (or None), never
    against the game production -- get_production() would wrongly fall back."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        os.unlink(path)
        reg = CalibrationRegistry(path=path)
        reg.register(_profile("nba_game_v1", "game"))
        reg._apply_promotion("nba_game_v1")  # game production exists

        prop_candidate = _profile("nba_prop_v1", "prop")
        # get_production would fall back to the game profile for market="prop":
        assert reg.get_production("NBA", market="prop").profile_id == "nba_game_v1"
        # The gating-incumbent lookup must return None (no prop production), never
        # crossing markets to the game profile.
        assert reg.gating_incumbent(prop_candidate) is None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_exact_incumbent_matches_same_market():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        os.unlink(path)
        reg = CalibrationRegistry(path=path)
        reg.register(_profile("nba_prop_v1", "prop"))
        reg._apply_promotion("nba_prop_v1")

        challenger = _profile("nba_prop_v2", "prop", version=2)
        incumbent = reg.gating_incumbent(challenger)
        assert incumbent is not None
        assert incumbent.profile_id == "nba_prop_v1"
    finally:
        if os.path.exists(path):
            os.unlink(path)
