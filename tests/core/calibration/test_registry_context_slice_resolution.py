import pytest

from omega.core.calibration.registry import CalibrationRegistry
from omega.core.calibration.profiles import CalibrationProfile


def test_registry_slice_resolution(monkeypatch):
    registry = CalibrationRegistry()

    profiles_data = {
        "profiles": [
            {
                "profile_id": "base_nba",
                "method": "isotonic",
                "league": "NBA",
                "market": "game",
                "status": "production",
                "context_slice": None,
                "version": 1,
                "dataset_hash": "aaa",
                "params": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                "metrics": {},
                "training_window": "365d",
                "sample_size": 1000
            },
            {
                "profile_id": "playoff_nba",
                "method": "isotonic",
                "league": "NBA",
                "market": "game",
                "status": "production",
                "context_slice": "playoff",
                "version": 1,
                "dataset_hash": "bbb",
                "params": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                "metrics": {},
                "training_window": "365d",
                "sample_size": 1000
            },
            {
                "profile_id": "wrong_market_nba",
                "method": "isotonic",
                "league": "NBA",
                "market": "spread",
                "status": "production",
                "context_slice": "back_to_back",
                "version": 1,
                "dataset_hash": "ccc",
                "params": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                "metrics": {},
                "training_window": "365d",
                "sample_size": 1000
            }
        ]
    }
    monkeypatch.setattr(registry, "_load", lambda: profiles_data)

    # 1. Exact slice match wins
    prof = registry.get_production("NBA", context_slice="playoff", market="game")
    assert prof is not None
    assert prof.profile_id == "playoff_nba"

    # 2. Missing slice falls back to base
    prof2 = registry.get_production("NBA", context_slice="back_to_back", market="game")
    assert prof2 is not None
    assert prof2.profile_id == "base_nba"

    # 3. Wrong-market slice cannot leak
    prof3 = registry.get_production("NBA", context_slice="back_to_back", market="spread")
    assert prof3 is not None
    assert prof3.profile_id == "wrong_market_nba"
    # Asking for game market with that slice should fall back to base
    prof4 = registry.get_production("NBA", context_slice="back_to_back", market="game")
    assert prof4 is not None
    assert prof4.profile_id == "base_nba"

    # 4. No profile means None
    assert registry.get_production("NFL", context_slice="playoff", market="game") is None
