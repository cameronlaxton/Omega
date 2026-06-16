
from omega.core.calibration.profiles import CalibrationProfile
from omega.core.calibration.registry import CalibrationRegistry


def test_gating_incumbent_slice_isolation(monkeypatch):
    registry = CalibrationRegistry()

    # We have a base profile, but NO playoff profile
    profiles = [
        CalibrationProfile(
            profile_id="base_nba",
            method="isotonic",
            league="NBA",
            market="game",
            status="production",
            context_slice=None,
            version=1,
            dataset_hash="x",
            training_window="365d",
            sample_size=1000,
            params={"x": [0,1], "y": [0,1]},
            metrics={}
        ),
        CalibrationProfile(
            profile_id="base_nba_spread",
            method="isotonic",
            league="NBA",
            market="spread",
            status="production",
            context_slice=None,
            version=1,
            dataset_hash="x",
            training_window="365d",
            sample_size=1000,
            params={"x": [0,1], "y": [0,1]},
            metrics={}
        )
    ]
    monkeypatch.setattr(registry, "list_profiles", lambda **kwargs: [p for p in profiles if kwargs.get("status") == "production" and kwargs.get("league") == "NBA"])

    # Playoff candidate should NOT gate against base
    candidate_playoff = CalibrationProfile(
        profile_id="playoff_cand",
        method="isotonic",
        league="NBA",
        market="game",
        status="candidate",
        context_slice="playoff",
        version=2,
        dataset_hash="y",
        training_window="365d",
        sample_size=1000,
        params={},
        metrics={}
    )
    incumbent = registry.gating_incumbent(candidate_playoff)
    assert incumbent is None

    # Base candidate SHOULD gate against base
    candidate_base = CalibrationProfile(
        profile_id="base_cand",
        method="isotonic",
        league="NBA",
        market="game",
        status="candidate",
        context_slice=None,
        version=2,
        dataset_hash="z",
        training_window="365d",
        sample_size=1000,
        params={},
        metrics={}
    )
    incumbent2 = registry.gating_incumbent(candidate_base)
    assert incumbent2 is not None
    assert incumbent2.profile_id == "base_nba"

    candidate_spread = CalibrationProfile(
        profile_id="spread_cand",
        method="isotonic",
        league="NBA",
        market="spread",
        status="candidate",
        context_slice="playoff",
        version=2,
        dataset_hash="spread",
        training_window="365d",
        sample_size=1000,
        params={},
        metrics={}
    )
    assert registry.gating_incumbent(candidate_spread) is None
