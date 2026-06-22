from omega.core.calibration.probability import apply_calibration_audited


def test_apply_calibration_audited_sport_mapping(monkeypatch):
    from omega.core.calibration import probability
    from omega.core.calibration.profiles import CalibrationProfile

    def mock_get_active_profile(league, context_slice, market):
        # Return a mock object simulating the correct registry resolution
        return CalibrationProfile(
            profile_id=f"prof_{league}_{context_slice}",
            method="isotonic",
            league=league,
            status="production",
            version=1,
            dataset_hash="x",
            params={},
            metrics={},
            context_slice=context_slice,
            training_window="365d",
            sample_size=1000,
        )

    monkeypatch.setattr(probability, "_get_active_profile", mock_get_active_profile)

    # NFL short_week context must map to short_week
    _, audit = apply_calibration_audited(0.5, league="NFL", context_hints={"thursday": True})
    assert audit["resolved_slice"] == "short_week"
    assert audit["profile_id"] == "prof_NFL_short_week"

    # Soccer cup_match mapping
    _, audit = apply_calibration_audited(0.5, league="EPL", context_hints={"knockout": True})
    assert audit["resolved_slice"] == "cup_match"
    assert audit["profile_id"] == "prof_EPL_cup_match"

    # NBA back to back
    _, audit = apply_calibration_audited(0.5, league="NBA", context_hints={"rest_days": 0})
    assert audit["resolved_slice"] == "back_to_back"
    assert audit["profile_id"] == "prof_NBA_back_to_back"
