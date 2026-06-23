from omega.ops.fit_calibration import _unique_profile_id


def test_unique_profile_id_with_slice():
    pid = _unique_profile_id(
        method="isotonic",
        league="NBA",
        version=1,
        dataset_hash="1234567890abcdef",
        market="game",
        context_slice="playoff",
    )
    assert pid == "iso_nba_playoff_v1_1234567890abcdef"

    pid_base = _unique_profile_id(
        method="shrinkage",
        league="EPL",
        version=2,
        dataset_hash="abcdef",
        market="draw",
        context_slice=None,
    )
    assert pid_base == "shrink_epl_draw_v2_abcdef"
