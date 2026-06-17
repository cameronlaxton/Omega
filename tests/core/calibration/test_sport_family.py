import pytest

from omega.core.calibration.sport_family import sport_family_for_league


def test_sport_family_for_league():
    # Basketball
    assert sport_family_for_league("NBA") == "basketball"
    assert sport_family_for_league("WNBA") == "basketball"

    # American Football
    assert sport_family_for_league("NFL") == "american_football"
    assert sport_family_for_league("NCAAF") == "american_football"

    # Soccer
    assert sport_family_for_league("MLS") == "soccer"
    assert sport_family_for_league("EPL") == "soccer"
    assert sport_family_for_league("WORLD_CUP") == "soccer"

    # Tennis
    assert sport_family_for_league("ATP") == "tennis"
    assert sport_family_for_league("WTA") == "tennis"
    assert sport_family_for_league("GRAND_SLAM") == "tennis"

    # Baseball
    assert sport_family_for_league("MLB") == "baseball"
    assert sport_family_for_league("KBO") == "baseball"

    # Hockey
    assert sport_family_for_league("NHL") == "hockey"

    # Unknown/Fallback
    assert sport_family_for_league("UNKNOWN_LEAGUE") == "unknown"
