"""As-of league-mean index used for empirical-Bayes rating shrinkage.

The index must be leak-safe: the league mean at a decision time may only reflect
games strictly *before* it (same-timestamp games excluded), so a snapshot can
never embed a same-slate or future result via the baseline.
"""

from __future__ import annotations

from omega.historical.contracts import HistoricalEvent, HistoricalOutcome
from omega.historical.replay import as_of_league_mean, build_league_rating_index


def _ev(eid: str, date: str) -> HistoricalEvent:
    return HistoricalEvent(
        event_id=eid,
        league="WORLD_CUP",
        sport_family="soccer",
        start_time=date,
        home_team="A",
        away_team="B",
        source_name="t",
    )


def _oc(eid: str, hs: int, as_: int) -> HistoricalOutcome:
    return HistoricalOutcome(event_id=eid, home_score=hs, away_score=as_, result="home_win")


def test_as_of_league_mean_excludes_current_and_future():
    events = [
        _ev("e1", "2023-01-01T00:00:00+00:00"),  # scores 2,0
        _ev("e2", "2023-02-01T00:00:00+00:00"),  # scores 4,2
        _ev("e3", "2023-03-01T00:00:00+00:00"),  # scores 6,0 (future relative to e2)
    ]
    outcomes = {
        "e1": _oc("e1", 2, 0),
        "e2": _oc("e2", 4, 2),
        "e3": _oc("e3", 6, 0),
    }
    index = build_league_rating_index(events, outcomes)

    # Before any game: no prior data.
    assert as_of_league_mean(index, "2022-12-31T00:00:00+00:00") is None
    # At e1's timestamp: e1 itself is excluded (bisect_left) → still no prior data.
    assert as_of_league_mean(index, "2023-01-01T00:00:00+00:00") is None
    # At e2's timestamp: only e1's two scores (2, 0) → mean 1.0. e2/e3 excluded.
    assert as_of_league_mean(index, "2023-02-01T00:00:00+00:00") == 1.0
    # At e3's timestamp: e1 (2,0) + e2 (4,2) = 8 over 4 entries → mean 2.0. e3 excluded.
    assert as_of_league_mean(index, "2023-03-01T00:00:00+00:00") == 2.0
    # After all games: 2+0+4+2+6+0 = 14 over 6 → mean ~2.333.
    assert abs(as_of_league_mean(index, "2023-04-01T00:00:00+00:00") - 14 / 6) < 1e-9


def test_empty_index_returns_none():
    index = build_league_rating_index([], {})
    assert as_of_league_mean(index, "2023-01-01T00:00:00+00:00") is None
