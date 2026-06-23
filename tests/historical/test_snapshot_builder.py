"""Snapshot builder: as-of safety, required keys, slices, provenance, hashing."""

from __future__ import annotations

from omega.historical.contracts import HistoricalEvent
from omega.historical.snapshots import (
    MatchupHistory,
    TeamGameRow,
    build_feature_snapshot,
)

DECISION = "2023-10-01T15:00:00+00:00"
START = "2023-10-01T17:00:00+00:00"


def _event(league="NFL", family="american_football", **kw) -> HistoricalEvent:
    return HistoricalEvent(
        event_id="e1",
        league=league,
        sport_family=family,
        start_time=START,
        home_team="Home",
        away_team="Away",
        source_name="t",
        **kw,
    )


def test_rating_shrinkage_opt_in_only():
    """shrink_ratings defaults off → raw rolling mean; on → EB blend toward baseline."""
    history = MatchupHistory(
        home_rows=[TeamGameRow(date="2023-09-01", points_for=40, points_against=10)],
        away_rows=[TeamGameRow(date="2023-09-10", points_for=24, points_against=21)],
        league_baseline={"off_rating": 20.0, "def_rating": 20.0},
    )
    # Default: raw mean, baseline ignored for a team that has data.
    raw = build_feature_snapshot(_event(), history, DECISION)
    assert raw.home_context["off_rating"] == 40.0
    # Opt-in: n=1 game, n0=5 → (1*40 + 5*20)/6 = 23.333, shrunk toward league mean.
    # (off_rating is rounded to 3 decimals by the snapshot builder.)
    shr = build_feature_snapshot(_event(), history, DECISION, shrink_ratings=True)
    assert shr.home_context["off_rating"] == round((1 * 40 + 5 * 20) / 6, 3)
    # Shrinkage pulls the extreme rating toward the baseline (less over-dispersed).
    assert 20.0 < shr.home_context["off_rating"] < 40.0
    # A team with data is still "provided" — shrinkage does not flip context_source.
    assert shr.context_source == "provided"


def test_post_decision_rows_excluded():
    history = MatchupHistory(
        home_rows=[
            TeamGameRow(date="2023-09-01", points_for=20, points_against=10),
            TeamGameRow(date="2023-09-15", points_for=30, points_against=20),
            # LEAK: this game is after the decision cutoff and must be ignored.
            TeamGameRow(date="2023-10-05", points_for=100, points_against=0),
        ],
        away_rows=[TeamGameRow(date="2023-09-10", points_for=24, points_against=21)],
    )
    snap = build_feature_snapshot(_event(), history, DECISION)
    # off_rating reflects only the two pre-decision games (mean 25), never the 100.
    assert snap.home_context["off_rating"] == 25.0
    assert snap.as_of is not None and snap.as_of < DECISION


def test_game_context_has_required_keys():
    snap = build_feature_snapshot(_event(), MatchupHistory(), DECISION)
    assert "is_playoff" in snap.game_context
    assert "rest_days" in snap.game_context
    assert isinstance(snap.game_context["rest_days"], int)


def test_required_team_keys_per_sport():
    nfl = build_feature_snapshot(
        _event(),
        MatchupHistory(
            home_rows=[TeamGameRow(date="2023-09-15", points_for=21, points_against=17)]
        ),
        DECISION,
    )
    assert {"off_rating", "def_rating"} <= set(nfl.home_context)

    nba = build_feature_snapshot(
        _event(league="NBA", family="basketball"),
        MatchupHistory(
            home_rows=[TeamGameRow(date="2023-09-30", points_for=110, points_against=104)]
        ),
        DECISION,
    )
    assert {"off_rating", "def_rating", "pace"} <= set(nba.home_context)

    tennis = build_feature_snapshot(
        _event(league="ATP", family="tennis"),
        MatchupHistory(
            home_rows=[
                TeamGameRow(
                    date="2023-09-20",
                    serve_points_won=60,
                    serve_points_total=90,
                    return_points_won=30,
                    return_points_total=90,
                )
            ],
            away_rows=[
                TeamGameRow(
                    date="2023-09-21",
                    serve_points_won=55,
                    serve_points_total=90,
                    return_points_won=35,
                    return_points_total=90,
                )
            ],
        ),
        DECISION,
    )
    assert {"serve_win_pct", "return_win_pct"} <= set(tennis.home_context)


def test_default_context_when_no_history():
    snap = build_feature_snapshot(_event(), MatchupHistory(), DECISION)
    assert snap.context_source == "default"
    assert {"off_rating", "def_rating"} <= set(snap.home_context)


def test_stale_flag_when_history_old():
    history = MatchupHistory(
        home_rows=[TeamGameRow(date="2023-01-01", points_for=20, points_against=10)],
        away_rows=[TeamGameRow(date="2023-01-02", points_for=24, points_against=21)],
    )
    snap = build_feature_snapshot(_event(), history, DECISION, staleness_days=60)
    assert snap.is_stale is True


def test_neutral_site_labeled_not_swapped():
    ev = _event(is_neutral_site=True)
    snap = build_feature_snapshot(ev, MatchupHistory(), DECISION)
    assert snap.context_labels["neutral_site"] is True
    assert snap.game_context["neutral_site"] is True


def test_hash_is_deterministic_and_sensitive():
    h1 = build_feature_snapshot(_event(), MatchupHistory(), DECISION)
    h2 = build_feature_snapshot(_event(), MatchupHistory(), DECISION)
    assert h1.feature_snapshot_hash == h2.feature_snapshot_hash != ""

    history = MatchupHistory(
        home_rows=[TeamGameRow(date="2023-09-15", points_for=40, points_against=10)]
    )
    h3 = build_feature_snapshot(_event(), history, DECISION)
    assert h3.feature_snapshot_hash != h1.feature_snapshot_hash


def test_basketball_back_to_back_slice():
    # Home played the day before the event → rest_days == 0 → back_to_back True.
    history = MatchupHistory(
        home_rows=[TeamGameRow(date="2023-09-30", points_for=110, points_against=108)],
    )
    snap = build_feature_snapshot(_event(league="NBA", family="basketball"), history, DECISION)
    assert snap.context_labels.get("back_to_back") is True
