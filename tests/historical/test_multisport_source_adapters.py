from __future__ import annotations

import zipfile

import pandas as pd

from omega.historical.adapters.retrosheet import RetrosheetGameLogAdapter
from omega.historical.adapters.soccer_football_data import SoccerFootballDataAdapter
from omega.historical.adapters.sportsdataverse import SportsDataverseScheduleAdapter
from omega.historical.odds_snapshots import build_odds_snapshot


def test_sportsdataverse_basketball_filters_incomplete_and_maps_playoffs(tmp_path):
    path = tmp_path / "schedule.parquet"
    pd.DataFrame([
        {
            "game_id": 1, "start_date": "2025-06-01T00:00Z",
            "home_display_name": "New York Knicks", "away_display_name": "Indiana Pacers",
            "home_score": 111, "away_score": 109, "status_type_completed": True,
            "neutral_site": False, "season": 2025, "season_type": 3,
        },
        {
            "game_id": 2, "start_date": "2026-06-01T00:00Z",
            "home_display_name": "New York Knicks", "away_display_name": "Indiana Pacers",
            "home_score": 0, "away_score": 0, "status_type_completed": False,
            "neutral_site": False, "season": 2026, "season_type": 2,
        },
    ]).to_parquet(path, index=False)

    adapter = SportsDataverseScheduleAdapter("NBA")
    events = adapter.read_events(path)
    outcomes = adapter.read_outcomes(path)
    assert len(events) == len(outcomes) == 1
    assert events[0].is_playoff is True
    assert events[0].identity_status == "complete"
    assert outcomes[0].result == "home_win"


def test_sportsdataverse_nhl_expands_abbreviations(tmp_path):
    path = tmp_path / "schedule.parquet"
    pd.DataFrame([{
        "game_id": 2024020001, "game_time": "2024-10-04T17:00:00Z",
        "home_team_abbr": "BUF", "away_team_abbr": "NJD",
        "home_score": 1, "away_score": 4, "game_state": "OFF",
        "season_full": "20242025", "game_type": "R",
    }]).to_parquet(path, index=False)

    event = SportsDataverseScheduleAdapter("NHL").read_events(path)[0]
    assert event.home_team == "Buffalo Sabres"
    assert event.away_team == "New Jersey Devils"
    assert event.identity_status == "complete"


def test_retrosheet_zip_maps_identity_and_score(tmp_path):
    archive = tmp_path / "gl2025.zip"
    row = [""] * 161
    row[0], row[1] = "20250401", "0"
    row[3], row[6] = "NYA", "LAN"
    row[9], row[10] = "3", "5"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("GL2025.TXT", ",".join(row) + "\n")

    adapter = RetrosheetGameLogAdapter()
    event = adapter.read_events(archive)[0]
    outcome = adapter.read_outcomes(archive)[0]
    assert event.away_team == "New York Yankees"
    assert event.home_team == "Los Angeles Dodgers"
    assert event.identity_status == "complete"
    assert outcome.result == "home_win"


def test_retrosheet_doubleheader_ids_are_distinct(tmp_path):
    archive = tmp_path / "gl2025.zip"
    rows = []
    for game_number in ("1", "2"):
        row = [""] * 161
        row[0], row[1] = "20250401", game_number
        row[3], row[6], row[9], row[10] = "NYA", "LAN", "3", "5"
        rows.append(",".join(row))
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("GL2025.TXT", "\n".join(rows) + "\n")

    events = RetrosheetGameLogAdapter().read_events(archive)
    assert len({event.event_id for event in events}) == 2


def test_football_data_keeps_opening_and_closing_separate(tmp_path):
    path = tmp_path / "epl.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,B365CH,B365CD,B365CA,"
        "B365>2.5,B365<2.5,B365C>2.5,B365C<2.5\n"
        "01/08/2025,Arsenal,Chelsea,2,1,2.00,3.50,4.00,1.90,3.60,4.20,1.80,2.00,1.75,2.10\n",
        encoding="utf-8",
    )
    adapter = SoccerFootballDataAdapter("EPL")
    odds = adapter.read_odds(path, alias_table={
        "canonical": ["Arsenal", "Chelsea"], "aliases": {}
    })
    assert {o.tier_hint for o in odds} == {"opening", "closing"}
    snapshot = build_odds_snapshot(
        odds[0].event_key, odds, "2025-08-01T00:00:00+00:00",
        event_start="2025-08-01T00:00:00+00:00",
    )
    assert snapshot.decision
    assert snapshot.closing
    assert all(q.book == "bet365" for q in snapshot.opening + snapshot.closing)
