"""Basketball / baseball / hockey lanes: adapters, slices, and end-to-end replay."""

from __future__ import annotations

from omega.historical.adapters.csv_games import CsvGamesAdapter
from omega.historical.adapters.nba_csv import NbaCsvAdapter
from omega.historical.contracts import ReplayConfig
from omega.historical.replay import ReplayDataset, ReplayEngine, build_team_histories
from omega.historical.snapshots import MatchupHistory, build_feature_snapshot
from omega.historical.walk_forward import _slice_of


def _replay(events, outcomes, league, backtest_store, tmp_path, **extra):
    dataset = ReplayDataset(
        events=events, outcomes={o.event_id: o for o in outcomes}, **extra
    )
    config = ReplayConfig(
        dataset_manifest_id=f"{league}-fixture",
        backtest_db_path=str(tmp_path / "bt.db"),
        n_iterations=200,
    )
    return ReplayEngine(backtest_store, config).run(dataset, replay_id=f"{league}-1", league=league)


def test_nba_csv_adapter_parses(fixtures_dir):
    a = NbaCsvAdapter("NBA")
    path = fixtures_dir / "nba_kaggle_sample.csv"
    events = a.read_events(path)
    assert len(events) == 8
    assert all(e.sport_family == "basketball" for e in events)
    # the PLAYOFF=Y row is flagged
    assert sum(e.is_playoff for e in events) == 1


def test_basketball_back_to_back_from_fixture(fixtures_dir):
    a = NbaCsvAdapter("NBA")
    path = fixtures_dir / "nba_kaggle_sample.csv"
    events = a.read_events(path)
    outcomes = {o.event_id: o for o in a.read_outcomes(path)}
    hist = build_team_histories(events, outcomes)
    target = next(
        e for e in events if e.home_team == "Celtics" and e.start_time.startswith("2023-10-27")
    )
    snap = build_feature_snapshot(
        target,
        MatchupHistory(
            home_rows=hist.get(target.home_team, []), away_rows=hist.get(target.away_team, [])
        ),
        target.start_time,
    )
    assert snap.context_labels.get("back_to_back") is True


def test_hockey_three_in_four_from_fixture(fixtures_dir):
    a = CsvGamesAdapter("NHL")
    path = fixtures_dir / "nhl_games_sample.csv"
    events = a.read_events(path)
    outcomes = {o.event_id: o for o in a.read_outcomes(path)}
    hist = build_team_histories(events, outcomes)
    target = next(
        e for e in events if e.home_team == "Bruins" and e.start_time.startswith("2023-10-14")
    )
    snap = build_feature_snapshot(
        target,
        MatchupHistory(
            home_rows=hist.get(target.home_team, []), away_rows=hist.get(target.away_team, [])
        ),
        target.start_time,
    )
    assert snap.context_labels.get("three_in_four") is True


def test_baseball_park_factor_slice_from_extra(fixtures_dir):
    a = CsvGamesAdapter("MLB")
    path = fixtures_dir / "mlb_games_sample.csv"
    event = a.read_events(path)[0]
    snap = build_feature_snapshot(
        event, MatchupHistory(), event.start_time, extra_game_context={"park_factor": 1.2}
    )
    assert snap.context_labels.get("park_factor_extreme") is True


def test_extra_passthrough_slice_reachable_in_walk_forward():
    # Slices needing richer inputs (goalie status, injuries, weather, division) are
    # supplied via game_context and remain selectable for calibration slicing.
    trace = {"context_labels": {}, "input_snapshot": {"game_context": {"goalie_confirmed": True}}}
    assert _slice_of(trace, ["goalie_confirmed"]) == "goalie_confirmed"


def test_nba_end_to_end_replay(fixtures_dir, backtest_store, tmp_path):
    a = NbaCsvAdapter("NBA")
    path = fixtures_dir / "nba_kaggle_sample.csv"
    result = _replay(a.read_events(path), a.read_outcomes(path), "NBA", backtest_store, tmp_path)
    assert result.n_persisted == 8
    assert result.n_skipped == 0


def test_mlb_end_to_end_replay(fixtures_dir, backtest_store, tmp_path):
    a = CsvGamesAdapter("MLB")
    path = fixtures_dir / "mlb_games_sample.csv"
    result = _replay(a.read_events(path), a.read_outcomes(path), "MLB", backtest_store, tmp_path)
    assert result.n_persisted == 8


def test_nhl_end_to_end_replay(fixtures_dir, backtest_store, tmp_path):
    a = CsvGamesAdapter("NHL")
    path = fixtures_dir / "nhl_games_sample.csv"
    result = _replay(a.read_events(path), a.read_outcomes(path), "NHL", backtest_store, tmp_path)
    assert result.n_persisted == 8
