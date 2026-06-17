"""First-class lane adapters (NFL/soccer/tennis) parse fixtures and replay end-to-end."""

from __future__ import annotations

from omega.historical.adapters.nflfast_csv import NflfastCsvAdapter
from omega.historical.adapters.soccer_football_data import SoccerFootballDataAdapter
from omega.historical.adapters.tennis_atp_csv import TennisAtpCsvAdapter
from omega.historical.contracts import ReplayConfig
from omega.historical.replay import ReplayDataset, ReplayEngine


def test_nflfast_adapter_parses(fixtures_dir):
    a = NflfastCsvAdapter("NFL")
    path = fixtures_dir / "nfl_nflverse_sample.csv"
    events = a.read_events(path)
    outcomes = a.read_outcomes(path)
    assert len(events) == 8
    assert len(outcomes) == 8
    assert all(e.sport_family == "american_football" for e in events)
    assert all(not e.is_playoff and not e.is_neutral_site for e in events)


def test_football_data_adapter_parses_events_outcomes_odds(fixtures_dir):
    a = SoccerFootballDataAdapter("EPL")
    path = fixtures_dir / "soccer_football_data_sample.csv"
    events = a.read_events(path)
    outcomes = a.read_outcomes(path)
    odds = a.read_odds(path)
    assert len(events) == 8
    assert len(outcomes) == 8
    markets = {o.market for o in odds}
    assert "home_draw_away" in markets
    assert "total" in markets
    # decimal odds were converted to valid American odds
    assert all(o.odds >= 100 or o.odds <= -100 for o in odds)


def test_tennis_adapter_parses_with_serve_history(fixtures_dir):
    a = TennisAtpCsvAdapter("ATP")
    path = fixtures_dir / "tennis_atp_sample.csv"
    events = a.read_events(path)
    outcomes = a.read_outcomes(path)
    history = a.read_serve_history(path)
    extra = a.read_extra_context(path)
    assert len(events) == 8
    # outcomes are 1/0 by match winner, never by seating
    assert all({o.home_score, o.away_score} == {0, 1} for o in outcomes)
    assert history, "expected per-player serve/return history"
    assert any("surface" in ctx for ctx in extra.values())
    assert any(ctx.get("best_of") == 5 for ctx in extra.values())


def test_nfl_end_to_end_replay(fixtures_dir, backtest_store, tmp_path):
    a = NflfastCsvAdapter("NFL")
    path = fixtures_dir / "nfl_nflverse_sample.csv"
    events = a.read_events(path)
    outcomes = {o.event_id: o for o in a.read_outcomes(path)}
    dataset = ReplayDataset(events=events, outcomes=outcomes)

    config = ReplayConfig(
        dataset_manifest_id="nfl-fixture",
        backtest_db_path=str(tmp_path / "bt.db"),
        n_iterations=200,
    )
    result = ReplayEngine(backtest_store, config).run(dataset, replay_id="nfl-1", league="NFL")
    assert result.n_persisted == 8
    assert result.n_skipped == 0
    # A week-4 event should have computed (non-default) context from prior weeks.
    sources = {
        backtest_store.get_trace(r.trace_id)["trace_quality"]["context_source"]
        for r in result.manifest.records
    }
    assert "provided" in sources


def test_soccer_end_to_end_replay_with_odds(fixtures_dir, backtest_store, tmp_path):
    a = SoccerFootballDataAdapter("EPL")
    path = fixtures_dir / "soccer_football_data_sample.csv"
    events = a.read_events(path)
    outcomes = {o.event_id: o for o in a.read_outcomes(path)}
    odds = ReplayDataset.group_odds(a.read_odds(path))
    dataset = ReplayDataset(events=events, outcomes=outcomes, odds=odds)

    config = ReplayConfig(
        dataset_manifest_id="epl-fixture",
        backtest_db_path=str(tmp_path / "bt.db"),
        n_iterations=200,
        enable_staking=True,
    )
    result = ReplayEngine(backtest_store, config).run(dataset, replay_id="epl-1", league="EPL")
    assert result.n_persisted == 8
    # outcomes attach for every replayed game
    for rec in result.manifest.records:
        assert backtest_store.get_outcome(rec.trace_id) is not None


def test_tennis_end_to_end_probability_only(fixtures_dir, backtest_store, tmp_path):
    a = TennisAtpCsvAdapter("ATP")
    path = fixtures_dir / "tennis_atp_sample.csv"
    events = a.read_events(path)
    outcomes = {o.event_id: o for o in a.read_outcomes(path)}
    dataset = ReplayDataset(
        events=events,
        outcomes=outcomes,
        extra_context=a.read_extra_context(path),
        history_override=a.read_serve_history(path),
    )
    config = ReplayConfig(
        dataset_manifest_id="atp-fixture",
        backtest_db_path=str(tmp_path / "bt.db"),
        n_iterations=200,
    )
    result = ReplayEngine(backtest_store, config).run(dataset, replay_id="atp-1", league="ATP")
    # probability-only (no odds) but every match still produces a normal trace
    assert result.n_persisted == 8
