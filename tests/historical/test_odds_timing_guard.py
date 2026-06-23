"""Odds-timing safety: gates betting only, never probability calibration."""

from __future__ import annotations

from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    OddsObservation,
    ReplayConfig,
)
from omega.historical.identity import event_key
from omega.historical.normalize import parse_datetime_utc
from omega.historical.odds_timing import (
    OddsTimingClass,
    allows_clv,
    allows_roi,
    is_selection_safe,
    timing_class_for_source,
)
from omega.historical.replay import ReplayDataset, ReplayEngine

LEAGUE = "NFL"
FAMILY = "american_football"


def _event(date: str, home: str, away: str) -> HistoricalEvent:
    start = parse_datetime_utc(date)
    return HistoricalEvent(
        event_id=event_key(LEAGUE, start, home, away),
        league=LEAGUE,
        sport_family=FAMILY,
        start_time=start,
        home_team=home,
        away_team=away,
        source_name="test",
    )


def _oc(ev, hs, as_):
    return HistoricalOutcome(
        event_id=ev.event_id,
        home_score=hs,
        away_score=as_,
        result=HistoricalOutcome.derive_result(hs, as_),
    )


def _dataset() -> ReplayDataset:
    e1 = _event("2023-09-10", "Team A", "Team B")
    e2 = _event("2023-09-17", "Team C", "Team A")
    e3 = _event("2023-09-24", "Team B", "Team C")
    target = _event("2023-10-01", "Team A", "Team C")
    outcomes = {
        e1.event_id: _oc(e1, 24, 17),
        e2.event_id: _oc(e2, 20, 27),
        e3.event_id: _oc(e3, 30, 21),
        target.event_id: _oc(target, 28, 24),
    }
    obs: list[OddsObservation] = []
    for ev, hp, ap in [(e1, -110, -110), (e2, -110, -110), (e3, -110, -110), (target, 200, 200)]:
        obs.append(
            OddsObservation(
                event_key=ev.event_id, market="moneyline", selection_descriptor="home", odds=hp
            )
        )
        obs.append(
            OddsObservation(
                event_key=ev.event_id, market="moneyline", selection_descriptor="away", odds=ap
            )
        )
    return ReplayDataset(
        events=[e1, e2, e3, target], outcomes=outcomes, odds=ReplayDataset.group_odds(obs)
    )


def _cfg(tmp_path, timing: str) -> ReplayConfig:
    return ReplayConfig(
        dataset_manifest_id="m",
        backtest_db_path=str(tmp_path / "bt.db"),
        enable_staking=True,
        n_iterations=200,
        odds_timing_class=timing,
    )


def test_timing_class_registry_and_guards():
    assert timing_class_for_source("football_data") is OddsTimingClass.DECISION_TIME_SAFE
    assert timing_class_for_source("mystery_dump") is OddsTimingClass.TIMING_UNKNOWN
    assert is_selection_safe("decision_time_safe") is True
    assert is_selection_safe("closing_only") is False
    assert is_selection_safe("timing_unknown") is False
    assert allows_roi("timing_unknown") is False
    assert allows_clv("closing_only") is True
    assert allows_clv("timing_unknown") is False


def test_decision_time_safe_allows_staking(backtest_store, tmp_path):
    ReplayEngine(backtest_store, _cfg(tmp_path, "decision_time_safe")).run(
        _dataset(), replay_id="r-safe", league=LEAGUE
    )
    assert backtest_store.query_ledger(provenance="historical_replay")


def test_timing_unknown_blocks_staking_but_not_calibration(backtest_store, tmp_path):
    ReplayEngine(backtest_store, _cfg(tmp_path, "timing_unknown")).run(
        _dataset(), replay_id="r-unk", league=LEAGUE
    )
    # No staking on unknown-timing odds...
    assert backtest_store.query_ledger(provenance="historical_replay") == []
    # ...but probability calibration is unaffected: traces still eligible.
    eligible = backtest_store.query_traces(
        execution_mode="historical_replay",
        has_outcome=True,
        calibration_eligible_only=True,
        limit=100,
    )
    assert len(eligible) == 4
