"""Phase 4 invariants: replay stores RAW probabilities, grades eligibly, binds outcomes.

These lock the contract the calibration backfill depends on:
* persisted ``predictions`` are the raw simulation output, never calibrated probs;
* a graded historical_replay trace is calibration-eligible by construction;
* the attached outcome is bound to the correct trace by trace_id.
"""

from __future__ import annotations

from omega.historical.contracts import HistoricalEvent, HistoricalOutcome, ReplayConfig
from omega.historical.identity import event_key
from omega.historical.normalize import parse_datetime_utc
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


def _outcome(ev: HistoricalEvent, hs: int, as_: int) -> HistoricalOutcome:
    return HistoricalOutcome(
        event_id=ev.event_id,
        home_score=hs,
        away_score=as_,
        result=HistoricalOutcome.derive_result(hs, as_),
    )


def _run(store, tmp_path):
    e1 = _event("2023-09-10", "Team A", "Team B")
    e2 = _event("2023-09-17", "Team C", "Team A")
    e3 = _event("2023-09-24", "Team B", "Team C")
    target = _event("2023-10-01", "Team A", "Team C")
    outcomes = {
        e1.event_id: _outcome(e1, 24, 17),
        e2.event_id: _outcome(e2, 20, 27),
        e3.event_id: _outcome(e3, 30, 21),
        target.event_id: _outcome(target, 28, 24),
    }
    ds = ReplayDataset(events=[e1, e2, e3, target], outcomes=outcomes, odds={})
    cfg = ReplayConfig(
        dataset_manifest_id="m", backtest_db_path=str(tmp_path / "backtest.db"), n_iterations=200
    )
    result = ReplayEngine(store, cfg).run(ds, replay_id="r", league=LEAGUE)
    return result, target


def test_predictions_are_raw_simulation(backtest_store, tmp_path):
    result, target = _run(backtest_store, tmp_path)
    rec = next(r for r in result.manifest.records if r.event_id == target.event_id)
    trace = backtest_store.get_trace(rec.trace_id)

    preds = trace["predictions"]
    sim = trace["result"]["simulation"]
    # predictions mirror the RAW simulation block (no post-calibration values).
    assert preds["home_win_prob"] == sim["home_win_prob"]
    assert preds["away_win_prob"] == sim["away_win_prob"]
    # The raw sim block carries no calibrated probability.
    assert "calibrated_prob" not in preds


def test_replayed_trace_is_calibration_eligible(backtest_store, tmp_path):
    _run(backtest_store, tmp_path)
    eligible = backtest_store.query_traces(
        execution_mode="historical_replay",
        has_outcome=True,
        calibration_eligible_only=True,
        limit=100,
    )
    assert len(eligible) == 4
    assert all(t["trace_quality"]["calibration_eligible"] is True for t in eligible)


def test_outcome_bound_to_correct_trace(backtest_store, tmp_path):
    result, target = _run(backtest_store, tmp_path)
    rec = next(r for r in result.manifest.records if r.event_id == target.event_id)
    outcome = backtest_store.get_outcome(rec.trace_id)
    assert outcome["home_score"] == 28
    assert outcome["away_score"] == 24
    assert outcome["result"] == "home_win"
