"""Player-prop replay: eligible prop traces, decision-time lines, void exclusion.

Props stay league-scoped player-stat markets. The decision-time line/prices drive
the prediction; the realized stat_value is attached only as the outcome; void/DNP
props are excluded from prop calibration.
"""

from __future__ import annotations

from omega.core.calibration.fitter import CalibrationFitter
from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    HistoricalPropMarket,
    HistoricalPropOutcome,
    ReplayConfig,
)
from omega.historical.identity import event_key
from omega.historical.normalize import parse_datetime_utc
from omega.historical.replay import ReplayDataset, ReplayEngine, _recommended_prop_side

LG, FAM = "NBA", "basketball"


def _event(d, h, a):
    s = parse_datetime_utc(d)
    return HistoricalEvent(
        event_id=event_key(LG, s, h, a), league=LG, sport_family=FAM,
        start_time=s, home_team=h, away_team=a, source_name="test",
    )


def _oc(e, hs, as_, props=None):
    return HistoricalOutcome(
        event_id=e.event_id, home_score=hs, away_score=as_,
        result=HistoricalOutcome.derive_result(hs, as_), prop_outcomes=props or [],
    )


def _dataset():
    e1 = _event("2024-01-01", "Lakers", "Celtics")
    e2 = _event("2024-01-08", "Heat", "Lakers")
    e3 = _event("2024-01-15", "Celtics", "Heat")
    tg = _event("2024-01-22", "Lakers", "Heat")
    props = [
        HistoricalPropOutcome(player_name="LeBron James", stat_type="pts", stat_value=29.0),
        HistoricalPropOutcome(player_name="Bench Guy", stat_type="pts", stat_value=None, void=True),
    ]
    outcomes = {
        e1.event_id: _oc(e1, 110, 100),
        e2.event_id: _oc(e2, 105, 108),
        e3.event_id: _oc(e3, 99, 101),
        tg.event_id: _oc(tg, 112, 104, props),
    }
    markets = {
        tg.event_id: [
            HistoricalPropMarket(event_key=tg.event_id, player_name="LeBron James",
                                 stat_type="pts", line=24.5, over_price=-110, under_price=-110),
            HistoricalPropMarket(event_key=tg.event_id, player_name="Bench Guy",
                                 stat_type="pts", line=8.5, over_price=-110, under_price=-110),
        ]
    }
    prop_context = {
        f"{tg.event_id}|LeBron James|pts": {"pts_mean": 27.0, "pts_std": 6.0},
        f"{tg.event_id}|Bench Guy|pts": {"pts_mean": 9.0, "pts_std": 4.0},
    }
    ds = ReplayDataset(
        events=[e1, e2, e3, tg], outcomes=outcomes, odds={},
        prop_markets=markets, prop_context=prop_context,
    )
    return ds, tg


def _run(store, tmp_path):
    ds, tg = _dataset()
    cfg = ReplayConfig(
        dataset_manifest_id="m", backtest_db_path=str(tmp_path / "bt.db"), n_iterations=300
    )
    ReplayEngine(store, cfg).run(ds, replay_id="r", league=LG)
    return tg


def test_prop_replay_eligible_and_decision_time_line(backtest_store, tmp_path):
    _run(backtest_store, tmp_path)
    props = backtest_store.query_traces(
        execution_mode="historical_replay", has_outcome=True,
        calibration_eligible_only=True, limit=100,
    )
    prop_traces = [t for t in props if t.get("kind") == "prop"]
    assert len(prop_traces) == 1  # pass recommendations are not graded as phantom sides

    lebron = next(
        t for t in prop_traces
        if t["_prop_outcomes"][0]["player_name"] == "LeBron James"
    )
    # Predictions are RAW over/under probabilities.
    assert lebron["predictions"].get("over_prob") is not None
    po = lebron["_prop_outcomes"][0]
    # Line is the decision-time market line (24.5), independent of the realized value (29).
    assert po["line"] == 24.5
    assert po["stat_value"] == 29.0
    assert po["result"] == "win"


def test_void_prop_excluded_from_calibration(backtest_store, tmp_path):
    _run(backtest_store, tmp_path)
    props = backtest_store.query_traces(
        execution_mode="historical_replay", has_outcome=True,
        calibration_eligible_only=True, limit=100,
    )
    preds, outs = CalibrationFitter().extract_prop_pairs(props)
    # Only the non-void prop contributes a calibration pair.
    assert len(preds) == 1
    assert outs == [1]


def test_missing_prop_context_replay_trace_is_ineligible(backtest_store, tmp_path):
    ds, tg = _dataset()
    ds.prop_markets = {
        tg.event_id: [
            HistoricalPropMarket(
                event_key=tg.event_id,
                player_name="No Context",
                stat_type="pts",
                line=7.5,
                over_price=-110,
                under_price=-110,
            )
        ]
    }
    ds.prop_context = {}
    ds.outcomes[tg.event_id] = ds.outcomes[tg.event_id].model_copy(
        update={
            "prop_outcomes": [
                HistoricalPropOutcome(
                    player_name="No Context",
                    stat_type="pts",
                    stat_value=9.0,
                )
            ]
        }
    )
    cfg = ReplayConfig(
        dataset_manifest_id="m", backtest_db_path=str(tmp_path / "bt.db"), n_iterations=300
    )
    ReplayEngine(backtest_store, cfg).run(ds, replay_id="r", league=LG)

    traces = backtest_store.query_traces(
        execution_mode="historical_replay", has_outcome=True, limit=100
    )
    prop_trace = next(t for t in traces if t.get("kind") == "prop")
    assert prop_trace["input_snapshot"]["player_name"] == "No Context"
    assert prop_trace["result"]["status"] == "skipped"
    assert prop_trace["trace_quality"]["calibration_eligible"] is False
    assert "engine_skipped" in prop_trace["trace_quality"]["calibration_exclusion_reasons"]

    eligible = backtest_store.query_traces(
        execution_mode="historical_replay",
        has_outcome=True,
        calibration_eligible_only=True,
        limit=100,
    )
    assert all(
        t.get("kind") != "prop" or t["input_snapshot"]["player_name"] != "No Context"
        for t in eligible
    )


def test_recommended_prop_side_only_allows_over_under():
    assert _recommended_prop_side({"result": {"recommendation": "under"}}) == "under"
    assert _recommended_prop_side({"result": {"recommendation": "OVER"}}) == "over"
    assert _recommended_prop_side({"result": {"recommendation": "pass"}}) is None
