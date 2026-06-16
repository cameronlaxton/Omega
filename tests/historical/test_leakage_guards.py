"""Leakage guard rejects every unsafe pre-game configuration with a reason."""

from __future__ import annotations

from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalFeatureSnapshot,
    HistoricalMarketSnapshot,
    OddsQuote,
)
from omega.historical.leakage import evaluate_leakage

EVENT_START = "2023-10-01T17:00:00+00:00"
DECISION = "2023-10-01T15:00:00+00:00"


def _event() -> HistoricalEvent:
    return HistoricalEvent(
        event_id="e1",
        league="NFL",
        sport_family="american_football",
        start_time=EVENT_START,
        home_team="Kansas City Chiefs",
        away_team="Philadelphia Eagles",
        source_name="t",
    )


def _snapshot(*, decision_time: str = DECISION, as_of: str | None = "2023-09-24T00:00:00+00:00"):
    return HistoricalFeatureSnapshot(
        event_id="e1",
        league="NFL",
        sport_family="american_football",
        decision_time=decision_time,
        game_context={"is_playoff": False, "rest_days": 7},
        as_of=as_of,
    )


def test_clean_passes():
    status = evaluate_leakage(_event(), _snapshot())
    assert status.is_clean
    assert status.reasons == []


def test_post_event_features_rejected():
    snap = _snapshot(as_of="2023-10-02T00:00:00+00:00")  # after the event
    status = evaluate_leakage(_event(), snap)
    assert not status.is_clean
    assert "post_event_features" in status.reasons


def test_unknown_decision_time_rejected():
    snap = _snapshot(decision_time="")
    status = evaluate_leakage(_event(), snap)
    assert "unknown_decision_time" in status.reasons


def test_decision_after_event_rejected():
    snap = _snapshot(decision_time="2023-10-01T18:00:00+00:00")  # after kickoff
    status = evaluate_leakage(_event(), snap)
    assert "decision_after_event" in status.reasons


def test_feature_after_decision_time_rejected():
    snap = _snapshot(as_of="2023-10-01T16:00:00+00:00")  # after decision, before start
    status = evaluate_leakage(_event(), snap)
    assert "feature_after_decision_time" in status.reasons


def test_closing_line_as_decision_rejected():
    odds = HistoricalMarketSnapshot(
        event_id="e1",
        decision_time=DECISION,
        decision=[
            OddsQuote(
                market="moneyline",
                selection_descriptor="home",
                odds=-150,
                timestamp="2023-10-01T17:30:00+00:00",  # after kickoff → closing
            )
        ],
    )
    status = evaluate_leakage(_event(), _snapshot(), odds)
    assert "closing_as_decision" in status.reasons


def test_rolling_window_including_event_rejected():
    status = evaluate_leakage(
        _event(),
        _snapshot(),
        rolling_window_ends=["2023-10-05T00:00:00+00:00"],  # window reaches past the event
    )
    assert "rolling_window_includes_event" in status.reasons


def test_policy_fail_vs_skip():
    snap = _snapshot(as_of="2023-10-02T00:00:00+00:00")
    assert evaluate_leakage(_event(), snap, policy="skip").status == "skipped"
    assert evaluate_leakage(_event(), snap, policy="fail").status == "failed"
