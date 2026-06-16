"""Replay exercises the 3-way game + draw plane and closing-line attach.

Stands in for the EPL worked example using NHL/hockey (regulation draws via the
fast_score Poisson backend, no Dixon-Coles rho prior required): proves draw_prob
is produced, draws grade, the draw calibration plane yields fittable pairs, and
closing lines attach for CLV.
"""

from __future__ import annotations

from omega.core.calibration.fitter import CalibrationFitter
from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    OddsObservation,
    ReplayConfig,
)
from omega.historical.identity import event_key
from omega.historical.normalize import parse_datetime_utc
from omega.historical.replay import ReplayDataset, ReplayEngine

LEAGUE = "NHL"
FAMILY = "hockey"


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
        event_id=ev.event_id, home_score=hs, away_score=as_,
        result=HistoricalOutcome.derive_result(hs, as_),
    )


def _dataset():
    e1 = _event("2024-01-01", "Bruins", "Rangers")
    e2 = _event("2024-01-08", "Oilers", "Bruins")
    e3 = _event("2024-01-15", "Rangers", "Oilers")
    target = _event("2024-01-22", "Bruins", "Oilers")
    outcomes = {
        e1.event_id: _oc(e1, 3, 2),
        e2.event_id: _oc(e2, 2, 2),  # draw (regulation tie)
        e3.event_id: _oc(e3, 1, 1),  # draw
        target.event_id: _oc(target, 2, 2),  # draw
    }
    obs: list[OddsObservation] = []
    for ev in (e1, e2, e3, target):
        obs.append(OddsObservation(event_key=ev.event_id, market="moneyline", selection_descriptor="home", odds=-120))
        obs.append(OddsObservation(event_key=ev.event_id, market="moneyline", selection_descriptor="away", odds=110))
    obs.append(
        OddsObservation(
            event_key=target.event_id, market="moneyline",
            selection_descriptor="home", odds=-135, tier_hint="closing",
        )
    )
    return ReplayDataset(events=[e1, e2, e3, target], outcomes=outcomes, odds=ReplayDataset.group_odds(obs)), target


def test_replay_draw_plane(backtest_store, tmp_path):
    dataset, target = _dataset()
    cfg = ReplayConfig(
        dataset_manifest_id="m", backtest_db_path=str(tmp_path / "bt.db"), n_iterations=400
    )
    result = ReplayEngine(backtest_store, cfg).run(dataset, replay_id="r", league=LEAGUE)
    assert result.n_persisted == 4

    rec = next(r for r in result.manifest.records if r.event_id == target.event_id)
    trace = backtest_store.get_trace(rec.trace_id)
    assert trace["predictions"].get("draw_prob") is not None

    closing = backtest_store.get_closing_lines(rec.trace_id)
    assert any(c["market"] == "moneyline" for c in closing)

    graded = backtest_store.query_traces(
        execution_mode="historical_replay", has_outcome=True,
        calibration_eligible_only=True, limit=100,
    )
    preds, outs = CalibrationFitter().extract_draw_pairs(graded)
    assert len(preds) == len(outs) >= 1
    assert set(outs) <= {0, 1}
