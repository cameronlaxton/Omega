"""Replay produces normal traces, attaches outcomes/closing lines, logs replay bets."""

from __future__ import annotations

from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    OddsObservation,
    ReplayConfig,
)
from omega.historical.identity import event_key
from omega.historical.normalize import parse_datetime_utc
from omega.historical.replay import ReplayDataset, ReplayEngine

LEAGUE = "NFL"
FAMILY = "american_football"


def _event(date: str, home: str, away: str, **kw) -> HistoricalEvent:
    start = parse_datetime_utc(date)
    return HistoricalEvent(
        event_id=event_key(LEAGUE, start, home, away),
        league=LEAGUE,
        sport_family=FAMILY,
        start_time=start,
        home_team=home,
        away_team=away,
        source_name="test",
        **kw,
    )


def _outcome(ev: HistoricalEvent, hs: int, as_: int) -> HistoricalOutcome:
    return HistoricalOutcome(
        event_id=ev.event_id,
        home_score=hs,
        away_score=as_,
        result=HistoricalOutcome.derive_result(hs, as_),
    )


def _dataset() -> tuple[ReplayDataset, HistoricalEvent]:
    # A small slate giving the final game prior-game history for both teams.
    e1 = _event("2023-09-10", "Team A", "Team B")
    e2 = _event("2023-09-17", "Team C", "Team A")
    e3 = _event("2023-09-24", "Team B", "Team C")
    target = _event("2023-10-01", "Team A", "Team C")
    events = [e1, e2, e3, target]
    outcomes = {
        e1.event_id: _outcome(e1, 24, 17),
        e2.event_id: _outcome(e2, 20, 27),
        e3.event_id: _outcome(e3, 30, 21),
        target.event_id: _outcome(target, 28, 24),
    }

    # Untimestamped pre-match moneyline prices (trusted as decision odds) plus a
    # labelled closing price on the target game for CLV.
    obs: list[OddsObservation] = []

    def add_ml(ev: HistoricalEvent, home_price: float, away_price: float) -> None:
        obs.append(
            OddsObservation(
                event_key=ev.event_id, market="moneyline",
                selection_descriptor="home", odds=home_price,
            )
        )
        obs.append(
            OddsObservation(
                event_key=ev.event_id, market="moneyline",
                selection_descriptor="away", odds=away_price,
            )
        )

    for ev in (e1, e2, e3):
        add_ml(ev, -110, -110)
    # Both sides +200 → whichever side the model favors carries clear value.
    add_ml(target, 200, 200)
    obs.append(
        OddsObservation(
            event_key=target.event_id, market="moneyline",
            selection_descriptor="home", odds=-180, tier_hint="closing",
        )
    )

    dataset = ReplayDataset(
        events=events, outcomes=outcomes, odds=ReplayDataset.group_odds(obs)
    )
    return dataset, target


def _config(tmp_path) -> ReplayConfig:
    return ReplayConfig(
        dataset_manifest_id="m-test",
        backtest_db_path=str(tmp_path / "backtest.db"),
        session_id="hist-test",
        enable_staking=True,
        odds_timing_class="decision_time_safe",
        n_iterations=200,
    )


def test_replay_persists_normal_traces_with_metadata(backtest_store, tmp_path):
    dataset, target = _dataset()
    engine = ReplayEngine(backtest_store, _config(tmp_path))
    result = engine.run(dataset, replay_id="r-1", league=LEAGUE)

    assert result.n_persisted == 4
    assert result.n_skipped == 0

    rec = next(r for r in result.manifest.records if r.event_id == target.event_id)
    trace = backtest_store.get_trace(rec.trace_id)
    assert trace is not None
    assert trace["historical_replay"] is True
    assert trace["replay_id"] == "r-1"
    assert trace["dataset_manifest_id"] == "m-test"
    assert trace["event_id"] == target.event_id
    assert trace["feature_snapshot_hash"]
    assert trace["odds_snapshot_hash"]
    assert trace["leakage_status"] == "clean"

    # Calibration-backfill tagging: replayed traces carry the historical_replay
    # selection tag + provenance, while predictions stay RAW (uncalibrated).
    assert trace["execution_mode"] == "historical_replay"
    assert trace["artifact_schema_version"] == 1
    assert trace["source_provenance"]["source_name"] == "test"
    assert trace["source_provenance"]["dataset_manifest_id"] == "m-test"
    preds = trace["predictions"]
    assert isinstance(preds, dict) and preds.get("home_win_prob") is not None


def test_replay_attaches_outcome_and_closing_line(backtest_store, tmp_path):
    dataset, target = _dataset()
    engine = ReplayEngine(backtest_store, _config(tmp_path))
    result = engine.run(dataset, replay_id="r-2", league=LEAGUE)
    rec = next(r for r in result.manifest.records if r.event_id == target.event_id)

    outcome = backtest_store.get_outcome(rec.trace_id)
    assert outcome is not None
    assert outcome["home_score"] == 28 and outcome["away_score"] == 24

    closing = backtest_store.get_closing_lines(rec.trace_id)
    assert any(c["market"] == "moneyline" for c in closing)


def test_replay_logs_historical_replay_ledger_and_suppresses_autolog(backtest_store, tmp_path):
    dataset, _target = _dataset()
    engine = ReplayEngine(backtest_store, _config(tmp_path))
    engine.run(dataset, replay_id="r-3", league=LEAGUE)

    replay_bets = backtest_store.query_ledger(provenance="historical_replay")
    engine_auto = backtest_store.query_ledger(provenance="engine_auto")
    assert replay_bets, "expected at least one historical_replay ledger bet"
    assert engine_auto == [], "autolog must be suppressed during replay"


def test_replay_emits_candidate_selection_binding(backtest_store, tmp_path):
    dataset, _target = _dataset()
    engine = ReplayEngine(backtest_store, _config(tmp_path))
    result = engine.run(dataset, replay_id="r-4", league=LEAGUE)

    assert result.selections, "expected at least one candidate selection"
    sel = result.selections[0]
    assert sel.trace_id
    assert sel.ledger_id
    assert sel.decision_odds is not None
    assert sel.replay_id == "r-4"


def test_leakage_skip_does_not_persist(backtest_store, tmp_path, monkeypatch):
    # The engine is leak-safe by construction, so the guard is a defensive net.
    # Force it to flag to verify the engine's skip branch: no trace persisted,
    # event recorded as skipped.
    from omega.historical import replay as replay_mod
    from omega.historical.leakage import LeakageStatus

    monkeypatch.setattr(
        replay_mod,
        "evaluate_leakage",
        lambda *a, **k: LeakageStatus(status="skipped", reasons=["post_event_features"]),
    )
    dataset, _target = _dataset()
    engine = ReplayEngine(backtest_store, _config(tmp_path))
    result = engine.run(dataset, replay_id="r-5", league=LEAGUE)

    assert result.n_persisted == 0
    assert result.n_skipped == len(dataset.events)
    assert all(r.trace_id is None for r in result.manifest.records)
    assert all(r.leakage_status == "skipped" for r in result.manifest.records)
    assert backtest_store.query_ledger(provenance="historical_replay") == []


def test_leakage_fail_policy_raises(backtest_store, tmp_path, monkeypatch):
    import pytest

    from omega.historical import replay as replay_mod
    from omega.historical.leakage import LeakageStatus

    monkeypatch.setattr(
        replay_mod,
        "evaluate_leakage",
        lambda *a, **k: LeakageStatus(status="failed", reasons=["post_event_features"]),
    )
    dataset, _target = _dataset()
    cfg = _config(tmp_path).model_copy(update={"leakage_policy": "fail"})
    engine = ReplayEngine(backtest_store, cfg)
    with pytest.raises(replay_mod.LeakageError):
        engine.run(dataset, replay_id="r-6", league=LEAGUE)
