"""Contract construction, serialization round-trips, and stable hashing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omega.historical.contracts import (
    BacktestReport,
    FoldResult,
    FrozenProfileRef,
    HealthBlock,
    HistoricalEvent,
    HistoricalFeatureSnapshot,
    HistoricalMarketSnapshot,
    HistoricalOutcome,
    HistoricalPropOutcome,
    MetricBlock,
    OddsObservation,
    OddsQuote,
    ReplayCandidateSelection,
    ReplayConfig,
    ReplayEventRecord,
    ReplayTraceManifest,
    WalkForwardConfig,
    stable_hash,
)


def test_stable_hash_is_order_independent():
    a = stable_hash({"x": 1, "y": 2})
    b = stable_hash({"y": 2, "x": 1})
    assert a == b
    assert a != stable_hash({"x": 1, "y": 3})


def test_historical_event_roundtrip_and_stable_id():
    ev = HistoricalEvent(
        event_id="e1",
        league="NFL",
        sport_family="american_football",
        start_time="2023-09-10T17:00:00+00:00",
        home_team="Kansas City Chiefs",
        away_team="Detroit Lions",
        source_name="csv_games",
    )
    restored = HistoricalEvent.model_validate(ev.model_dump(mode="json"))
    assert restored == ev
    # stable_event_id ignores the supplied event_id and is deterministic
    assert ev.stable_event_id() == restored.stable_event_id()


def test_event_rejects_unknown_field():
    with pytest.raises(ValidationError):
        HistoricalEvent(
            event_id="e1",
            league="NFL",
            sport_family="american_football",
            start_time="2023-09-10T17:00:00+00:00",
            home_team="A",
            away_team="B",
            source_name="csv_games",
            bogus_field=1,
        )


@pytest.mark.parametrize(
    "home,away,expected",
    [(24, 17, "home_win"), (10, 21, "away_win"), (3, 3, "draw"), (None, 1, None)],
)
def test_outcome_result_derivation(home, away, expected):
    assert HistoricalOutcome.derive_result(home, away) == expected


def test_prop_outcome_requires_stat_value_unless_void():
    assert HistoricalPropOutcome(player_name="P", stat_type="pts", stat_value=1.0).stat_value == 1.0
    assert HistoricalPropOutcome(player_name="P", stat_type="pts", void=True).stat_value is None
    with pytest.raises(ValidationError):
        HistoricalPropOutcome(player_name="P", stat_type="pts")
    with pytest.raises(ValidationError):
        HistoricalPropOutcome(player_name="P", stat_type="pts", stat_value=1.0, void=True)


def test_market_snapshot_hash_excludes_closing():
    base = dict(
        event_id="e1",
        decision_time="2023-09-10T16:00:00+00:00",
        decision=[
            OddsQuote(market="moneyline", selection_descriptor="home", odds=-150),
        ],
    )
    snap_no_close = HistoricalMarketSnapshot(**base)
    snap_with_close = HistoricalMarketSnapshot(
        **base,
        closing=[OddsQuote(market="moneyline", selection_descriptor="home", odds=-180)],
    )
    # Closing odds are CLV-only and must not change the decision-snapshot identity.
    assert snap_no_close.compute_hash() == snap_with_close.compute_hash()


def test_market_snapshot_decision_lookup():
    snap = HistoricalMarketSnapshot(
        event_id="e1",
        decision_time="t",
        decision=[OddsQuote(market="total", selection_descriptor="over_45.5", odds=-110, line=45.5)],
    )
    q = snap.decision_quote("total", "over_45.5")
    assert q is not None and q.line == 45.5
    assert snap.decision_quote("total", "under_45.5") is None


def test_feature_snapshot_hash_is_key_order_independent():
    s1 = HistoricalFeatureSnapshot(
        event_id="e1",
        league="NFL",
        sport_family="american_football",
        decision_time="t",
        home_context={"off_rating": 1.0, "def_rating": 2.0},
        away_context={"def_rating": 2.0, "off_rating": 1.0},
        game_context={"is_playoff": False, "rest_days": 7},
    )
    s2 = HistoricalFeatureSnapshot(
        event_id="e1",
        league="NFL",
        sport_family="american_football",
        decision_time="t",
        home_context={"def_rating": 2.0, "off_rating": 1.0},
        away_context={"off_rating": 1.0, "def_rating": 2.0},
        game_context={"rest_days": 7, "is_playoff": False},
    )
    assert s1.compute_hash() == s2.compute_hash()


def test_replay_config_hash_stable_and_db_path_irrelevant():
    c1 = ReplayConfig(dataset_manifest_id="m1", backtest_db_path="/tmp/a.db", code_version="v1")
    c2 = ReplayConfig(dataset_manifest_id="m1", backtest_db_path="/tmp/b.db", code_version="v1")
    # db path is operational, not part of the determinism key
    assert c1.config_hash() == c2.config_hash()
    c3 = ReplayConfig(dataset_manifest_id="m2", backtest_db_path="/tmp/a.db", code_version="v1")
    assert c1.config_hash() != c3.config_hash()
    c4 = ReplayConfig(
        dataset_manifest_id="m1",
        backtest_db_path="/tmp/a.db",
        code_version="v1",
        odds_timing_class="closing_only",
    )
    assert c1.config_hash() != c4.config_hash()


def test_replay_observation_and_manifest_roundtrip():
    obs = OddsObservation(
        event_key="ek",
        market="moneyline",
        selection_descriptor="home",
        odds=-150,
        tier_hint="closing",
    )
    assert OddsObservation.model_validate(obs.model_dump()) == obs

    manifest = ReplayTraceManifest(
        replay_id="r1",
        dataset_manifest_id="m1",
        league="NFL",
        records=[
            ReplayEventRecord(
                event_id="e1",
                trace_id="t1",
                decision_time="t",
                leakage_status="clean",
            )
        ],
    )
    restored = ReplayTraceManifest.model_validate(manifest.model_dump(mode="json"))
    assert restored.records[0].trace_id == "t1"


def test_walk_forward_config_defaults():
    cfg = WalkForwardConfig()
    assert cfg.mode == "expanding"
    assert cfg.min_slice_samples == 30
    assert cfg.markets == ["game"]


def test_fold_and_report_roundtrip():
    fold = FoldResult(
        fold_index=0,
        train_end="2023-06-01T00:00:00+00:00",
        test_start="2023-06-01T00:00:00+00:00",
        test_end="2023-07-01T00:00:00+00:00",
        n_train=120,
        n_test=30,
        metrics_by_market={"game": MetricBlock(raw_brier=0.25, calibrated_brier=0.22, n=30)},
        health=HealthBlock(missing_odds_rate=0.1),
        frozen_profiles=[
            FrozenProfileRef(
                market="game",
                context_slice=None,
                method="isotonic",
                profile_id="iso_nfl_v1",
                profile_hash="abc123",
                sample_size=120,
                params_snapshot={"calibration_map": {"0.55": 0.53}},
            )
        ],
    )
    report = BacktestReport(
        manifest_id="m1",
        replay_id="r1",
        league="NFL",
        walk_forward_config=WalkForwardConfig(),
        folds=[fold],
        aggregate_metrics_by_market={"game": MetricBlock(raw_brier=0.25, calibrated_brier=0.22, n=30)},
    )
    restored = BacktestReport.model_validate(report.model_dump(mode="json"))
    assert restored.folds[0].frozen_profiles[0].profile_hash == "abc123"
    assert restored.aggregate_metrics_by_market["game"].calibrated_brier == 0.22


def test_candidate_selection_roundtrip():
    sel = ReplayCandidateSelection(
        replay_id="r1",
        event_id="e1",
        trace_id="t1",
        market="moneyline",
        selection_descriptor="home",
        raw_prob=0.6,
        calibrated_prob=0.57,
        profile_id="iso_nfl_v1",
        profile_hash="abc123",
        decision_odds=-150,
        decision_time="t",
        edge=0.04,
        ledger_id="L1",
        clv=0.012,
    )
    restored = ReplayCandidateSelection.model_validate(sel.model_dump(mode="json"))
    assert restored.ledger_id == "L1"
    assert restored.clv == 0.012
