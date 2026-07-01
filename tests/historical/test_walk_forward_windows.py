"""Walk-forward window selection is chronological and never leaks the future."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from omega.historical.contracts import ReplayCandidateSelection, WalkForwardConfig
from omega.historical.walk_forward import _generate_folds, partition_fold, run_walk_forward

UTC = timezone.utc
BASE = datetime(2023, 1, 1, tzinfo=UTC)


def _graded(i: int, dt: str | None = None) -> dict:
    return {
        "_dt": dt or (BASE + timedelta(days=i)).isoformat(),
        "predictions": {"home_win_prob": 55},
        "_outcome": {"result": "home_win"},
        "context_labels": {"is_playoff": False, "rest_days": 7},
        "event_id": f"e{i}",
    }


def test_generate_folds_expanding():
    dates = [(BASE + timedelta(days=i)).isoformat() for i in range(60)]
    cfg = WalkForwardConfig(test_window_days=20, step_days=20)
    folds = _generate_folds(dates, cfg)
    assert len(folds) >= 2
    # The first 20-day window is reserved for training.
    assert folds[0][0] == BASE + timedelta(days=20)


def test_partition_excludes_future_from_train_and_test():
    graded = [_graded(i) for i in range(40)]
    future = _graded(999, dt=(BASE + timedelta(days=500)).isoformat())
    graded.append(future)
    ts = (BASE + timedelta(days=20)).isoformat()
    te = (BASE + timedelta(days=40)).isoformat()

    train, test, _ = partition_fold(graded, ts, te, WalkForwardConfig())
    assert all(t["_dt"] < ts for t in train)
    assert future not in train
    assert all(ts <= t["_dt"] < te for t in test)
    assert future not in test


def test_rolling_window_bounds_training():
    graded = [_graded(i) for i in range(60)]
    ts = (BASE + timedelta(days=40)).isoformat()
    te = (BASE + timedelta(days=50)).isoformat()
    cfg = WalkForwardConfig(mode="rolling", train_window_days=10)
    train, _test, tr_start = partition_fold(graded, ts, te, cfg)
    assert tr_start == (BASE + timedelta(days=30)).isoformat()
    assert all(tr_start <= t["_dt"] < ts for t in train)


def _persist(store, i: int, p_home: float, home_win: bool, dt: str) -> None:
    tid = f"t{i}"
    trace = {
        "trace_id": tid,
        "run_id": tid,
        "timestamp": dt,
        "decision_time": dt,
        "historical_replay": True,
        "league": "NFL",
        "kind": "game",
        "matchup": "X @ Y",
        "event_id": tid,
        "predictions": {"home_win_prob": p_home, "away_win_prob": 100 - p_home},
        "context_labels": {"is_playoff": False, "rest_days": 7},
        "input_snapshot": {"game_context": {"is_playoff": False, "rest_days": 7}},
        "trace_quality": {
            "calibration_eligible": True,
            "context_source": "provided",
            "identity_status": "complete",
            "calibration_exclusion_reasons": [],
        },
    }
    store.persist(trace)
    hs, as_ = (2, 1) if home_win else (1, 2)
    store.attach_outcome(tid, hs, as_, source="backtest")


def test_walk_forward_runs_and_separates_metrics(backtest_store):
    rng = random.Random(0)
    for i in range(90):
        dt = (BASE + timedelta(days=i)).isoformat()
        p = 50 + (i % 5) * 5  # 50..70
        home_win = rng.random() < p / 100.0
        _persist(backtest_store, i, p, home_win, dt)

    cfg = WalkForwardConfig(
        mode="expanding", test_window_days=20, step_days=20, min_train_samples=30
    )
    report = run_walk_forward(
        backtest_store, config=cfg, league="NFL", replay_id="r", dataset_manifest_id="m"
    )

    assert report.folds, "expected at least one fold"
    for f in report.folds:
        assert f.n_train >= 30
        # training ends exactly at the test window start (no overlap into the future)
        assert f.train_end == f.test_start
        gm = f.metrics_by_market["game"]
        assert gm.raw_brier is not None
        assert gm.calibrated_brier is not None
        assert gm.n == f.n_test
        # at least the base profile was frozen and recorded with a hash
        assert any(p.profile_hash for p in f.frozen_profiles)
    assert "game" in report.aggregate_metrics_by_market


def test_walk_forward_emits_model_vs_market_and_scorecard(backtest_store):
    """Selections-bearing run activates the issue-#28 incremental-edge metrics + scorecard."""
    for i in range(40):
        dt = (BASE + timedelta(days=i)).isoformat()
        _persist(backtest_store, i, 55, i % 2 == 0, dt)

    # Moneyline selections: the model sits well above the +100 (0.50) market on the home
    # side, so every decision is divergent — the scorecard's "game" plane must populate.
    sels = [
        ReplayCandidateSelection(
            replay_id="r",
            event_id=f"t{i}",
            trace_id=f"t{i}",
            market="moneyline",
            selection_descriptor="home",
            raw_prob=0.55,
            calibrated_prob=0.60,
            decision_odds=100,
            decision_time=(BASE + timedelta(days=i)).isoformat(),
            stake_amount=100.0,
        )
        for i in range(40)
    ]

    cfg = WalkForwardConfig(test_window_days=20, step_days=20, min_train_samples=10)
    report = run_walk_forward(
        backtest_store,
        config=cfg,
        league="NFL",
        replay_id="r",
        dataset_manifest_id="m",
        selections=sels,
    )

    # Aggregate model-vs-market block exists for the game plane and saw every decision.
    assert "game" in report.aggregate_model_vs_market_by_market
    mvm = report.aggregate_model_vs_market_by_market["game"]
    assert mvm.n == 40
    assert mvm.n_divergent == 40  # 0.60 vs 0.50 is past the 0.02 threshold for all

    # Scorecard fuses the plane and carries the coherence flag.
    game_rows = [r for r in report.scorecard if r.market == "game"]
    assert len(game_rows) == 1
    assert game_rows[0].mean_signed_divergence is not None
    assert isinstance(game_rows[0].clv_coherent, bool)

    # Per-fold block is populated too.
    assert any(f.model_vs_market is not None for f in report.folds)


def _persist_prop(store, i: int, over_prob: float, over_wins: bool, dt: str) -> None:
    """Persist a graded prop replay trace (kind='prop') with one over-side outcome."""
    tid = f"p{i}"
    trace = {
        "trace_id": tid,
        "run_id": tid,
        "timestamp": dt,
        "decision_time": dt,
        "historical_replay": True,
        "league": "NBA",
        "kind": "prop",
        "matchup": "Player pts",
        "event_id": tid,
        "predictions": {"over_prob": over_prob, "under_prob": 100 - over_prob},
        "input_snapshot": {"player_name": "P", "prop_type": "pts", "game_context": {}},
        "context_labels": {},
        "trace_quality": {
            "calibration_eligible": True,
            "context_source": "provided",
            "identity_status": "complete",
            "calibration_exclusion_reasons": [],
        },
    }
    store.persist(trace)
    # line=20; an over bet wins when the realized stat exceeds it.
    stat_value = 30.0 if over_wins else 10.0
    store.attach_prop_outcome(tid, "P", "pts", stat_value, 20.0, "over", source="backtest")


def test_walk_forward_prop_plane_produces_metrics(backtest_store):
    """The prop plane now flows through walk-forward (was skipped before Wave 2)."""
    rng = random.Random(1)
    for i in range(80):
        dt = (BASE + timedelta(days=i)).isoformat()
        p = 50 + (i % 5) * 6  # 50..74
        over_wins = rng.random() < p / 100.0
        _persist_prop(backtest_store, i, p, over_wins, dt)

    cfg = WalkForwardConfig(
        mode="expanding", test_window_days=20, step_days=20, min_train_samples=10, markets=["prop"]
    )
    report = run_walk_forward(
        backtest_store, config=cfg, league="NBA", replay_id="r", dataset_manifest_id="m"
    )

    assert report.folds, "expected at least one prop fold"
    assert "prop" in report.aggregate_metrics_by_market
    assert report.aggregate_metrics_by_market["prop"].raw_brier is not None
    # Later folds have enough train pairs to actually fit a prop calibration profile.
    assert any(
        ref.market == "prop" for f in report.folds for ref in f.frozen_profiles
    ), "expected at least one fold to freeze a prop calibration profile"


def _persist_cover(store, i: int, cover_prob: float, spread_home: float, margin: int, dt: str) -> None:
    """Persist a graded game trace carrying a home_cover_prob + a decision spread line."""
    tid = f"c{i}"
    hs, as_ = (20 + margin, 20) if margin >= 0 else (20, 20 - margin)  # home - away == margin
    trace = {
        "trace_id": tid,
        "run_id": tid,
        "timestamp": dt,
        "decision_time": dt,
        "historical_replay": True,
        "league": "NFL",
        "kind": "game",
        "matchup": "X @ Y",
        "event_id": tid,
        "predictions": {
            "home_win_prob": 55,
            "away_win_prob": 45,
            "home_cover_prob": cover_prob,
        },
        "odds_snapshot": {"spread_home": spread_home},
        "input_snapshot": {"odds": {"spread_home": spread_home}, "game_context": {}},
        "context_labels": {},
        "trace_quality": {
            "calibration_eligible": True,
            "context_source": "provided",
            "identity_status": "complete",
            "calibration_exclusion_reasons": [],
        },
    }
    store.persist(trace)
    store.attach_outcome(tid, hs, as_, source="backtest")


def test_walk_forward_cover_plane_produces_metrics(backtest_store):
    """The point-spread (cover) plane now flows through walk-forward (Wave 3a)."""
    rng = random.Random(2)
    for i in range(80):
        dt = (BASE + timedelta(days=i)).isoformat()
        cover_prob = 50 + (i % 5) * 6  # 50..74
        margin = rng.choice([-14, -7, -3, 4, 7, 14])  # vs -3.5 line: never a push
        _persist_cover(backtest_store, i, cover_prob, -3.5, margin, dt)

    cfg = WalkForwardConfig(
        mode="expanding", test_window_days=20, step_days=20, min_train_samples=10, markets=["cover"]
    )
    report = run_walk_forward(
        backtest_store, config=cfg, league="NFL", replay_id="r", dataset_manifest_id="m"
    )

    assert report.folds, "expected at least one cover fold"
    assert "cover" in report.aggregate_metrics_by_market
    assert report.aggregate_metrics_by_market["cover"].raw_brier is not None
    assert any(
        ref.market == "cover" for f in report.folds for ref in f.frozen_profiles
    ), "expected at least one fold to freeze a cover calibration profile"
