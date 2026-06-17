"""Walk-forward window selection is chronological and never leaks the future."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from omega.historical.contracts import WalkForwardConfig
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

    cfg = WalkForwardConfig(mode="expanding", test_window_days=20, step_days=20, min_train_samples=30)
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
