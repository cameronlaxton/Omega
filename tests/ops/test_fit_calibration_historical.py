"""omega-fit-calibration historical-replay flags + date-windowed split.

Covers: the fitter selects historical_replay traces from --historical-db, excludes
them by default, enforces the same-season leakage guard, and partitions disjoint
date windows by event decision_time (not the replay run timestamp).
"""

from __future__ import annotations

from types import SimpleNamespace

from omega.historical.contracts import HistoricalEvent, HistoricalOutcome, ReplayConfig
from omega.historical.identity import event_key
from omega.historical.normalize import parse_datetime_utc
from omega.historical.replay import ReplayDataset, ReplayEngine
from omega.ops import fit_calibration
from omega.trace.store import TraceStore

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


def _build_history_db(path: str) -> int:
    """Replay a small NFL slate into a dedicated DB. Returns persisted count."""
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
    ds = ReplayDataset(events=events, outcomes=outcomes, odds={})
    store = TraceStore(db_path=path)
    try:
        cfg = ReplayConfig(dataset_manifest_id="m", backtest_db_path=path, n_iterations=200)
        res = ReplayEngine(store, cfg).run(ds, replay_id="r", league=LEAGUE)
    finally:
        store.close()
    return res.n_persisted


def _ns(**kw) -> SimpleNamespace:
    base = dict(
        league=LEAGUE,
        db=None,
        historical_db=None,
        historical_only=False,
        include_historical=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# --- pure helpers -----------------------------------------------------------


def test_in_window_bounds():
    assert fit_calibration._in_window("2023-09-15", "2023-09-01", "2023-09-30") is True
    assert fit_calibration._in_window("2023-10-01", "2023-09-01", "2023-09-30") is False
    assert fit_calibration._in_window("2023-08-31", "2023-09-01", None) is False
    assert fit_calibration._in_window("2023-09-15", None, None) is True
    assert fit_calibration._in_window("", None, None) is False


def test_decision_date_prefers_event_decision_time():
    assert fit_calibration._decision_date({"decision_time": "2023-09-10T17:00:00Z"}) == "2023-09-10"
    # Falls back to the analysis timestamp for live traces.
    assert fit_calibration._decision_date({"timestamp": "2024-01-02T00:00:00Z"}) == "2024-01-02"


# --- selection --------------------------------------------------------------


def test_historical_only_selects_replay_traces(tmp_path):
    hist = str(tmp_path / "hist.db")
    assert _build_history_db(hist) == 4

    loaded = fit_calibration._load_graded_traces(
        _ns(db=str(tmp_path / "live.db"), historical_db=hist, historical_only=True)
    )
    assert len(loaded) == 4
    assert all(t.get("execution_mode") == "historical_replay" for t in loaded)
    # Every loaded trace carries the event decision_time used for windowing.
    assert all(fit_calibration._decision_date(t).startswith("2023-") for t in loaded)


def test_default_excludes_historical(tmp_path):
    # An empty live DB with no historical flags yields zero graded traces.
    loaded = fit_calibration._load_graded_traces(_ns(db=str(tmp_path / "live.db")))
    assert loaded == []


# --- CLI guards (return codes) ---------------------------------------------


def test_cli_dry_run_sees_historical(tmp_path):
    hist = str(tmp_path / "hist.db")
    _build_history_db(hist)
    rc = fit_calibration.main(
        [
            "--league", LEAGUE,
            "--db", str(tmp_path / "live.db"),
            "--historical-only",
            "--historical-db", hist,
            "--dry-run",
            "--shadow-min-samples", "1",
            "--min-samples", "1",
        ]
    )
    assert rc == 0


def test_cli_default_empty_live_errors(tmp_path):
    rc = fit_calibration.main(["--league", LEAGUE, "--db", str(tmp_path / "live.db"), "--dry-run"])
    assert rc == 1


def test_same_season_guard_blocks(tmp_path):
    hist = str(tmp_path / "hist.db")
    _build_history_db(hist)
    common = [
        "--league", LEAGUE,
        "--db", str(tmp_path / "live.db"),
        "--historical-only",
        "--historical-db", hist,
        "--dry-run",
        "--shadow-min-samples", "1",
        "--min-samples", "1",
    ]
    # Overlapping windows (holdout_start < train_end) → blocked.
    rc_blocked = fit_calibration.main(
        common + ["--train-end", "2023-09-30", "--holdout-start", "2023-09-01"]
    )
    assert rc_blocked == 1
    # Explicit shadow override → allowed.
    rc_allowed = fit_calibration.main(
        common
        + ["--train-end", "2023-09-30", "--holdout-start", "2023-09-01", "--allow-same-season-shadow"]
    )
    assert rc_allowed == 0


def test_disjoint_date_windows_ok(tmp_path):
    hist = str(tmp_path / "hist.db")
    _build_history_db(hist)
    rc = fit_calibration.main(
        [
            "--league", LEAGUE,
            "--db", str(tmp_path / "live.db"),
            "--historical-only",
            "--historical-db", hist,
            "--dry-run",
            "--shadow-min-samples", "1",
            "--min-samples", "1",
            "--train-end", "2023-09-20",
            "--holdout-start", "2023-09-21",
        ]
    )
    assert rc == 0


def test_mutually_exclusive_flags(tmp_path):
    rc = fit_calibration.main(
        [
            "--league", LEAGUE,
            "--historical-only",
            "--include-historical",
            "--historical-db", str(tmp_path / "hist.db"),
            "--dry-run",
        ]
    )
    assert rc == 1
