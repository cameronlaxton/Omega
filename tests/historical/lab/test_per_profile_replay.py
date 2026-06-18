"""Per-profile replay: the calibration override seam + candidate-vs-incumbent betting."""

from __future__ import annotations

import pytest

from omega.core.calibration.probability import (
    apply_calibration,
    calibration_registry_override,
)
from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus
from omega.core.calibration.registry import CalibrationRegistry
from omega.historical.contracts import (
    BettingBlock,
    HistoricalEvent,
    HistoricalOutcome,
    OddsObservation,
    ReplayConfig,
)
from omega.historical.identity import event_key
from omega.historical.lab.per_profile_replay import compare_profiles
from omega.historical.lab.schemas import Window
from omega.historical.normalize import parse_datetime_utc
from omega.historical.replay import ReplayDataset

LEAGUE = "NFL"
FAMILY = "american_football"


def _profile(method: str, params: dict, *, league=LEAGUE, market="game") -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=f"{method}_{league}_{market}",
        version=1,
        method=method,
        league=league,
        market=market,
        status=ProfileStatus.PRODUCTION,
        params=params,
        training_window="t",
        sample_size=100,
        dataset_hash="h",
        metrics={},
    )


# --- the override seam (behavior-preserving) ------------------------------


def test_override_routes_to_isolated_registry(tmp_path):
    reg_path = tmp_path / "iso_profiles.json"
    CalibrationRegistry(path=str(reg_path)).register(
        _profile("shrinkage", {"shrink_factor": 0.5}, league="TESTLG")
    )

    # With the override active, the isolated profile is applied: 0.5 + 0.5*(0.9-0.5)=0.7.
    with calibration_registry_override(str(reg_path)):
        assert apply_calibration(0.9, league="TESTLG", market="game") == pytest.approx(0.7)

    # Outside the override, TESTLG has no production profile → static identity (0.9).
    assert apply_calibration(0.9, league="TESTLG", market="game") == pytest.approx(0.9)


def test_override_is_scoped_and_restored(tmp_path):
    reg_path = tmp_path / "iso_profiles.json"
    CalibrationRegistry(path=str(reg_path)).register(
        _profile("shrinkage", {"shrink_factor": 0.5}, league="TESTLG")
    )
    before = apply_calibration(0.9, league="TESTLG", market="game")
    with calibration_registry_override(str(reg_path)):
        pass
    after = apply_calibration(0.9, league="TESTLG", market="game")
    assert before == after  # ContextVar restored on exit


# --- per-profile replay through the real engine ---------------------------


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


def _dataset() -> ReplayDataset:
    e1 = _event("2023-09-10", "Team A", "Team B")
    e2 = _event("2023-09-17", "Team C", "Team A")
    e3 = _event("2023-09-24", "Team B", "Team C")
    target = _event("2023-10-01", "Team A", "Team C")
    events = [e1, e2, e3, target]
    outcomes = {
        e1.event_id: HistoricalOutcome(event_id=e1.event_id, home_score=24, away_score=17, result="home_win"),
        e2.event_id: HistoricalOutcome(event_id=e2.event_id, home_score=20, away_score=27, result="away_win"),
        e3.event_id: HistoricalOutcome(event_id=e3.event_id, home_score=30, away_score=21, result="home_win"),
        target.event_id: HistoricalOutcome(event_id=target.event_id, home_score=28, away_score=24, result="home_win"),
    }
    obs: list[OddsObservation] = []
    for ev in (e1, e2, e3):
        for sd, price in (("home", -110), ("away", -110)):
            obs.append(OddsObservation(event_key=ev.event_id, market="moneyline", selection_descriptor=sd, odds=price))
    obs.append(OddsObservation(event_key=target.event_id, market="moneyline", selection_descriptor="home", odds=200))
    obs.append(OddsObservation(event_key=target.event_id, market="moneyline", selection_descriptor="away", odds=200))
    obs.append(
        OddsObservation(
            event_key=target.event_id, market="moneyline",
            selection_descriptor="home", odds=-180, tier_hint="closing",
        )
    )
    return ReplayDataset(events=events, outcomes=outcomes, odds=ReplayDataset.group_odds(obs))


def _config(tmp_path) -> ReplayConfig:
    return ReplayConfig(
        dataset_manifest_id="m-test",
        backtest_db_path=str(tmp_path / "base.db"),
        enable_staking=True,
        odds_timing_class="decision_time_safe",
        n_iterations=200,
    )


_HOLDOUT = Window(start="2023-10-01", end="2023-10-01")


def test_compare_runs_through_real_engine(tmp_path):
    art = compare_profiles(
        _dataset(),
        league=LEAGUE,
        base_config=_config(tmp_path),
        candidate=_profile("none", {}),
        incumbent=_profile("shrinkage", {"shrink_factor": 0.3}),
        holdout_window=_HOLDOUT,
        work_dir=tmp_path / "ppr",
    )
    assert art["basis"] == "per_profile_replay"
    assert art["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")
    assert "candidate" in art and "incumbent" in art
    assert isinstance(art["candidate"]["n_bets"], int)


def test_compare_is_deterministic(tmp_path):
    kw = dict(
        league=LEAGUE,
        base_config=_config(tmp_path),
        candidate=_profile("none", {}),
        incumbent=None,
        holdout_window=_HOLDOUT,
    )
    a = compare_profiles(_dataset(), work_dir=tmp_path / "a", **kw)
    b = compare_profiles(_dataset(), work_dir=tmp_path / "b", **kw)
    assert a["verdict"] == b["verdict"]
    assert a["candidate"] == b["candidate"]


def test_compare_inconclusive_when_candidate_clv_missing(tmp_path, monkeypatch):
    import omega.historical.lab.per_profile_replay as ppr

    def _fake_replay_window_betting(*args, tag, **kwargs):
        if tag == "candidate":
            return BettingBlock(n_bets=3, roi=0.12, avg_clv=None)
        return BettingBlock(n_bets=3, roi=0.0, avg_clv=0.01)

    monkeypatch.setattr(ppr, "replay_window_betting", _fake_replay_window_betting)
    art = ppr.compare_profiles(
        _dataset(),
        league=LEAGUE,
        base_config=_config(tmp_path),
        candidate=_profile("none", {}),
        incumbent=_profile("shrinkage", {"shrink_factor": 0.3}),
        holdout_window=_HOLDOUT,
        work_dir=tmp_path / "ppr",
    )
    assert art["verdict"] == "INCONCLUSIVE"
    assert art["reason"] == "candidate_betting_metrics_unavailable"


def test_compare_inconclusive_when_betting_incumbent_clv_missing(tmp_path, monkeypatch):
    import omega.historical.lab.per_profile_replay as ppr

    def _fake_replay_window_betting(*args, tag, **kwargs):
        if tag == "candidate":
            return BettingBlock(n_bets=3, roi=0.12, avg_clv=0.02)
        return BettingBlock(n_bets=3, roi=0.0, avg_clv=None)

    monkeypatch.setattr(ppr, "replay_window_betting", _fake_replay_window_betting)
    art = ppr.compare_profiles(
        _dataset(),
        league=LEAGUE,
        base_config=_config(tmp_path),
        candidate=_profile("none", {}),
        incumbent=_profile("shrinkage", {"shrink_factor": 0.3}),
        holdout_window=_HOLDOUT,
        work_dir=tmp_path / "ppr",
    )
    assert art["verdict"] == "INCONCLUSIVE"
    assert art["reason"] == "incumbent_betting_metrics_unavailable"
