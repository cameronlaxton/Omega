"""Orchestrator integration: seed an isolated store, run the lab, inspect artifacts."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from omega.core.calibration.registry import CalibrationRegistry
from omega.historical.contracts import BettingBlock, ReplayCandidateSelection
from omega.historical.lab.orchestrator import _clv_artifact, run_lab_from_store
from omega.historical.lab.runner import LabCommandRunner
from omega.historical.lab.schemas import HistoricalLabRun, Window
from omega.trace.store import TraceStore

UTC = timezone.utc
BASE = datetime(2023, 1, 1, tzinfo=UTC)

TRAIN = Window(start="2023-01-01", end="2023-04-30")
VALID = Window(start="2023-05-01", end="2023-06-15")
HOLD = Window(start="2023-06-16", end="2023-07-31")

_CLEAN_PROV = {"code_version": "omega-test", "git_commit": "abc123", "working_tree_dirty": False}


def _persist(store: TraceStore, i: int, p_home: float, home_win: bool, dt: str) -> None:
    tid = f"t{i}"
    store.persist(
        {
            "trace_id": tid,
            "run_id": tid,
            "timestamp": dt,
            "decision_time": dt,
            "historical_replay": True,
            "execution_mode": "historical_replay",
            "replay_id": "r1",
            "league": "FIFA_INTL",
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
    )
    hs, as_ = (2, 1) if home_win else (1, 2)
    store.attach_outcome(tid, hs, as_, source="backtest")


def _seed(store: TraceStore, n: int = 180) -> None:
    rng = random.Random(0)
    for i in range(n):
        dt = (BASE + timedelta(days=i)).isoformat()
        p = 45 + (i % 6) * 5  # 45..70
        _persist(store, i, p, rng.random() < p / 100.0, dt)


def _run(tmp_path, *, auto_promote=False):
    store = TraceStore(db_path=str(tmp_path / "replay.db"))
    _seed(store)
    reg = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    out_dir = tmp_path / "lab_runs" / "lab_test"
    runner = LabCommandRunner(out_dir / "command_log.jsonl")
    try:
        result = run_lab_from_store(
            store,
            runner=runner,
            out_dir=out_dir,
            lab_run_id="lab_test",
            league="FIFA_INTL",
            plane="game",
            dataset_manifest_id="m1",
            replay_id="r1",
            train_window=TRAIN,
            validation_window=VALID,
            holdout_window=HOLD,
            replay_db_path=str(tmp_path / "replay.db"),
            replay_config_hash="cfg",
            dataset_hash="ds",
            registry=reg,
            live_store=None,
            auto_promote=auto_promote,
            provenance_info=_CLEAN_PROV,
        )
    finally:
        store.close()
    return result, reg, out_dir


def test_orchestrator_writes_all_artifacts(tmp_path):
    result, _reg, out_dir = _run(tmp_path)
    for key in (
        "lab_run",
        "attempted_variants",
        "promotion_evidence",
        "backtest_report",
        "backtest_parity",
        "historical_live_parity",
        "registry_audit",
        "clv_walk_forward",
    ):
        assert key in result.paths
    assert (out_dir / "LAB_RUN.json").exists()
    assert (out_dir / "command_log.jsonl").exists()
    # LAB_RUN.json round-trips through the schema (validators all pass).
    restored = HistoricalLabRun.model_validate(json.loads((out_dir / "LAB_RUN.json").read_text()))
    assert restored.lab_run_id == "lab_test"
    assert restored.attempted_variant_count == len(result.ledger.variants)


def test_orchestrator_selects_winner_and_seals(tmp_path):
    result, _reg, _ = _run(tmp_path)
    assert result.ledger.selected is not None
    assert result.lab_run.holdout_sealed is True
    assert result.lab_run.holdout_access_count == 1


def test_orchestrator_does_not_promote_without_live_or_clv(tmp_path):
    # No live store + no closing lines → live INCONCLUSIVE / CLV INCONCLUSIVE.
    result, reg, _ = _run(tmp_path, auto_promote=True)
    assert result.lab_run.promotion_status != "promoted"
    # Auto-promote armed but refused → nothing reached production.
    assert all(p.status.value != "production" for p in reg.list_profiles())


def test_orchestrator_evidence_only_leaves_registry_clean(tmp_path):
    _result, reg, _ = _run(tmp_path, auto_promote=False)
    assert reg.list_profiles() == []  # grid + evidence-only never registers


# --- CLV verdict from real betting metrics --------------------------------


def _report(bb):
    return SimpleNamespace(aggregate_betting=bb)


def test_clv_inconclusive_without_betting():
    assert _clv_artifact(None)["verdict"] == "INCONCLUSIVE"
    assert _clv_artifact(_report(None))["verdict"] == "INCONCLUSIVE"
    assert _clv_artifact(_report(BettingBlock(n_bets=0)))["verdict"] == "INCONCLUSIVE"
    assert _clv_artifact(_report(BettingBlock(n_bets=5, avg_clv=None)))["verdict"] == "INCONCLUSIVE"


def test_clv_pass_when_beats_close_and_profitable():
    a = _clv_artifact(_report(BettingBlock(n_bets=8, avg_clv=0.02, roi=0.1, net_pnl=12.0)))
    assert a["verdict"] == "PASS"
    assert a["basis"] == "absolute_floor"
    assert a["n_bets"] == 8


def test_clv_fail_when_clv_negative():
    assert (
        _clv_artifact(_report(BettingBlock(n_bets=8, avg_clv=-0.01, roi=0.1)))["verdict"] == "FAIL"
    )
    assert (
        _clv_artifact(_report(BettingBlock(n_bets=8, avg_clv=0.01, roi=-0.2)))["verdict"] == "FAIL"
    )


def test_orchestrator_threads_selections_into_real_betting(tmp_path):
    # Real selections + closing lines must flow through to betting_metrics so CLV
    # is a graded verdict (PASS/FAIL), not the no-bets INCONCLUSIVE default.
    store = TraceStore(db_path=str(tmp_path / "replay.db"))
    rng = random.Random(0)
    sels = []
    for i in range(180):
        dt = (BASE + timedelta(days=i)).isoformat()
        p = 45 + (i % 6) * 5
        _persist(store, i, p, rng.random() < p / 100.0, dt)
        store.attach_closing_line(f"t{i}", "moneyline", "home_moneyline", -150, None, dt, "test")
        sels.append(
            ReplayCandidateSelection(
                replay_id="r1",
                event_id=f"t{i}",
                trace_id=f"t{i}",
                market="moneyline",
                selection_descriptor="home_moneyline",
                raw_prob=p / 100.0,
                decision_time=dt,
                decision_odds=-130,
                decision_line=None,
                stake_amount=10.0,
            )
        )
    reg = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    out_dir = tmp_path / "lab_runs" / "lab_clv"
    runner = LabCommandRunner(out_dir / "command_log.jsonl")
    try:
        run_lab_from_store(
            store,
            runner=runner,
            out_dir=out_dir,
            lab_run_id="lab_clv",
            league="FIFA_INTL",
            plane="game",
            dataset_manifest_id="m1",
            replay_id="r1",
            train_window=TRAIN,
            validation_window=VALID,
            holdout_window=HOLD,
            replay_db_path=str(tmp_path / "replay.db"),
            registry=reg,
            live_store=None,
            auto_promote=False,
            provenance_info=_CLEAN_PROV,
            selections=sels,
        )
    finally:
        store.close()
    clv = json.loads((out_dir / "clv_walk_forward.json").read_text())
    assert clv["n_bets"] > 0
    assert clv["verdict"] in ("PASS", "FAIL")  # graded, not INCONCLUSIVE
    assert clv["basis"] == "absolute_floor"


# --- per-profile replay gating (the 2x replay only runs when odds exist) ---


def _run_with_dataset(tmp_path, dataset, replay_config):
    store = TraceStore(db_path=str(tmp_path / "replay.db"))
    _seed(store)
    reg = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    out_dir = tmp_path / "lab_runs" / "lab_g"
    runner = LabCommandRunner(out_dir / "command_log.jsonl")
    try:
        run_lab_from_store(
            store,
            runner=runner,
            out_dir=out_dir,
            lab_run_id="lab_g",
            league="FIFA_INTL",
            plane="game",
            dataset_manifest_id="m1",
            replay_id="r1",
            train_window=TRAIN,
            validation_window=VALID,
            holdout_window=HOLD,
            replay_db_path=str(tmp_path / "replay.db"),
            registry=reg,
            live_store=None,
            auto_promote=False,
            provenance_info=_CLEAN_PROV,
            dataset=dataset,
            replay_config=replay_config,
        )
    finally:
        store.close()
    return out_dir


def test_per_profile_replay_runs_when_dataset_has_odds(tmp_path, monkeypatch):
    import omega.historical.lab.orchestrator as orch

    seen = {}

    def _fake_compare(dataset, **kw):
        seen["called"] = True
        return {"verdict": "PASS", "basis": "per_profile_replay", "n_bets": 2}

    monkeypatch.setattr(orch, "compare_profiles", _fake_compare)
    out_dir = _run_with_dataset(tmp_path, SimpleNamespace(odds={"e": [1]}), object())
    clv = json.loads((out_dir / "clv_walk_forward.json").read_text())
    assert seen.get("called") is True
    assert clv["basis"] == "per_profile_replay"


def test_per_profile_replay_skipped_without_odds(tmp_path, monkeypatch):
    import omega.historical.lab.orchestrator as orch

    seen = {}

    def _fake_compare(dataset, **kw):
        seen["called"] = True
        return {}

    monkeypatch.setattr(orch, "compare_profiles", _fake_compare)
    out_dir = _run_with_dataset(tmp_path, SimpleNamespace(odds={}), object())
    clv = json.loads((out_dir / "clv_walk_forward.json").read_text())
    assert seen.get("called") is None  # gated off → 2x replay never runs
    assert clv.get("basis") != "per_profile_replay"
