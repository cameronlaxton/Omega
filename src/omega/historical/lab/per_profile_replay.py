"""Candidate-vs-incumbent betting via per-profile replay through the real engine.

CLV/ROI depend on which bets the engine *selects* and how it *sizes* them, and the
engine applies calibration at selection time — so a faithful candidate-vs-incumbent
betting delta cannot be derived from a single replay. This module replays the
holdout window twice through the real :class:`ReplayEngine`, each time with a
specific profile activated in an **isolated** registry via
``calibration_registry_override`` (zero pollution of the production registry), then
compares the realized betting blocks through the single ``betting_metrics``
authority. No edge/staking/grading math is re-implemented here.

It only produces signal when the dataset carries decision odds; the orchestrator
gates the (2×) replay on ``dataset.odds`` so odds-less datasets stay INCONCLUSIVE
without paying the cost.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from omega.core.calibration.probability import calibration_registry_override
from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus
from omega.core.calibration.registry import CalibrationRegistry
from omega.historical.contracts import BettingBlock, ReplayConfig
from omega.historical.lab.schemas import Window
from omega.historical.metrics import betting_metrics
from omega.historical.replay import ReplayDataset, ReplayEngine
from omega.ops.fit_calibration import _in_window
from omega.trace.store import TraceStore


def _isolated_registry(path: Path, profile: CalibrationProfile | None) -> None:
    """Write an isolated registry whose sole production entry is ``profile`` (or empty)."""
    reg = CalibrationRegistry(path=str(path))
    if profile is not None:
        reg.register(profile.model_copy(update={"status": ProfileStatus.PRODUCTION}))


def _filter_max_date(dataset: ReplayDataset, max_date: str) -> ReplayDataset:
    """Drop events after ``max_date`` (keep pre-holdout history + holdout events)."""
    keep = {e.event_id for e in dataset.events if str(e.start_time)[:10] <= max_date}
    return dataclasses.replace(
        dataset,
        events=[e for e in dataset.events if e.event_id in keep],
        outcomes={k: v for k, v in dataset.outcomes.items() if k in keep},
        odds={k: v for k, v in dataset.odds.items() if k in keep},
    )


def replay_window_betting(
    dataset: ReplayDataset,
    *,
    league: str,
    base_config: ReplayConfig,
    profile: CalibrationProfile | None,
    work_dir: Path,
    tag: str,
    meter_window: Window,
) -> BettingBlock:
    """Replay ``dataset`` with ``profile`` active; grade selections in the meter window.

    ``profile=None`` replays against an empty isolated registry (the static-policy
    baseline), so the comparison never reads the production registry.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    reg_path = work_dir / f"{tag}_profiles.json"
    store_path = work_dir / f"{tag}.db"
    _isolated_registry(reg_path, profile)
    config = base_config.model_copy(
        update={"backtest_db_path": str(store_path), "enable_staking": True}
    )
    store = TraceStore(db_path=str(store_path))
    try:
        with calibration_registry_override(str(reg_path)):
            result = ReplayEngine(store, config).run(dataset, replay_id=f"ppr_{tag}", league=league)
        sels = [
            s
            for s in result.selections
            if _in_window(str(s.decision_time)[:10], meter_window.start, meter_window.end)
        ]
        graded = store.get_graded_traces(league=league, limit=1_000_000)
        outcomes_by_event = {
            t["event_id"]: t["_outcome"]
            for t in graded
            if t.get("event_id") and t.get("_outcome")
        }
        closing_by_trace = {s.trace_id: store.get_closing_lines(s.trace_id) for s in sels}
        return betting_metrics(sels, outcomes_by_event, closing_by_trace)
    finally:
        store.close()


def compare_profiles(
    dataset: ReplayDataset,
    *,
    league: str,
    base_config: ReplayConfig,
    candidate: CalibrationProfile,
    incumbent: CalibrationProfile | None,
    holdout_window: Window,
    work_dir: Path,
    roi_tol: float = 0.0,
    clv_tol: float = 0.0,
) -> dict[str, Any]:
    """Replay the holdout under candidate vs incumbent; return a CLV/ROI verdict.

    ``non_regression`` (and PASS) requires measured candidate ROI/CLV and, when
    the incumbent placed bets, measured incumbent ROI/CLV. INCONCLUSIVE when the
    candidate places no graded bets or closing-line CLV is unavailable.
    """
    filtered = _filter_max_date(dataset, holdout_window.end)
    cand = replay_window_betting(
        filtered,
        league=league,
        base_config=base_config,
        profile=candidate,
        work_dir=work_dir,
        tag="candidate",
        meter_window=holdout_window,
    )
    inc = replay_window_betting(
        filtered,
        league=league,
        base_config=base_config,
        profile=incumbent,
        work_dir=work_dir,
        tag="incumbent",
        meter_window=holdout_window,
    )

    if cand.n_bets == 0:
        return {
            "verdict": "INCONCLUSIVE",
            "basis": "per_profile_replay",
            "reason": "no_candidate_bets",
            "n_bets": 0,
            "incumbent_present": incumbent is not None,
            "candidate": cand.model_dump(),
            "incumbent": inc.model_dump(),
        }

    if cand.roi is None or cand.avg_clv is None:
        return {
            "verdict": "INCONCLUSIVE",
            "basis": "per_profile_replay",
            "reason": "candidate_betting_metrics_unavailable",
            "n_bets": cand.n_bets,
            "incumbent_present": incumbent is not None,
            "candidate": cand.model_dump(),
            "incumbent": inc.model_dump(),
        }

    if inc.n_bets > 0 and (inc.roi is None or inc.avg_clv is None):
        return {
            "verdict": "INCONCLUSIVE",
            "basis": "per_profile_replay",
            "reason": "incumbent_betting_metrics_unavailable",
            "n_bets": cand.n_bets,
            "incumbent_present": incumbent is not None,
            "candidate": cand.model_dump(),
            "incumbent": inc.model_dump(),
        }

    cr = cand.roi
    ir = inc.roi if inc.n_bets > 0 and inc.roi is not None else 0.0
    cc = cand.avg_clv
    ic = inc.avg_clv if inc.n_bets > 0 and inc.avg_clv is not None else 0.0
    non_regression = (cr >= ir - roi_tol) and (cc >= ic - clv_tol)
    return {
        "verdict": "PASS" if non_regression else "FAIL",
        "basis": "per_profile_replay",
        "non_regression": non_regression,
        "n_bets": cand.n_bets,
        "incumbent_present": incumbent is not None,
        "candidate_roi": cand.roi,
        "incumbent_roi": inc.roi,
        "candidate_avg_clv": cand.avg_clv,
        "incumbent_avg_clv": inc.avg_clv,
        "candidate": cand.model_dump(),
        "incumbent": inc.model_dump(),
    }
