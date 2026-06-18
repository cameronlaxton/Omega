"""Lab orchestrator — chain the existing engine into one auditable run.

``run_lab`` replays a dataset into an isolated DB (``omega-replay-history``) then
calls ``run_lab_from_store``, which does the net-new glue: load graded replay
traces → variant grid → holdout seal → walk-forward (betting/CLV) → backtest
parity → historical-live parity → registry audit → evidence + optional
auto-promote → write the four net-new artifacts. Every modeling call delegates to
the single existing implementation; nothing here recomputes a metric, fit, or
promotion gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omega.core.calibration.registry import CalibrationRegistry
from omega.historical.contracts import ReplayConfig, WalkForwardConfig, stable_hash
from omega.historical.lab import grid as grid_mod
from omega.historical.lab import provenance as provenance_mod
from omega.historical.lab.evidence import EvidenceContext, resolve
from omega.historical.lab.per_profile_replay import compare_profiles
from omega.historical.lab.report import render
from omega.historical.lab.runner import LabCommandRunner
from omega.historical.lab.schemas import (
    AttemptedVariantLedger,
    HistoricalLabRun,
    PromotionEvidenceBundle,
    Window,
    assert_consistent,
)
from omega.historical.lab.seal import seal_winner
from omega.historical.manifests import (
    load_dataset_manifest,
    load_normalized_dataset,
    load_replay_manifest,
    load_selections,
)
from omega.historical.replay import ReplayDataset
from omega.historical.walk_forward import run_walk_forward
from omega.ops.backtest_parity import evaluate_backtest_parity
from omega.ops.historical_live_parity import evaluate_parity
from omega.ops.replay_history import main as replay_history_main
from omega.paths import default_trace_db_path
from omega.trace.store import TraceStore

_WF_PLANES = {"game", "draw"}  # walk_forward _PAIR_FN supports game/draw only


@dataclass
class LabRunResult:
    lab_run: HistoricalLabRun
    ledger: AttemptedVariantLedger
    evidence: PromotionEvidenceBundle
    paths: dict[str, str]


def lab_dir(root: str | Path | None, lab_run_id: str) -> Path:
    base = Path(root) if root is not None else Path("var/historical")
    return base / "lab_runs" / lab_run_id


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _clv_artifact(wf_report: Any) -> dict[str, Any]:
    """A CLV/ROI verdict derived from the walk-forward ``BettingBlock``.

    The numbers come from the single ``betting_metrics`` authority (graded replay
    selections vs outcomes + closing lines) — nothing is recomputed here. Verdict:
    INCONCLUSIVE when no graded bets exist (the honest result for an odds-less
    dataset); otherwise PASS iff the placed bets beat the close on average
    (avg_clv >= 0) and did not lose money (roi >= 0), else FAIL.

    ``basis="absolute_floor"``: this is the fallback absolute non-loss/positive-CLV
    check used when the odds-bearing per-profile replay path is unavailable. A true
    incumbent delta is produced separately by ``compare_profiles`` when the replay
    dataset carries odds.
    """
    if wf_report is None or wf_report.aggregate_betting is None:
        return {"verdict": "INCONCLUSIVE", "reason": "walk_forward_betting_unavailable", "n_bets": 0}
    bb = wf_report.aggregate_betting
    if bb.n_bets == 0 or bb.avg_clv is None:
        return {"verdict": "INCONCLUSIVE", "reason": "no_graded_bets", "n_bets": bb.n_bets}
    verdict = "PASS" if (bb.avg_clv >= 0 and (bb.roi is None or bb.roi >= 0)) else "FAIL"
    return {
        "verdict": verdict,
        "basis": "absolute_floor",
        "avg_clv": bb.avg_clv,
        "roi": bb.roi,
        "net_pnl": bb.net_pnl,
        "n_bets": bb.n_bets,
    }


def _resolve_prior_payload(rho_profile: str | None) -> dict[str, Any] | None:
    """Freeze the production Dixon-Coles rho for the soccer bivariate-DC backend.

    Mirrors ``replay_history._resolve_frozen_prior_payload`` so the per-profile CLV
    replay uses the same priors as the main replay. None for non-soccer / fast_score.
    """
    if not rho_profile:
        return None
    from omega.trace.priors import get_production_dc_profile

    store = TraceStore(db_path=str(default_trace_db_path()))
    try:
        prof = get_production_dc_profile(store, rho_profile)
    finally:
        store.close()
    if prof is None:
        return None
    return {"rho": prof.rho, "rho_profile_id": prof.profile_id, "rho_as_of_date": prof.as_of_date}


def run_lab_from_store(
    store: TraceStore,
    *,
    runner: LabCommandRunner,
    out_dir: Path,
    lab_run_id: str,
    league: str,
    plane: str,
    dataset_manifest_id: str,
    replay_id: str,
    train_window: Window,
    validation_window: Window,
    holdout_window: Window,
    replay_db_path: str,
    production_db_path: str | None = None,
    replay_config_hash: str = "",
    dataset_hash: str = "",
    registry: CalibrationRegistry | None = None,
    live_store: TraceStore | None = None,
    auto_promote: bool = False,
    sport_family: str | None = None,
    methods: tuple[str, ...] = grid_mod.DEFAULT_METHODS,
    slices: tuple[str, ...] = (),
    selections: list | None = None,
    replay_records: list | None = None,
    dataset: Any | None = None,
    replay_config: Any | None = None,
    provenance_info: dict[str, Any] | None = None,
) -> LabRunResult:
    """Grid → seal → walk-forward → parity → evidence → artifacts (no replay step).

    ``selections``/``replay_records`` are the replay's staked candidate selections
    and per-event audit records; threading them into the walk-forward is what makes
    betting ROI/CLV real (the single ``betting_metrics`` authority grades them). For
    odds-less datasets they are empty → CLV is honestly INCONCLUSIVE.
    """
    registry = registry or CalibrationRegistry()
    prov = provenance_info or provenance_mod.capture()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    graded = store.query_traces(
        league=league,
        execution_mode="historical_replay",
        has_outcome=True,
        calibration_eligible_only=True,
        limit=1_000_000,
    )
    runner.record("load_graded", "ok", outputs={"n": len(graded)})

    ledger = grid_mod.run_grid(
        graded,
        lab_run_id=lab_run_id,
        league=league,
        plane=plane,
        train_window=train_window,
        validation_window=validation_window,
        holdout_window=holdout_window,
        methods=methods,
        slices=slices,
        sport_family=sport_family,
    )
    runner.record("grid", "ok", outputs={"n_variants": len(ledger.variants)})

    seal = seal_winner(
        ledger,
        graded,
        league=league,
        plane=plane,
        train_window=train_window,
        validation_window=validation_window,
        holdout_window=holdout_window,
        sport_family=sport_family,
    )
    ledger = seal.ledger
    runner.record(
        "seal",
        "ok" if seal.holdout_sealed else "warn",
        outputs={
            "winner": seal.winner.variant_id if seal.winner else None,
            "holdout_access_count": seal.holdout_access_count,
            "winners_curse_risk": seal.winners_curse.risk,
        },
    )

    # Walk-forward (betting/CLV diagnostics + the reused METRICS artifact).
    wf_report = None
    if plane in _WF_PLANES:
        cfg = WalkForwardConfig(markets=[plane])
        wf_report = run_walk_forward(
            store,
            config=cfg,
            league=league,
            replay_id=replay_id,
            dataset_manifest_id=dataset_manifest_id,
            selections=selections or [],
            replay_records=replay_records or [],
        )
        paths["backtest_report"] = _write_json(
            out_dir / "backtest_report.json", wf_report.model_dump(mode="json")
        )
        runner.record("walk_forward", "ok", outputs={"folds": len(wf_report.folds)})
    else:
        runner.record("walk_forward", "skipped", notes=f"plane={plane} unsupported by walk_forward")

    # Backtest parity (candidate vs registry gating incumbent).
    bt_artifact: dict[str, Any] | None = None
    incumbent_id: str | None = None
    incumbent_profile = None
    if seal.winner_profile is not None:
        incumbent_profile = registry.gating_incumbent(seal.winner_profile)
        incumbent_id = incumbent_profile.profile_id if incumbent_profile else None
        bt_artifact = evaluate_backtest_parity(graded, seal.winner_profile, incumbent_profile, plane=plane)
        paths["backtest_parity"] = _write_json(out_dir / "backtest_parity.json", bt_artifact)
    runner.record(
        "backtest_parity",
        "ok" if bt_artifact else "skipped",
        outputs={"recommend": bool(bt_artifact and bt_artifact.get("recommend_promotion"))},
    )

    # CLV: a true candidate-vs-incumbent delta via per-profile holdout replay when
    # the dataset carries decision odds; otherwise the walk-forward betting block
    # (INCONCLUSIVE without graded bets). The 2× per-profile replay is gated on
    # ``dataset.odds`` so odds-less datasets never pay its cost.
    if (
        dataset is not None
        and replay_config is not None
        and getattr(dataset, "odds", None)
        and seal.winner_profile is not None
    ):
        clv_artifact = compare_profiles(
            dataset,
            league=league,
            base_config=replay_config,
            candidate=seal.winner_profile,
            incumbent=incumbent_profile,
            holdout_window=holdout_window,
            work_dir=out_dir / "clv_replay",
        )
        runner.record(
            "clv_per_profile_replay",
            "ok",
            outputs={"verdict": clv_artifact["verdict"], "n_bets": clv_artifact.get("n_bets")},
        )
    else:
        clv_artifact = _clv_artifact(wf_report)
        runner.record("clv_walk_forward", "ok", outputs={"verdict": clv_artifact["verdict"]})
    paths["clv_walk_forward"] = _write_json(out_dir / "clv_walk_forward.json", clv_artifact)

    # Historical-live parity (covariate-shift gate).
    live_traces: list[dict] = []
    if live_store is not None:
        live_traces = [
            t
            for t in live_store.get_graded_traces(league=league, limit=1_000_000)
            if t.get("execution_mode") != "historical_replay"
        ]
    live_artifact = evaluate_parity(graded, live_traces)
    paths["historical_live_parity"] = _write_json(
        out_dir / "historical_live_parity.json", live_artifact
    )
    runner.record("historical_live_parity", "ok", outputs={"state": live_artifact["state"]})

    # Registry audit (reuse list_profiles; no new auditor).
    audit = [p.model_dump(mode="json") for p in registry.list_profiles(league=league)]
    paths["registry_audit"] = _write_json(out_dir / "registry_audit.json", audit)

    # Evidence + optional auto-promote through the single gate.
    ctx = EvidenceContext(
        lab_run_id=lab_run_id,
        league=league,
        plane=plane,
        market=grid_mod.calibration_market_for_plane(plane),
        winner_profile=seal.winner_profile,
        winners_curse=seal.winners_curse,
        holdout_sealed=seal.holdout_sealed,
        attempted_variant_count=len(ledger.variants),
        working_tree_dirty=bool(prov.get("working_tree_dirty")),
        armed=auto_promote,
        incumbent_id=incumbent_id,
        backtest_parity=bt_artifact,
        backtest_parity_path=paths.get("backtest_parity"),
        clv_walk_forward=clv_artifact,
        clv_walk_forward_path=paths.get("clv_walk_forward"),
        live_parity=live_artifact,
        registry_audit_path=paths.get("registry_audit"),
    )
    bundle = resolve(registry, ctx)
    runner.record(
        "promotion",
        "ok",
        outputs={"decision": bundle.decision, "candidate_id": bundle.candidate_id},
    )
    paths["promotion_evidence"] = _write_json(
        out_dir / "PROMOTION_EVIDENCE.json", bundle.model_dump(mode="json")
    )
    paths["attempted_variants"] = _write_json(
        out_dir / "ATTEMPTED_VARIANTS.json", ledger.model_dump(mode="json")
    )

    # Precompute the remaining artifact paths so LAB_RUN.json lists every file.
    paths["lab_run"] = str(out_dir / "LAB_RUN.json")
    paths["report"] = str(out_dir / "REPORT.md")

    lab_run = HistoricalLabRun(
        lab_run_id=lab_run_id,
        code_version=str(prov.get("code_version", "")),
        git_commit=str(prov.get("git_commit", "unknown")),
        working_tree_dirty=bool(prov.get("working_tree_dirty")),
        dataset_manifest_id=dataset_manifest_id,
        dataset_hash=dataset_hash,
        league=league,
        plane=plane,
        replay_id=replay_id,
        replay_db_path=replay_db_path,
        production_db_path=production_db_path,
        replay_config_hash=replay_config_hash,
        profile_grid_hash=ledger.profile_grid_hash,
        attempted_variant_count=len(ledger.variants),
        train_window=train_window,
        validation_window=validation_window,
        holdout_window=holdout_window,
        holdout_sealed=seal.holdout_sealed,
        holdout_access_count=ledger.holdout_access_count,
        auto_promote_armed=auto_promote,
        promotion_candidate_id=bundle.candidate_id,
        promotion_status=bundle.decision,
        result_paths=paths,
    )
    assert_consistent(lab_run, ledger)
    _write_json(out_dir / "LAB_RUN.json", lab_run.model_dump(mode="json"))
    (out_dir / "REPORT.md").write_text(render(lab_run, ledger, bundle, wf_report), encoding="utf-8")

    return LabRunResult(lab_run=lab_run, ledger=ledger, evidence=bundle, paths=paths)


def run_lab(
    *,
    league: str,
    manifest_id: str,
    plane: str,
    replay_db_path: str,
    train_window: Window,
    validation_window: Window,
    holdout_window: Window,
    production_db_path: str | None = None,
    lab_run_id: str | None = None,
    replay_id: str | None = None,
    auto_promote: bool = False,
    methods: tuple[str, ...] = grid_mod.DEFAULT_METHODS,
    slices: tuple[str, ...] = (),
    sport_family: str | None = None,
    rho_profile: str | None = None,
    root: str | Path | None = None,
    registry: CalibrationRegistry | None = None,
) -> LabRunResult:
    """Full run: replay into the isolated DB, then orchestrate everything else."""
    lab_run_id = lab_run_id or f"lab_{manifest_id}_{plane}"
    replay_id = replay_id or f"labreplay_{manifest_id}_{plane}"
    out_dir = lab_dir(root, lab_run_id)
    runner = LabCommandRunner(out_dir / "command_log.jsonl")

    replay_argv = [
        "--league", league,
        "--manifest-id", manifest_id,
        "--db", replay_db_path,
        "--replay-id", replay_id,
        # Size historical bets so the walk-forward can grade ROI/CLV. Harmless on
        # odds-less datasets (no decision odds → no selections → CLV INCONCLUSIVE).
        "--enable-staking",
    ]
    if rho_profile:
        replay_argv += ["--rho-profile", rho_profile]
    if root is not None:
        replay_argv += ["--root", str(root)]
    runner.run_cli("replay", replay_history_main, replay_argv)

    manifest = load_dataset_manifest(manifest_id, root=root)
    dataset_hash = stable_hash(manifest.file_hash_index())
    replay_manifest = load_replay_manifest(replay_id, root=root)
    selections = load_selections(replay_id, root=root)
    replay_records = replay_manifest.records

    # Materialize the dataset + a base ReplayConfig for the per-profile CLV replay
    # (used only when the dataset carries odds; building it is cheap regardless).
    ds_parts = load_normalized_dataset(manifest_id, root=root)
    dataset = ReplayDataset(
        events=ds_parts["events"],
        outcomes=ds_parts["outcomes"],
        odds=ds_parts["odds"],
        extra_context=ds_parts["extra_context"],
        history_override=ds_parts["history_override"],
        prop_markets=ds_parts.get("prop_markets", {}),
        prop_context=ds_parts.get("prop_context", {}),
    )
    base_replay_config = ReplayConfig(
        dataset_manifest_id=manifest_id,
        backtest_db_path=replay_db_path,
        enable_staking=True,
        prior_payload=_resolve_prior_payload(rho_profile),
    )

    store = TraceStore(db_path=replay_db_path)
    live_store = (
        TraceStore(db_path=production_db_path or str(default_trace_db_path()))
        if production_db_path is not None
        else None
    )
    try:
        return run_lab_from_store(
            store,
            runner=runner,
            out_dir=out_dir,
            lab_run_id=lab_run_id,
            league=league,
            plane=plane,
            dataset_manifest_id=manifest_id,
            replay_id=replay_id,
            train_window=train_window,
            validation_window=validation_window,
            holdout_window=holdout_window,
            replay_db_path=replay_db_path,
            production_db_path=production_db_path,
            replay_config_hash=replay_manifest.config_hash,
            dataset_hash=dataset_hash,
            registry=registry,
            live_store=live_store,
            auto_promote=auto_promote,
            sport_family=sport_family,
            methods=methods,
            slices=slices,
            selections=selections,
            replay_records=replay_records,
            dataset=dataset,
            replay_config=base_replay_config,
        )
    finally:
        store.close()
        if live_store is not None:
            live_store.close()
