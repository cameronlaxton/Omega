"""Deterministic local proof for the Phase 6h trace lifecycle.

The harness exercises the operational lifecycle without live providers:
preflight, fixture odds/context, canonical analyze(), export validation, ingest,
outcome attachment, report rendering, session audit rendering, and a tiny
historical replay into an isolated backtest DB.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze
from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    OddsObservation,
    ReplayConfig,
)
from omega.historical.identity import event_key
from omega.historical.normalize import parse_datetime_utc
from omega.historical.replay import ReplayDataset, ReplayEngine
from omega.trace.audit_renderer import render_session_audit
from omega.trace.export_validator import validate_export_block
from omega.trace.session_sidecar import (
    append_audit_events,
    bootstrap_payload,
    create_sidecar,
)
from omega.trace.store import TraceStore

logger = logging.getLogger("omega.ops.prove_lifecycle")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SESSION_ID = "sess-lifecycle-proof"
_REPLAY_SESSION_ID = "sess-lifecycle-replay"
_SEED = 20260616


@dataclass
class StageResult:
    stage: str
    status: str
    detail: str
    artifact: str | None = None


@dataclass
class LifecycleResult:
    ok: bool
    work_dir: str
    stages: list[StageResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "work_dir": self.work_dir,
            "stages": [asdict(stage) for stage in self.stages],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _event_record(step: str, status: str, notes: str, trace_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "ts": _utc_now(),
        "event_type": "step",
        "step": step,
        "status": status,
        "notes": notes,
        "trace_ids": trace_ids or [],
    }


def _fixture_request() -> GameAnalysisRequest:
    return GameAnalysisRequest(
        home_team="Lifecycle Home",
        away_team="Lifecycle Away",
        league="NBA",
        odds={
            "moneyline_home": -120,
            "moneyline_away": 105,
            "spread_home": -2.5,
            "spread_home_price": -110,
            "spread_away_price": -110,
            "over_under": 221.5,
            "total_over_price": -110,
            "total_under_price": -110,
        },
        n_iterations=500,
        simulation_backend="fast_score",
        home_context={"off_rating": 115.0, "def_rating": 110.0, "pace": 99.0},
        away_context={"off_rating": 111.0, "def_rating": 113.0, "pace": 97.0},
        game_context={
            "is_playoff": False,
            "rest_days": 2,
            "source": "fixture:lifecycle",
        },
        seed=_SEED,
        evidence=[],
    )


def _fixture_export_block(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace": trace,
        "bet_record": None,
        "reasoning_inputs": {
            "sources": ["fixture:lifecycle"],
            "fields_gathered": ["odds", "home_context", "away_context", "game_context"],
            "missing_fields": [],
        },
        "reasoning_narrative": "Fixture-backed lifecycle proof; no live provider calls.",
        "downgrade_rationale": [],
    }


def _run_subprocess(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _historical_event(date: str, home: str, away: str) -> HistoricalEvent:
    league = "NFL"
    start = parse_datetime_utc(date)
    return HistoricalEvent(
        event_id=event_key(league, start, home, away),
        league=league,
        sport_family="american_football",
        start_time=start,
        home_team=home,
        away_team=away,
        source_name="fixture:lifecycle",
    )


def _historical_outcome(event: HistoricalEvent, home_score: int, away_score: int) -> HistoricalOutcome:
    return HistoricalOutcome(
        event_id=event.event_id,
        home_score=home_score,
        away_score=away_score,
        result=HistoricalOutcome.derive_result(home_score, away_score),
    )


def _replay_dataset() -> ReplayDataset:
    e1 = _historical_event("2023-09-10", "Team A", "Team B")
    e2 = _historical_event("2023-09-17", "Team C", "Team A")
    e3 = _historical_event("2023-09-24", "Team B", "Team C")
    target = _historical_event("2023-10-01", "Team A", "Team C")
    events = [e1, e2, e3, target]
    outcomes = {
        e1.event_id: _historical_outcome(e1, 24, 17),
        e2.event_id: _historical_outcome(e2, 20, 27),
        e3.event_id: _historical_outcome(e3, 30, 21),
        target.event_id: _historical_outcome(target, 28, 24),
    }

    observations: list[OddsObservation] = []

    def add_moneyline(event: HistoricalEvent, home_price: float, away_price: float) -> None:
        observations.append(
            OddsObservation(
                event_key=event.event_id,
                market="moneyline",
                selection_descriptor="home",
                odds=home_price,
            )
        )
        observations.append(
            OddsObservation(
                event_key=event.event_id,
                market="moneyline",
                selection_descriptor="away",
                odds=away_price,
            )
        )

    for event in (e1, e2, e3):
        add_moneyline(event, -110, -110)
    add_moneyline(target, 200, 200)
    observations.append(
        OddsObservation(
            event_key=target.event_id,
            market="moneyline",
            selection_descriptor="home",
            odds=-180,
            tier_hint="closing",
        )
    )
    return ReplayDataset(
        events=events,
        outcomes=outcomes,
        odds=ReplayDataset.group_odds(observations),
    )


def _prove(work_dir: Path, *, skip_preflight: bool) -> LifecycleResult:
    stages: list[StageResult] = []
    var_dir = work_dir / "var"
    trace_inbox = var_dir / "inbox" / "traces"
    session_dir = var_dir / "inbox" / "sessions"
    report_dir = var_dir / "reports"
    audit_dir = report_dir / "run_audits"
    db_path = var_dir / "omega_traces.db"
    replay_db_path = var_dir / "backtest_lifecycle.db"
    for path in (trace_inbox, session_dir, report_dir, audit_dir):
        path.mkdir(parents=True, exist_ok=True)

    sidecar_path = session_dir / f"{_SESSION_ID}.json"
    create_sidecar(
        sidecar_path,
        {
            **bootstrap_payload(
                _SESSION_ID,
                model_version="omega-core-phase6h",
                purpose="Phase 6h deterministic lifecycle proof",
                bankroll=1000.0,
                bankroll_confirmed=True,
            ),
            "league": "NBA",
            "window": "fixture:lifecycle",
            "effective_db_path": str(db_path),
            "runtime_db_status": "isolated_fixture",
            "pipeline_status": {"lifecycle_proof": "running"},
            "next_required_action": "none",
        },
    )

    if skip_preflight:
        stages.append(StageResult("preflight", "skipped", "--skip-preflight supplied"))
        append_audit_events(
            sidecar_path,
            [_event_record("preflight", "skipped", "--skip-preflight supplied")],
        )
    else:
        from omega.ops.cowork_preflight import run_formal_output_gate

        failures = run_formal_output_gate(require_mcp=False, repo_root=_REPO_ROOT)
        if failures:
            detail = "; ".join(failures)
            stages.append(StageResult("preflight", "fail", detail))
            append_audit_events(sidecar_path, [_event_record("preflight", "fail", detail)])
            return LifecycleResult(False, str(work_dir), stages)
        stages.append(StageResult("preflight", "pass", "formal output gate passed"))
        append_audit_events(
            sidecar_path,
            [_event_record("preflight", "ok", "formal output gate passed")],
        )

    request = _fixture_request()
    stages.append(StageResult("odds_context", "pass", "committed fixture odds/context prepared"))
    append_audit_events(
        sidecar_path,
        [_event_record("odds_context", "ok", "fixture odds/context prepared")],
    )

    trace = analyze(request, session_id=_SESSION_ID, bankroll=1000.0)
    trace_id = str(trace["trace_id"])
    if (trace.get("result") or {}).get("status") != "success":
        detail = f"analyze returned status={(trace.get('result') or {}).get('status')!r}"
        stages.append(StageResult("analyze", "fail", detail))
        append_audit_events(sidecar_path, [_event_record("analyze", "fail", detail, [trace_id])])
        return LifecycleResult(False, str(work_dir), stages)
    stages.append(StageResult("analyze", "pass", f"canonical analyze produced {trace_id}"))
    append_audit_events(
        sidecar_path,
        [_event_record("analyze", "ok", "canonical analyze completed", [trace_id])],
    )

    block = _fixture_export_block(trace)
    validation = validate_export_block(block, strict=True)
    if not validation.ok:
        detail = validation.summary()
        stages.append(StageResult("export_validate", "fail", detail))
        append_audit_events(
            sidecar_path,
            [_event_record("export_validate", "fail", detail, [trace_id])],
        )
        return LifecycleResult(False, str(work_dir), stages)

    export_path = trace_inbox / f"{trace_id}.json"
    export_path.write_text(json.dumps(block, indent=2), encoding="utf-8")
    stages.append(StageResult("export_validate", "pass", validation.summary(), str(export_path)))
    append_audit_events(
        sidecar_path,
        [_event_record("export_validate", "ok", validation.summary(), [trace_id])],
    )

    ingest = _run_subprocess(
        [
            sys.executable,
            "-m",
            "omega.ops.ingest_traces",
            "--inbox",
            str(trace_inbox),
            "--db",
            str(db_path),
            "--sidecar-dir",
            str(session_dir),
            "--strict",
        ],
        cwd=_REPO_ROOT,
    )
    if ingest.returncode != 0:
        detail = (ingest.stderr or ingest.stdout or "ingest failed").strip()
        stages.append(StageResult("ingest", "fail", detail))
        append_audit_events(sidecar_path, [_event_record("ingest", "fail", detail, [trace_id])])
        return LifecycleResult(False, str(work_dir), stages)
    stages.append(StageResult("ingest", "pass", "trace export ingested", str(db_path)))
    append_audit_events(sidecar_path, [_event_record("ingest", "ok", "trace export ingested", [trace_id])])

    store = TraceStore(db_path=str(db_path))
    try:
        store.attach_outcome(trace_id, home_score=110, away_score=104, source="fixture:lifecycle")
    finally:
        store.close()
    stages.append(StageResult("outcomes", "pass", "fixture outcome attached", str(db_path)))
    append_audit_events(
        sidecar_path,
        [_event_record("outcomes", "ok", "fixture outcome attached", [trace_id])],
    )

    report_path = report_dir / "lifecycle_report.md"
    report = _run_subprocess(
        [
            sys.executable,
            "-m",
            "omega.ops.report_calibration",
            "--db",
            str(db_path),
            "--out",
            str(report_path),
            "--sessions-inbox",
            str(session_dir),
            "--window-days",
            "3650",
        ],
        cwd=_REPO_ROOT,
    )
    if report.returncode != 0 or not report_path.exists():
        detail = (report.stderr or report.stdout or "report failed").strip()
        stages.append(StageResult("report", "fail", detail))
        append_audit_events(sidecar_path, [_event_record("report", "fail", detail, [trace_id])])
        return LifecycleResult(False, str(work_dir), stages)
    stages.append(StageResult("report", "pass", "calibration report rendered", str(report_path)))
    append_audit_events(
        sidecar_path,
        [_event_record("report", "ok", "calibration report rendered", [trace_id])],
    )

    audit_path = render_session_audit(
        _SESSION_ID,
        db_path=db_path,
        sidecar_dir=session_dir,
        out_dir=audit_dir,
    )
    stages.append(StageResult("audit", "pass", "session audit rendered", str(audit_path)))

    replay_store = TraceStore(db_path=str(replay_db_path))
    try:
        replay_config = ReplayConfig(
            dataset_manifest_id="lifecycle-fixture",
            backtest_db_path=str(replay_db_path),
            session_id=_REPLAY_SESSION_ID,
            enable_staking=False,
            n_iterations=200,
        )
        replay_result = ReplayEngine(replay_store, replay_config).run(
            _replay_dataset(),
            replay_id="lifecycle-replay",
            league="NFL",
        )
    finally:
        replay_store.close()
    if replay_result.n_persisted <= 0:
        detail = f"replay persisted {replay_result.n_persisted} traces"
        stages.append(StageResult("replay", "fail", detail, str(replay_db_path)))
        return LifecycleResult(False, str(work_dir), stages)
    stages.append(
        StageResult(
            "replay",
            "pass",
            f"isolated historical replay persisted {replay_result.n_persisted} traces",
            str(replay_db_path),
        )
    )

    return LifecycleResult(True, str(work_dir), stages)


def run(
    *,
    work_dir: str | Path | None = None,
    keep_artifacts: bool = False,
    skip_preflight: bool = False,
) -> LifecycleResult:
    temp_root: Path | None = None
    if work_dir is None:
        temp_root = Path(tempfile.mkdtemp(prefix="omega-lifecycle-"))
        root = temp_root
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)

    result: LifecycleResult | None = None
    try:
        result = _prove(root, skip_preflight=skip_preflight)
        return result
    finally:
        if temp_root is not None and not keep_artifacts and result is not None and result.ok:
            shutil.rmtree(temp_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove the Omega lifecycle with local fixtures.")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.json_output else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run(
        work_dir=args.work_dir,
        keep_artifacts=args.keep_artifacts,
        skip_preflight=args.skip_preflight,
    )
    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        status = "PASS" if result.ok else "FAIL"
        print(f"Lifecycle proof: {status} ({result.work_dir})")
        for stage in result.stages:
            artifact = f" [{stage.artifact}]" if stage.artifact else ""
            print(f"- {stage.stage}: {stage.status} - {stage.detail}{artifact}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
