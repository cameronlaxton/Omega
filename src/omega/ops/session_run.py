"""omega-session-run: hardened daily multi-sport session orchestrator.

Owns the non-engine phases of a daily Omega session end-to-end so the operator
does not need to manually discover commands, sidecar rules, or sport-key
mappings. The engine phase itself stays where it belongs (omega_run_batch via
MCP, or the AGENTS.md batch-rule fallback) — this wrapper is the spine around
it, not a second engine.

Workflow
--------
Phase 1  open/reopen the session sidecar (create_sidecar + bootstrap_payload;
         fail-closed on collision/closed/corrupt/mismatched reopen). Records
         effective_db_path + runtime_db_status from TraceStore resolution.
Phase 2  formal preflight gate (omega-cowork-preflight --formal-output-gate).
Phase 3  soccer prior-coverage gate (if any --leagues value is a soccer league).
Phase 4  list active tennis events (if TENNIS/ATP/WTA in --leagues).
Phase 5  print a structured ANALYSIS PLAN — the exact MCP or CLI commands to run
         for the engine phase. >3 analyses => omega_run_batch (MCP preferred).
Phase 6  if --ingest: validate the trace inbox (omega-validate-trace-export
         --strict), then omega-ingest-traces --strict with the session sidecar
         directory threaded through. Validation failure blocks ingest.
Phase 7  if --render-report: run omega-render-session-report --kind intake.
Phase 8  if --close: render session audit markdown, validate all sidecars, then
         close the sidecar (closed_at, exec_stats, pipeline_status,
         next_required_action). A session with hard failures is left open so it
         can be retried with --reopen.

Every phase appends sidecar audit events (except in --dry-run). Engine-owned
numeric outputs never go into audit event inputs/outputs — append_audit_events
rejects them (ProtectedValueError).

Modes
-----
--dry-run   print the exact commands and sidecar intent for every phase without
            mutating anything (no sidecar create/append, no subprocesses).
--proof     run the deterministic fixture lifecycle proof (prove_lifecycle.py)
            in an isolated temp dir — demonstrates the full open→analyze→
            export→ingest→report→audit chain without touching production
            DB/session state.

The command exits 0 when all completed phases succeeded, 1 on hard failure, and
2 when coverage gates downgraded the recommended output mode (non-fatal; the
plan is still emitted so the operator can run the session in research mode).

Example
-------
    # Morning: open the session, run gates, print the engine plan.
    omega-session-run --leagues MLB,TENNIS --mlb-games 9 --tennis-games 10

    # After the engine phase (omega_run_batch) has exported traces:
    omega-session-run --session-id sess-20260702-143502a1b2 --reopen \\
        --leagues MLB,TENNIS --ingest --render-report --close
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.paths import session_inbox_dir, trace_inbox_dir  # noqa: E402
from omega.trace.session_sidecar import (  # noqa: E402
    append_audit_events,
    bootstrap_payload,
    close_sidecar,
    create_sidecar,
)

logger = logging.getLogger("omega.ops.session_run")
UTC = timezone.utc

# League codes classified by sport family
_SOCCER_LEAGUES = frozenset(
    {
        "EPL",
        "MLS",
        "LA_LIGA",
        "BUNDESLIGA",
        "SERIE_A",
        "LIGUE_1",
        "CHAMPIONS_LEAGUE",
        "WORLD_CUP",
        "FIFA_WORLD_CUP_2026",
        "FIFA_INTL",
        "FIFA_FRIENDLY",
        "FIFA_NATIONS_LEAGUE",
        "FIFA_QUALIFIERS",
    }
)
_TENNIS_LEAGUES = frozenset({"ATP", "WTA", "GRAND_SLAM", "TENNIS"})
_PROP_CAPABLE_LEAGUES = frozenset({"MLB", "NBA", "NFL", "NHL", "WNBA"})


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _generate_session_id(date: str) -> str:
    # Canonical sidecar id format: sess-YYYYMMDD-HHMMSSXXXX (compact date,
    # UTC seconds, 4 random hex chars) — matches omega-session-bootstrap's
    # convention. The random suffix is belt-and-suspenders only: the real
    # collision guard is create_sidecar()'s fail-closed existence check, which
    # catches reused *manually-supplied* IDs (the sess-20260701-ops1 case) too.
    compact = date.replace("-", "")
    ts = datetime.now(UTC).strftime("%H%M%S")
    return f"sess-{compact}-{ts}{uuid.uuid4().hex[:4]}"


def _league_list(raw: str) -> list[str]:
    return [lg.strip().upper() for lg in raw.split(",") if lg.strip()]


def _soccer_leagues_in(leagues: list[str]) -> list[str]:
    return [lg for lg in leagues if lg in _SOCCER_LEAGUES]


def _tennis_leagues_in(leagues: list[str]) -> list[str]:
    return [lg for lg in leagues if lg in _TENNIS_LEAGUES]


def _module_cmd(module: str, *args: str) -> list[str]:
    # Every phase command is `python -m omega.ops.<module>` so the printed
    # dry-run line is exactly reproducible in any environment where the package
    # is importable (no PATH/.exe entrypoint dependence).
    return [sys.executable, "-m", module, *args]


def _run_subprocess(cmd: list[str], *, label: str, dry_run: bool) -> int:
    """Run a phase command and return its exit code. Prints output live."""
    if dry_run:
        print(f"  [DRY-RUN] would run ({label}): {' '.join(cmd)}")
        return 0
    print(f"  [{label}] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


class _SessionContext:
    """Per-invocation execution accounting threaded through the phases.

    Holds only execution metadata (exit codes, failure names) — never
    engine-owned numeric outputs, which must stay in var/omega_traces.db.
    """

    def __init__(self, session_id: str, sidecar_path: Path, *, dry_run: bool) -> None:
        self.session_id = session_id
        self.sidecar_path = sidecar_path
        self.dry_run = dry_run
        self.sidecar_open = False
        self.phase_rcs: dict[str, int] = {}
        self.failures: list[str] = []

    def append_event(
        self,
        *,
        event_type: str,
        step: str,
        status: str,
        notes: str,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        if self.dry_run or not self.sidecar_open:
            return
        append_audit_events(
            self.sidecar_path,
            [
                {
                    "ts": _utc_now(),
                    "event_type": event_type,
                    "step": step,
                    "status": status,
                    "notes": notes,
                    "inputs": inputs,
                    "trace_ids": [],
                }
            ],
        )

    def run_command_phase(
        self,
        phase: str,
        cmd: list[str],
        *,
        event_type: str = "command",
        ok_codes: tuple[int, ...] = (0,),
    ) -> int:
        rc = _run_subprocess(cmd, label=phase, dry_run=self.dry_run)
        self.phase_rcs[phase] = rc
        status = "ok" if rc in ok_codes else "fail"
        self.append_event(
            event_type=event_type,
            step=phase,
            status=status,
            notes=f"exit={rc}: {' '.join(cmd)}",
        )
        if status == "fail":
            self.failures.append(phase)
        return rc


def _resolve_trace_store(db: str | None) -> tuple[str | None, str | None]:
    """Best-effort TraceStore path resolution for the sidecar record.

    Read-only so resolution can never create a DB, run migrations, or trigger
    the redirect guard. Never blocks opening a session.
    """
    try:
        from omega.trace.store import TraceStore

        store = TraceStore(db_path=db, read_only=True)
        try:
            return store.db_path, store.db_path_source
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001 — resolution must never block the session
        logger.warning("TraceStore resolution failed (%s); recording unknown", exc)
        return None, "unknown"


def _phase_open_sidecar(
    ctx: _SessionContext,
    *,
    date: str,
    leagues: list[str],
    reopen: bool,
    bankroll: float,
    bankroll_confirmed: bool,
    model_version: str,
    db: str | None,
) -> int:
    """Phase 1: create or reopen the session sidecar. Fail-closed via create_sidecar."""
    print(f"\n[Phase 1] Session sidecar: {ctx.sidecar_path}")
    effective_db_path, runtime_db_status = _resolve_trace_store(db)
    if ctx.dry_run:
        intent = "reopen (same open session only)" if reopen else "create"
        print(f"  [DRY-RUN] would {intent} sidecar via create_sidecar + bootstrap_payload")
        print(
            f"  [DRY-RUN] sidecar intent: session_id={ctx.session_id} "
            f"model_version={model_version} bankroll={bankroll} "
            f"bankroll_confirmed={bankroll_confirmed} league={','.join(leagues)} "
            f"window={date}"
        )
        print(
            f"  [DRY-RUN] would record effective_db_path={effective_db_path} "
            f"runtime_db_status={runtime_db_status}"
        )
        return 0

    if reopen and not ctx.sidecar_path.exists():
        print(
            f"[FAIL] --reopen was given but no sidecar exists at {ctx.sidecar_path}. "
            "Reopen is only for continuing the SAME open session; check the session id."
        )
        return 1

    ctx.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    existed_before = ctx.sidecar_path.exists()
    payload = {
        **bootstrap_payload(
            ctx.session_id,
            model_version=model_version,
            purpose=f"daily operator session ({', '.join(leagues)})",
            bankroll=bankroll,
            bankroll_confirmed=bankroll_confirmed,
            effective_db_path=effective_db_path,
            runtime_db_status=runtime_db_status,
        ),
        "league": ",".join(leagues),
        "window": date,
        "pipeline_status": {"session_run": "open"},
        "next_required_action": (
            "run engine phase (omega_run_batch), then re-invoke omega-session-run "
            "with --reopen --ingest --render-report --close"
        ),
    }
    try:
        create_sidecar(ctx.sidecar_path, payload, allow_reopen=reopen)
    except FileExistsError as exc:
        # Collision, closed session, corrupt sidecar, or mismatched reopen.
        print(f"[FAIL] {exc}")
        return 1
    ctx.sidecar_open = True

    reopened = reopen and existed_before
    step = "session_reopen" if reopened else "session_open"
    ctx.append_event(
        event_type="step",
        step=step,
        status="ok",
        notes=f"sidecar {'reopened' if reopened else 'created'} at {ctx.sidecar_path}",
        inputs={
            "reopen": reopen,
            "effective_db_path": effective_db_path,
            "runtime_db_status": runtime_db_status,
        },
    )
    print(f"[OK] Sidecar {'reopened' if reopened else 'created'} (session={ctx.session_id}).")
    return 0


def _phase_preflight(ctx: _SessionContext, *, skip: bool) -> int:
    """Phase 2: formal preflight gate."""
    print("\n[Phase 2] Formal preflight gate...")
    if skip:
        print("[SKIP] --skip-preflight supplied (not recommended for production sessions).")
        ctx.append_event(
            event_type="preflight",
            step="cowork_preflight",
            status="skipped",
            notes="--skip-preflight supplied",
        )
        return 0
    cmd = _module_cmd("omega.ops.cowork_preflight", "--formal-output-gate")
    rc = _run_subprocess(cmd, label="preflight", dry_run=ctx.dry_run)
    ctx.phase_rcs["preflight"] = rc
    ctx.append_event(
        event_type="preflight",
        step="cowork_preflight",
        status="ok" if rc == 0 else "fail",
        notes=f"exit={rc}: {' '.join(cmd)}",
    )
    if rc != 0 and not ctx.dry_run:
        ctx.failures.append("preflight")
        print(
            f"[FAIL] formal preflight gate exited {rc}. Fix the reported issues before "
            "proceeding — do not emit formal Omega numeric outputs."
        )
        return rc
    print("[OK] Preflight passed." if not ctx.dry_run else "  [DRY-RUN] preflight skipped.")
    return 0


def _phase_soccer_gate(
    ctx: _SessionContext,
    soccer_leagues: list[str],
    *,
    home_team: str | None,
    away_team: str | None,
    db: str | None,
) -> list[str]:
    """Phase 3: soccer prior-coverage gate. Returns downgraded leagues."""
    downgraded: list[str] = []
    if not soccer_leagues:
        return downgraded

    print(f"\n[Phase 3] Checking soccer prior coverage for: {', '.join(soccer_leagues)}")
    for league in soccer_leagues:
        cmd = _module_cmd(
            "omega.ops.soccer_prior_coverage", "--league", league, "--format", "summary"
        )
        if home_team:
            cmd += ["--home-team", home_team]
        if away_team:
            cmd += ["--away-team", away_team]
        if db:
            cmd += ["--db", db]

        rc = _run_subprocess(cmd, label=f"soccer-gate-{league}", dry_run=ctx.dry_run)
        if rc == 2:
            print(
                f"  [DOWNGRADE] {league}: weak/no DC rho profile → output mode forced to "
                f"research_candidate for this league."
            )
            downgraded.append(league)
            ctx.append_event(
                event_type="downgrade",
                step=f"soccer_prior_coverage_{league}",
                status="warn",
                notes=f"{league}: weak/no DC rho profile; output mode → research_candidate",
            )
        elif rc != 0 and not ctx.dry_run:
            print(
                f"  [ERROR] Soccer prior coverage check for {league} failed (exit {rc}). "
                "Session can proceed but FIFA outputs may be unreliable."
            )
            ctx.append_event(
                event_type="quality_gate",
                step=f"soccer_prior_coverage_{league}",
                status="warn",
                notes=f"coverage check failed (exit={rc}); outputs may be unreliable",
            )
        else:
            ctx.append_event(
                event_type="quality_gate",
                step=f"soccer_prior_coverage_{league}",
                status="ok",
                notes=f"prior coverage acceptable (exit={rc})",
            )

    return downgraded


def _phase_tennis_events(ctx: _SessionContext, tennis_leagues: list[str]) -> None:
    """Phase 4: list active tennis events for discovered sport keys."""
    if not tennis_leagues:
        return

    print(f"\n[Phase 4] Listing active tennis events for: {', '.join(tennis_leagues)}")
    for tour in tennis_leagues:
        league = "ATP" if tour == "TENNIS" else tour
        cmd = _module_cmd(
            "omega.ops.resolve_odds", "--list-events", "--league", league, "--format", "summary"
        )
        rc = _run_subprocess(cmd, label=f"tennis-events-{league}", dry_run=ctx.dry_run)
        if rc not in (0, 2) and not ctx.dry_run:
            print(
                f"  [WARN] Tennis event listing for {league} returned {rc}. "
                "No active tournament key may be available right now."
            )
        ctx.append_event(
            event_type="command",
            step=f"tennis_events_{league}",
            status="ok" if rc in (0, 2) else "warn",
            notes=f"exit={rc}: {' '.join(cmd)}",
        )


def _analysis_plan(
    *,
    session_id: str,
    date: str,
    leagues: list[str],
    mlb_games: int,
    mlb_props: int,
    tennis_games: int,
    fifa_games: int,
    mode: str,
    downgraded_leagues: list[str],
    require_actionable_min: int,
) -> str:
    """Render the structured analysis plan the operator needs to execute."""
    lines = [
        "=" * 70,
        "ANALYSIS PLAN",
        "=" * 70,
        f"session_id  : {session_id}",
        f"date        : {date}",
        f"leagues     : {', '.join(leagues)}",
        f"mode        : {mode}",
        "",
    ]

    if downgraded_leagues:
        lines += [
            "!! COVERAGE DOWNGRADES (use research_candidate mode for these):",
        ]
        for lg in downgraded_leagues:
            lines.append(f"   - {lg}: no production rho → research_candidate")
        lines.append("")

    lines += [
        "ENGINE PHASE — more than 3 analyses means omega_run_batch, per the",
        "AGENTS.md batch rule. Do NOT hand-roll an analyze() loop.",
        "",
        "Option A — MCP (preferred when server is running):",
        "  Use omega_run_batch with the following parameters.",
        "  See batch_rule in AGENTS.md for exact contract.",
        "",
        "  omega_run_batch(",
        f'    session_id="{session_id}",',
        "    bankroll=1000.0,",
        "    entries=[",
        "      # Game Entry Example:",
        "      {",
        '        "kind": "game",',
        f'        "league": "{leagues[0] if leagues else "MLB"}",',
        '        "home_team": "Home Team",',
        '        "away_team": "Away Team",',
        f'        "game_date": "{date}",',
        '        "home_context": {"off_rating": 4.5, "def_rating": 3.8},',
        '        "away_context": {"off_rating": 4.1, "def_rating": 4.2},',
        '        "game_context": {"is_playoff": False, "rest_days": 4},',
        '        "evidence": [',
        "          {",
        '            "signal_type": "rest_advantage",',
        '            "category": "situational",',
        '            "plane": "game",',
        '            "value": 2,',
        '            "source": "agent_reasoning",',
        '            "confidence": 0.8,',
        '            "window": "matchup",',
        '            "direction": "home"',
        "          }",
        "        ],",
        '        "reasoning_narrative": "Narrative summary of reasoning here...",',
        '        "reasoning_sources": ["espn.com"]',
        "      },",
        "      # Player Prop Entry Example:",
        "      {",
        '        "kind": "prop",',
        f'        "league": "{leagues[0] if leagues else "MLB"}",',
        '        "home_team": "Home Team",',
        '        "away_team": "Away Team",',
        f'        "game_date": "{date}",',
        '        "player_name": "Player Name",',
        '        "prop_type": "strikeouts_pitched",  # or other stat key',
        '        "player_context": {',
        '          "strikeouts_pitched_mean": 5.8,',
        '          "strikeouts_pitched_std": 2.1,',
        '          "sample_size": 10',
        "        },",
        '        "game_context": {"is_playoff": False, "rest_days": 4},',
        '        "evidence": []',
        "      }",
        "    ]",
    ]

    if "MLB" in leagues:
        lines += [
            f"    # MLB Info: {mlb_games} games + {mlb_props} prop sweeps planned",
            "    # Include: h2h, spreads, totals for each game",
            "    # Props: strikeouts_pitched, hits, total_bases, earned_runs",
        ]

    if any(lg in _TENNIS_LEAGUES for lg in leagues):
        lines += [
            f"    # TENNIS Info: {tennis_games} matches planned",
            "    # Requires: surface context, serve_win_pct, return_win_pct",
        ]

    if any(lg in _SOCCER_LEAGUES for lg in leagues):
        effective_mode = "research_candidate" if downgraded_leagues else mode
        lines += [
            f"    # FIFA/Soccer Info: {fifa_games} matches planned",
            f"    # output_mode={effective_mode!r} (downgraded due to prior coverage)"
            if downgraded_leagues
            else "",
            "    # Requires: rho prior (injected automatically from priors_dixon_coles)",
        ]

    lines += [
        "  )",
        "",
        "Option B — Batch script (ONLY if MCP is unavailable):",
        "  Follow the AGENTS.md batch-rule fallback contract exactly:",
        "  cowork_preflight.run_formal_output_gate() before any analyze() call,",
        "  deterministic sha256 seeds, evidence per trace, no hardcoded quality.",
        "",
        "After engine phase completes, export traces to var/inbox/traces/ and run:",
        "",
        "  omega-validate-trace-export var/inbox/traces  # strict pre-ingest gate",
        "  omega-ingest-traces                          # ingest exported traces",
        "  omega-render-session-report \\",
        "    --kind intake \\",
        f"    --session-id {session_id}                  # render audit report",
        "",
        "Or re-invoke session-run with the same --session-id plus",
        "  --reopen --ingest --render-report --close    # full closeout",
        "=" * 70,
    ]

    # Remove blank strings that sneak in from conditional appends
    return "\n".join(line for line in lines if line is not None)


def _phase_validate_and_ingest(
    ctx: _SessionContext,
    *,
    trace_inbox: Path,
    sidecar_dir: Path,
    db: str | None,
    verbose: bool,
) -> None:
    """Phase 6: strict export validation gate, then ingest. Validation failure
    blocks ingest — a wrong wrapper is fixed by re-wrapping, never by re-running
    analyze()."""
    print(f"\n[Phase 6] Validating trace exports in {trace_inbox}...")
    validate_cmd = _module_cmd("omega.ops.validate_trace_export", str(trace_inbox), "--strict")
    rc = ctx.run_command_phase("validate-trace-export", validate_cmd, event_type="quality_gate")
    if rc != 0 and not ctx.dry_run:
        print(
            f"[FAIL] export validation exited {rc}; ingest blocked. Re-wrap/re-export the "
            "failing files (do NOT re-run analyze), then re-invoke with --reopen --ingest."
        )
        return

    print(f"\n[Phase 6] Ingesting traces (strict) for session {ctx.session_id}...")
    ingest_cmd = _module_cmd(
        "omega.ops.ingest_traces",
        "--strict",
        "--inbox",
        str(trace_inbox),
        "--sidecar-dir",
        str(sidecar_dir),
    )
    if db:
        ingest_cmd += ["--db", db]
    if verbose:
        ingest_cmd.append("--verbose")
    rc = ctx.run_command_phase("ingest-traces", ingest_cmd)
    if rc != 0 and not ctx.dry_run:
        print(f"[FAIL] omega-ingest-traces exited {rc}.")
    else:
        print("[OK] Traces ingested." if not ctx.dry_run else "  [DRY-RUN]")


def _phase_render_report(
    ctx: _SessionContext,
    *,
    out_dir: Path | None,
    db: str | None,
    verbose: bool,
) -> None:
    """Phase 7: omega-render-session-report."""
    print(f"\n[Phase 7] Rendering session report for {ctx.session_id}...")
    cmd = _module_cmd(
        "omega.ops.render_session_report",
        "--kind",
        "intake",
        "--session-id",
        ctx.session_id,
    )
    if out_dir:
        cmd += ["--out-dir", str(out_dir)]
    if verbose:
        cmd.append("--verbose")
    if db:
        cmd += ["--db", db]
    rc = ctx.run_command_phase("render-report", cmd)
    if rc != 0 and not ctx.dry_run:
        print(f"[FAIL] omega-render-session-report exited {rc}.")
    else:
        print("[OK] Report written." if not ctx.dry_run else "  [DRY-RUN]")


def _phase_closeout(
    ctx: _SessionContext,
    *,
    sidecar_dir: Path,
    db: str | None,
    verbose: bool,
    date: str,
    leagues: list[str],
    downgraded: list[str],
) -> int:
    """Phase 8: audit markdown, sidecar validation, then close the sidecar."""
    print(f"\n[Phase 8] Closeout for session {ctx.session_id}...")

    audit_cmd = _module_cmd(
        "omega.ops.render_session_audits",
        "--session-ids",
        ctx.session_id,
        "--sidecar-dir",
        str(sidecar_dir),
    )
    if db:
        audit_cmd += ["--db", db]
    if verbose:
        audit_cmd.append("--verbose")
    ctx.run_command_phase("render-audits", audit_cmd)

    validate_cmd = _module_cmd(
        "omega.ops.validate_session_sidecars", "--sessions-inbox", str(sidecar_dir)
    )
    if verbose:
        validate_cmd.append("--verbose")
    ctx.run_command_phase("validate-sidecars", validate_cmd, event_type="quality_gate")

    if ctx.dry_run:
        print(
            "  [DRY-RUN] would close sidecar via close_sidecar(): set closed_at, "
            "exec_stats (phase exit codes), pipeline_status, next_required_action='none'"
        )
        return 0

    if ctx.failures:
        print(
            f"[FAIL] not closing sidecar — failed phase(s): {', '.join(ctx.failures)}. "
            "Session left open; fix and re-invoke with --reopen."
        )
        ctx.append_event(
            event_type="step",
            step="session_close",
            status="fail",
            notes=f"close refused; failed phases: {', '.join(ctx.failures)}",
        )
        return 1

    exec_stats = {
        "phases": dict(ctx.phase_rcs),
        "failed_phases": list(ctx.failures),
        "date": date,
        "leagues": leagues,
        "downgraded_leagues": downgraded,
    }
    pipeline_status = {
        "session_run": "closed",
        "phases": {phase: ("ok" if rc == 0 else "fail") for phase, rc in ctx.phase_rcs.items()},
    }
    ctx.append_event(
        event_type="step",
        step="session_close",
        status="ok",
        notes="all requested phases succeeded; closing sidecar",
    )
    try:
        close_sidecar(
            ctx.sidecar_path,
            exec_stats=exec_stats,
            pipeline_status=pipeline_status,
            next_required_action="none",
        )
    except (OSError, ValueError) as exc:
        print(f"[FAIL] could not close sidecar: {exc}")
        return 1
    print(f"[OK] Sidecar closed (session={ctx.session_id}).")
    return 0


def _run_proof(*, skip_preflight: bool, keep_artifacts: bool, dry_run: bool) -> int:
    """--proof: run the deterministic fixture lifecycle in an isolated temp dir."""
    print("\n[PROOF] Fixture lifecycle proof (isolated temp dir; no production mutation)")
    if dry_run:
        print(
            "  [DRY-RUN] would run prove_lifecycle.run(): preflight → fixture analyze → "
            "export validate → ingest → outcome attach → report → session audit → replay, "
            "all against an isolated work dir"
        )
        return 0
    from omega.ops import prove_lifecycle

    result = prove_lifecycle.run(keep_artifacts=keep_artifacts, skip_preflight=skip_preflight)
    status = "PASS" if result.ok else "FAIL"
    print(f"Lifecycle proof: {status} ({result.work_dir})")
    for stage in result.stages:
        artifact = f" [{stage.artifact}]" if stage.artifact else ""
        print(f"- {stage.stage}: {stage.status} - {stage.detail}{artifact}")
    return 0 if result.ok else 1


def run_session(
    *,
    session_id: str | None = None,
    date: str | None = None,
    leagues: list[str] | None = None,
    mlb_games: int = 0,
    mlb_props: int = 0,
    tennis_games: int = 0,
    fifa_games: int = 0,
    mode: str = "research-lean",
    require_actionable_min: int = 0,
    home_team: str | None = None,
    away_team: str | None = None,
    reopen: bool = False,
    ingest: bool = False,
    render_report: bool = False,
    close: bool = False,
    proof: bool = False,
    keep_proof_artifacts: bool = False,
    report_out_dir: Path | None = None,
    sidecar_dir: Path | None = None,
    trace_inbox: Path | None = None,
    bankroll: float = 1000.0,
    bankroll_confirmed: bool = False,
    model_version: str = "omega-session-run",
    dry_run: bool = False,
    skip_preflight: bool = False,
    verbose: bool = False,
    db: str | None = None,
) -> int:
    """Programmatic entry point (also called by main())."""
    if proof:
        return _run_proof(
            skip_preflight=skip_preflight, keep_artifacts=keep_proof_artifacts, dry_run=dry_run
        )

    leagues = leagues or []
    if not leagues:
        print("[FAIL] --leagues is required (unless --proof).")
        return 1
    if reopen and not session_id:
        print(
            "[FAIL] --reopen requires an explicit --session-id: reopen is only for "
            "continuing the SAME open session, never an auto-generated id."
        )
        return 1

    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    session_id = session_id or _generate_session_id(date)
    sidecar_dir = Path(sidecar_dir) if sidecar_dir else session_inbox_dir()
    trace_inbox = Path(trace_inbox) if trace_inbox else trace_inbox_dir()

    print(f"\nomega-session-run  |  session={session_id}  date={date}  leagues={','.join(leagues)}")
    if dry_run:
        print("  [DRY-RUN MODE — no sidecar writes, no subprocess calls]")

    ctx = _SessionContext(session_id, sidecar_dir / f"{session_id}.json", dry_run=dry_run)

    # Phase 1: open/reopen sidecar (fail-closed)
    rc = _phase_open_sidecar(
        ctx,
        date=date,
        leagues=leagues,
        reopen=reopen,
        bankroll=bankroll,
        bankroll_confirmed=bankroll_confirmed,
        model_version=model_version,
        db=db,
    )
    if rc != 0:
        return 1

    # Phase 2: formal preflight gate
    rc = _phase_preflight(ctx, skip=skip_preflight)
    if rc != 0 and not dry_run:
        return 1

    # Phase 3: soccer gate
    downgraded = _phase_soccer_gate(
        ctx,
        _soccer_leagues_in(leagues),
        home_team=home_team,
        away_team=away_team,
        db=db,
    )

    # Phase 4: tennis discovery
    _phase_tennis_events(ctx, _tennis_leagues_in(leagues))

    # Phase 5: print analysis plan (engine handoff — never run the engine here)
    plan = _analysis_plan(
        session_id=session_id,
        date=date,
        leagues=leagues,
        mlb_games=mlb_games,
        mlb_props=mlb_props,
        tennis_games=tennis_games,
        fifa_games=fifa_games,
        mode=mode,
        downgraded_leagues=downgraded,
        require_actionable_min=require_actionable_min,
    )
    print("\n[Phase 5] Analysis Plan\n")
    print(plan)
    ctx.append_event(
        event_type="note",
        step="analysis_plan",
        status="ok",
        notes="analysis plan emitted; engine phase is omega_run_batch (MCP preferred)",
        inputs={
            "mode": mode,
            "downgraded_leagues": downgraded,
            "planned": {
                "mlb_games": mlb_games,
                "mlb_props": mlb_props,
                "tennis_games": tennis_games,
                "fifa_games": fifa_games,
            },
        },
    )

    overall_rc = 2 if downgraded else 0

    # Phase 6: validate exports, then ingest
    if ingest:
        _phase_validate_and_ingest(
            ctx,
            trace_inbox=trace_inbox,
            sidecar_dir=sidecar_dir,
            db=db,
            verbose=verbose,
        )

    # Phase 7: render report
    if render_report:
        _phase_render_report(ctx, out_dir=report_out_dir, db=db, verbose=verbose)

    # Phase 8: closeout (audits + sidecar validation + close)
    if close:
        rc = _phase_closeout(
            ctx,
            sidecar_dir=sidecar_dir,
            db=db,
            verbose=verbose,
            date=date,
            leagues=leagues,
            downgraded=downgraded,
        )
        if rc != 0:
            overall_rc = 1

    if ctx.failures:
        overall_rc = 1

    print(f"\nomega-session-run complete  |  session={session_id}  exit={overall_rc}")
    return overall_rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help=("Unique session identifier (sess-YYYYMMDD-HHMMSSXXXX). Auto-generated if omitted."),
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Analysis date YYYY-MM-DD (default: today UTC).",
    )
    parser.add_argument(
        "--leagues",
        default=None,
        help="Comma-separated Omega league codes (e.g. MLB,TENNIS,FIFA_WORLD_CUP_2026). "
        "Required unless --proof.",
    )
    parser.add_argument(
        "--mlb-games", type=int, default=0, help="Number of MLB game analyses to plan."
    )
    parser.add_argument(
        "--mlb-props", type=int, default=0, help="Number of MLB prop analyses to plan."
    )
    parser.add_argument(
        "--tennis-games", type=int, default=0, help="Number of tennis match analyses to plan."
    )
    parser.add_argument(
        "--fifa-games", type=int, default=0, help="Number of FIFA/soccer game analyses to plan."
    )
    parser.add_argument(
        "--mode",
        default="research-lean",
        choices=["research-lean", "actionable", "research_candidate", "low_confidence_actionable"],
        help="Requested output mode for engine analysis.",
    )
    parser.add_argument(
        "--require-actionable-min",
        type=int,
        default=0,
        metavar="N",
        help="Warn in the plan if fewer than N actionable outputs are expected.",
    )
    parser.add_argument(
        "--home-team",
        default=None,
        help="Home team name for soccer prior-coverage xG lookup.",
    )
    parser.add_argument(
        "--away-team",
        default=None,
        help="Away team name for soccer prior-coverage xG lookup.",
    )
    parser.add_argument(
        "--reopen",
        action="store_true",
        help="Continue the SAME open session (requires --session-id). Fails closed on "
        "closed/corrupt/mismatched sidecars — never reuse another session's id.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Validate the trace inbox (strict), then run omega-ingest-traces --strict. "
        "Use when re-invoking after the engine phase has completed.",
    )
    parser.add_argument(
        "--render-report",
        action="store_true",
        help="After ingest (or plan), run omega-render-session-report --kind intake.",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Closeout: render session audit markdown, validate sidecars, then close the "
        "sidecar (closed_at + exec_stats). Refused if any phase failed.",
    )
    parser.add_argument(
        "--proof",
        action="store_true",
        help="Run the deterministic fixture lifecycle proof (prove_lifecycle) in an "
        "isolated temp dir; never mutates production DB/session state.",
    )
    parser.add_argument(
        "--keep-proof-artifacts",
        action="store_true",
        help="With --proof: keep the temp work dir for inspection.",
    )
    parser.add_argument(
        "--report-out-dir",
        type=Path,
        default=None,
        help="Directory for rendered report files.",
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help="Session sidecar directory (default: var/inbox/sessions).",
    )
    parser.add_argument(
        "--trace-inbox",
        type=Path,
        default=None,
        help="Trace export inbox to validate/ingest (default: var/inbox/traces).",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Session bankroll recorded in the sidecar (default: 1000).",
    )
    parser.add_argument(
        "--bankroll-confirmed",
        action="store_true",
        help="Mark the sidecar bankroll as operator-confirmed.",
    )
    parser.add_argument(
        "--model-version",
        default="omega-session-run",
        help="model_version recorded in the sidecar (e.g. the agent model id).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the formal preflight gate (not recommended for production sessions).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact commands and sidecar intent; mutate nothing.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--db", default=None, help="SQLite path override for DB-dependent phases.")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    leagues = _league_list(args.leagues) if args.leagues else []
    if not leagues and not args.proof:
        parser.error("--leagues must contain at least one league code (unless --proof)")

    return run_session(
        session_id=args.session_id,
        date=args.date,
        leagues=leagues,
        mlb_games=args.mlb_games,
        mlb_props=args.mlb_props,
        tennis_games=args.tennis_games,
        fifa_games=args.fifa_games,
        mode=args.mode,
        require_actionable_min=args.require_actionable_min,
        home_team=args.home_team,
        away_team=args.away_team,
        reopen=args.reopen,
        ingest=args.ingest,
        render_report=args.render_report,
        close=args.close,
        proof=args.proof,
        keep_proof_artifacts=args.keep_proof_artifacts,
        report_out_dir=args.report_out_dir,
        sidecar_dir=args.sidecar_dir,
        trace_inbox=args.trace_inbox,
        bankroll=args.bankroll,
        bankroll_confirmed=args.bankroll_confirmed,
        model_version=args.model_version,
        dry_run=args.dry_run,
        skip_preflight=args.skip_preflight,
        verbose=args.verbose,
        db=args.db,
    )


if __name__ == "__main__":
    raise SystemExit(main())
