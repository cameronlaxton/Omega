"""omega-session-run: hardened daily multi-sport session orchestrator.

Drives the non-engine phases of a daily multi-sport Omega session so that
the operator does not need to manually discover commands, schema details, or
sport-key mappings.

Workflow
--------
Phase 1  validate args, generate session ID (if not supplied), print header.
Phase 2  run omega-cowork-preflight (always).
Phase 3  soccer prior-coverage gate (if any --leagues value is a soccer league).
Phase 4  list active tennis events (if TENNIS/ATP/WTA in --leagues).
Phase 5  print a structured ANALYSIS PLAN — the exact MCP or CLI commands to run
         for the engine phase.  The operator carries these out (MCP or batch script).
Phase 6  if --ingest: run omega-ingest-traces to pull exported traces into the DB.
Phase 7  if --render-report: run omega-render-session-report.

The command exits 0 when all completed phases succeeded, 1 on hard failure, and 2
when coverage gates downgraded the recommended output mode (non-fatal; the plan is
still emitted so the operator can run the session in research mode).

Example
-------
    omega-session-run \\
        --session-id sess-20260619-live \\
        --date 2026-06-19 \\
        --leagues MLB,TENNIS,FIFA_WORLD_CUP_2026 \\
        --mlb-games 9 --mlb-props 10 \\
        --tennis-games 10 \\
        --fifa-games 2 \\
        --mode research-lean \\
        --require-actionable-min 1 \\
        --render-report --ingest

After the plan is printed and the engine phase completes (separately, via MCP or
omega-run-analyze batch), re-invoke with --ingest --render-report --session-id
<same id> to close out the session.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logger = logging.getLogger("omega.ops.session_run")
UTC = timezone.utc

# League codes classified by sport family
_SOCCER_LEAGUES = frozenset({
    "EPL", "MLS", "LA_LIGA", "BUNDESLIGA", "SERIE_A", "LIGUE_1",
    "CHAMPIONS_LEAGUE", "WORLD_CUP", "FIFA_WORLD_CUP_2026", "FIFA_INTL",
    "FIFA_FRIENDLY", "FIFA_NATIONS_LEAGUE", "FIFA_QUALIFIERS",
})
_TENNIS_LEAGUES = frozenset({"ATP", "WTA", "GRAND_SLAM", "TENNIS"})
_PROP_CAPABLE_LEAGUES = frozenset({"MLB", "NBA", "NFL", "NHL", "WNBA"})


def _generate_session_id(date: str) -> str:
    ts = datetime.now(UTC).strftime("%H%M")
    return f"sess-{date}-{ts}"


def _league_list(raw: str) -> list[str]:
    return [lg.strip().upper() for lg in raw.split(",") if lg.strip()]


def _soccer_leagues_in(leagues: list[str]) -> list[str]:
    return [lg for lg in leagues if lg in _SOCCER_LEAGUES]


def _tennis_leagues_in(leagues: list[str]) -> list[str]:
    return [lg for lg in leagues if lg in _TENNIS_LEAGUES]


def _run_subprocess(cmd: list[str], *, label: str, dry_run: bool) -> int:
    """Run a subprocess and return its exit code.  Prints output live."""
    if dry_run:
        logger.info("[DRY-RUN] would run: %s", " ".join(cmd))
        return 0
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _phase_preflight(dry_run: bool) -> int:
    """Phase 2: run omega-cowork-preflight."""
    print("\n[Phase 2] Running cowork preflight gate...")
    rc = _run_subprocess(["omega-cowork-preflight"], label="preflight", dry_run=dry_run)
    if rc != 0 and not dry_run:
        print(f"[FAIL] omega-cowork-preflight exited {rc}. Fix the reported issues before proceeding.")
        return rc
    print("[OK] Preflight passed." if rc == 0 else "[DRY-RUN] Preflight skipped.")
    return rc


def _phase_soccer_gate(
    soccer_leagues: list[str],
    *,
    home_team: str | None,
    away_team: str | None,
    dry_run: bool,
    db: str | None,
) -> tuple[int, list[str]]:
    """Phase 3: soccer prior-coverage gate.  Returns (exit_code, downgraded_leagues)."""
    downgraded: list[str] = []
    if not soccer_leagues:
        return 0, downgraded

    print(f"\n[Phase 3] Checking soccer prior coverage for: {', '.join(soccer_leagues)}")
    for league in soccer_leagues:
        cmd = ["omega-soccer-prior-coverage", "--league", league, "--format", "summary"]
        if home_team:
            cmd += ["--home-team", home_team]
        if away_team:
            cmd += ["--away-team", away_team]
        if db:
            cmd += ["--db", db]

        rc = _run_subprocess(cmd, label=f"soccer-gate-{league}", dry_run=dry_run)
        if rc == 2:
            print(
                f"  [DOWNGRADE] {league}: weak/no DC rho profile → output mode forced to "
                f"research_candidate for this league."
            )
            downgraded.append(league)
        elif rc != 0 and rc != 2 and not dry_run:
            print(f"  [ERROR] Soccer prior coverage check for {league} failed (exit {rc}). "
                  "Session can proceed but FIFA outputs may be unreliable.")

    return 0, downgraded


def _phase_tennis_events(
    tennis_leagues: list[str],
    *,
    date: str,
    dry_run: bool,
) -> int:
    """Phase 4: list active tennis events for discovered sport keys."""
    if not tennis_leagues:
        return 0

    print(f"\n[Phase 4] Listing active tennis events for: {', '.join(tennis_leagues)}")
    for tour in tennis_leagues:
        league = "ATP" if tour == "TENNIS" else tour
        cmd = ["omega-resolve-odds", "--list-events", "--league", league, "--format", "summary"]
        rc = _run_subprocess(cmd, label=f"tennis-events-{league}", dry_run=dry_run)
        if rc not in (0, 2) and not dry_run:
            print(
                f"  [WARN] Tennis event listing for {league} returned {rc}. "
                "No active tournament key may be available right now."
            )
    return 0


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
        "ENGINE PHASE (run via MCP or omega-run-analyze batch script):",
        "",
        "Option A — MCP (preferred when server is running):",
        "  Use omega_run_batch with the following parameters.",
        "  See batch_rule in AGENTS.md for exact contract.",
        "",
        "  omega_run_batch(",
        f"    session_id=\"{session_id}\",",
        f"    game_date=\"{date}\",",
    ]

    if "MLB" in leagues:
        lines += [
            f"    # MLB: {mlb_games} games + {mlb_props} prop sweeps",
            "    # Include: h2h, spreads, totals for each game",
            "    # Props: strikeouts_pitched, hits, total_bases, earned_runs",
        ]

    if any(lg in _TENNIS_LEAGUES for lg in leagues):
        lines += [
            f"    # TENNIS: {tennis_games} matches",
            "    # Requires: surface context, serve_win_pct, return_win_pct",
        ]

    if any(lg in _SOCCER_LEAGUES for lg in leagues):
        effective_mode = "research_candidate" if downgraded_leagues else mode
        lines += [
            f"    # FIFA/Soccer: {fifa_games} matches",
            f"    # output_mode={effective_mode!r} (downgraded due to prior coverage)" if downgraded_leagues else "",
            "    # Requires: rho prior (injected automatically from priors_dixon_coles)",
        ]

    lines += [
        "  )",
        "",
        "Option B — Batch script (MCP unavailable):",
        "  Ensure cowork_preflight.run_formal_output_gate() passes, then call",
        "  omega.core.contracts.service.analyze() per game/prop per AGENTS.md batch rule.",
        "",
        "After engine phase completes, export traces to var/inbox/traces/ and run:",
        "",
        f"  omega-ingest-traces                          # ingest exported traces",
        f"  omega-render-session-report \\",
        f"    --kind intake \\",
        f"    --session-id {session_id}                  # render audit report",
        "",
        "Or re-invoke session-run with the same --session-id plus --ingest --render-report.",
        "=" * 70,
    ]

    # Remove blank strings that sneak in from conditional appends
    return "\n".join(line for line in lines if line is not None)


def _phase_ingest(session_id: str, *, dry_run: bool, verbose: bool) -> int:
    """Phase 6: omega-ingest-traces."""
    print(f"\n[Phase 6] Ingesting traces for session {session_id}...")
    cmd = ["omega-ingest-traces"]
    if verbose:
        cmd.append("--verbose")
    rc = _run_subprocess(cmd, label="ingest-traces", dry_run=dry_run)
    if rc != 0 and not dry_run:
        print(f"[FAIL] omega-ingest-traces exited {rc}.")
    else:
        print("[OK] Traces ingested." if not dry_run else "[DRY-RUN]")
    return rc


def _phase_render_report(
    session_id: str,
    *,
    out_dir: Path | None,
    dry_run: bool,
    verbose: bool,
    db: str | None,
) -> int:
    """Phase 7: omega-render-session-report."""
    print(f"\n[Phase 7] Rendering session report for {session_id}...")
    cmd = ["omega-render-session-report", "--kind", "intake", "--session-id", session_id]
    if out_dir:
        cmd += ["--out-dir", str(out_dir)]
    if verbose:
        cmd.append("--verbose")
    if db:
        cmd += ["--db", db]
    rc = _run_subprocess(cmd, label="render-report", dry_run=dry_run)
    if rc != 0 and not dry_run:
        print(f"[FAIL] omega-render-session-report exited {rc}.")
    else:
        print("[OK] Report written." if not dry_run else "[DRY-RUN]")
    return rc


def run_session(
    *,
    session_id: str | None = None,
    date: str | None = None,
    leagues: list[str],
    mlb_games: int = 0,
    mlb_props: int = 0,
    tennis_games: int = 0,
    fifa_games: int = 0,
    mode: str = "research-lean",
    require_actionable_min: int = 0,
    home_team: str | None = None,
    away_team: str | None = None,
    ingest: bool = False,
    render_report: bool = False,
    report_out_dir: Path | None = None,
    dry_run: bool = False,
    skip_preflight: bool = False,
    verbose: bool = False,
    db: str | None = None,
) -> int:
    """Programmatic entry point (also called by main())."""
    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    session_id = session_id or _generate_session_id(date)

    print(
        f"\nomega-session-run  |  session={session_id}  date={date}  "
        f"leagues={','.join(leagues)}"
    )
    if dry_run:
        print("  [DRY-RUN MODE — no subprocess calls will be executed]")

    # Phase 2: preflight
    if not skip_preflight:
        rc = _phase_preflight(dry_run=dry_run)
        if rc != 0 and not dry_run:
            return 1

    # Phase 3: soccer gate
    soccer_leagues = _soccer_leagues_in(leagues)
    _, downgraded = _phase_soccer_gate(
        soccer_leagues,
        home_team=home_team,
        away_team=away_team,
        dry_run=dry_run,
        db=db,
    )

    # Phase 4: tennis discovery
    tennis_leagues = _tennis_leagues_in(leagues)
    _phase_tennis_events(tennis_leagues, date=date, dry_run=dry_run)

    # Phase 5: print analysis plan
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
    print(f"\n[Phase 5] Analysis Plan\n")
    print(plan)

    overall_rc = 2 if downgraded else 0

    # Phase 6: ingest (optional, run after engine phase completes)
    if ingest:
        rc = _phase_ingest(session_id, dry_run=dry_run, verbose=verbose)
        if rc != 0 and not dry_run:
            overall_rc = 1

    # Phase 7: render report (optional)
    if render_report:
        rc = _phase_render_report(
            session_id,
            out_dir=report_out_dir,
            dry_run=dry_run,
            verbose=verbose,
            db=db,
        )
        if rc != 0 and not dry_run:
            overall_rc = 1

    print(
        f"\nomega-session-run complete  |  session={session_id}  "
        f"exit={overall_rc}"
    )
    return overall_rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Unique session identifier (e.g. sess-20260619-live). Auto-generated if omitted.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Analysis date YYYY-MM-DD (default: today UTC).",
    )
    parser.add_argument(
        "--leagues",
        required=True,
        help="Comma-separated Omega league codes (e.g. MLB,TENNIS,FIFA_WORLD_CUP_2026).",
    )
    parser.add_argument("--mlb-games", type=int, default=0, help="Number of MLB game analyses to plan.")
    parser.add_argument("--mlb-props", type=int, default=0, help="Number of MLB prop analyses to plan.")
    parser.add_argument("--tennis-games", type=int, default=0, help="Number of tennis match analyses to plan.")
    parser.add_argument("--fifa-games", type=int, default=0, help="Number of FIFA/soccer game analyses to plan.")
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
        "--ingest",
        action="store_true",
        help="After printing the plan, run omega-ingest-traces. "
             "Use this when re-invoking after the engine phase has completed.",
    )
    parser.add_argument(
        "--render-report",
        action="store_true",
        help="After ingest (or plan), run omega-render-session-report.",
    )
    parser.add_argument(
        "--report-out-dir",
        type=Path,
        default=None,
        help="Directory for rendered report files.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the cowork-preflight gate (not recommended for production sessions).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; do not execute any subprocess.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--db", default=None, help="SQLite path override for DB-dependent phases.")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    leagues = _league_list(args.leagues)
    if not leagues:
        parser.error("--leagues must contain at least one league code")

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
        ingest=args.ingest,
        render_report=args.render_report,
        report_out_dir=args.report_out_dir,
        dry_run=args.dry_run,
        skip_preflight=args.skip_preflight,
        verbose=args.verbose,
        db=args.db,
    )


if __name__ == "__main__":
    raise SystemExit(main())
