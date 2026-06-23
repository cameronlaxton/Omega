"""
omega-db-status â€” read-only TraceStore DB doctor.

Reports the repo/default DB path, the would-be per-user runtime path (used when
the repo DB sits on a FUSE/network mount), existence, integrity_check, trace
counts, runtime/source divergence, latest session IDs, and a recommended action
â€” WITHOUT instantiating TraceStore, so it still works when the normal redirect
guard would refuse to open (e.g. a missing/malformed source DB).

Default mode is strictly read-only: it never seeds, copies, repairs, or mutates
any DB. The only mutating path is the explicit ``--seed`` flag, which copies a
valid non-empty source DB into an ABSENT runtime path (never overwriting).

Usage:
    omega-db-status
    omega-db-status --json
    omega-db-status --db /path/to/omega_traces.db
    omega-db-status --seed        # explicit: populate absent runtime DB

Exit codes:
    0 â€” report produced (and seed succeeded if requested)
    1 â€” effective DB exists but fails integrity_check, or --seed preconditions failed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.store import TraceStore, db_status, seed_runtime_db  # noqa: E402


def _render(status: dict) -> str:
    lines: list[str] = []
    lines.append("Omega TraceStore - DB status")
    lines.append("=" * 40)
    lines.append(f"requested            : {status['requested']}")
    lines.append(f"OMEGA_TRACE_DB       : {status['env_override']}")
    lines.append(f"repo default path    : {status['repo_default_path']}")
    lines.append(f"effective path       : {status['effective_path']}")
    lines.append(f"path source          : {status['source']}")
    lines.append(f"would-be runtime path: {status['would_be_runtime_path']}")
    lines.append("")
    lines.append(f"default exists       : {status['default_exists']}")
    lines.append(f"default integrity_ok : {status['default_integrity_ok']}")
    lines.append(f"default trace_count  : {status['default_trace_count']}")
    lines.append(f"effective exists     : {status['effective_exists']}")
    lines.append(f"effective integrity  : {status['effective_integrity_ok']}")
    lines.append(f"effective trace_count: {status['effective_trace_count']}")
    lines.append(f"runtime exists       : {status['runtime_exists']}")
    lines.append(f"EMPTY_HISTORY_MODE   : {str(status['empty_history_mode']).lower()}")
    lines.append(f"latest session_ids   : {', '.join(status['latest_session_ids']) or '(none)'}")
    if status["divergence"]:
        d = status["divergence"]
        lines.append("")
        lines.append("DIVERGENCE (no auto-merge):")
        lines.append(f"  source  {d['source_path']} -> {d['source_trace_count']} traces")
        lines.append(f"  runtime {d['runtime_path']} -> {d['runtime_trace_count']} traces")
    lines.append("")
    lines.append(f"RECOMMENDED ACTION   : {status['recommended_action']}")
    return "\n".join(lines)


def _path_under(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _workspace_identity(status: dict[str, Any]) -> str:
    source = status["source"]
    if source == "env_override":
        return "Env Override"
    if source == "auto_redirect_network_fs":
        return "Runtime Redirect"

    effective = Path(status["effective_path"])
    active_workspace = os.environ.get("OMEGA_LOCAL_WORKSPACE")
    if active_workspace:
        active_db = Path(active_workspace) / "var" / "omega_traces.db"
        try:
            return (
                "Matches Active Workspace"
                if effective.resolve() == active_db.resolve()
                else "Different Workspace"
            )
        except OSError:
            return "Unknown Workspace"

    default_path = Path(status["repo_default_path"])
    try:
        if effective.resolve() == default_path.resolve() or _path_under(effective, _REPO_ROOT):
            return "Matches Active Workspace"
    except OSError:
        pass
    return "Unknown Workspace"


def _identity_meta(status: dict[str, Any]) -> dict[str, Any]:
    path = str(Path(status["effective_path"]).resolve())
    return {
        "trace_store_db_path": path,
        "trace_store_db_source": status["source"],
        "workspace_identity": _workspace_identity(status),
    }


def _identity_header(status: dict[str, Any]) -> str:
    meta = _identity_meta(status)
    return f"TraceStore DB Path: {meta['trace_store_db_path']} [{meta['workspace_identity']}]"


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("--has-outcome must be true or false")


def _trace_row(trace: dict[str, Any]) -> dict[str, Any]:
    row = trace.get("_row") or {}
    result = trace.get("result") or {}
    return {
        "trace_id": row.get("trace_id") or trace.get("trace_id"),
        "timestamp": row.get("timestamp") or trace.get("timestamp"),
        "session_id": trace.get("session_id"),
        "kind": row.get("kind") or trace.get("kind"),
        "league": row.get("league") or trace.get("league"),
        "matchup": row.get("matchup") or trace.get("matchup"),
        "execution_mode": row.get("execution_mode") or trace.get("execution_mode"),
        "aggregate_quality": row.get("aggregate_quality") or trace.get("aggregate_quality"),
        "result_status": result.get("status"),
        "has_outcome": bool(trace.get("_outcome") or trace.get("_prop_outcomes")),
    }


def _render_traces(status: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    lines = [_identity_header(status), f"Trace rows: {len(traces)}", ""]
    for trace in traces:
        row = _trace_row(trace)
        lines.append(
            f"{row['timestamp'] or '(no timestamp)'} | {row['trace_id']} | "
            f"{row['league'] or '(league?)'} | {row['matchup'] or '(matchup?)'} | "
            f"status={row['result_status'] or '(unknown)'} | outcome={row['has_outcome']}"
        )
    if not traces:
        lines.append("(no traces matched)")
    return "\n".join(lines)


def _render_ledger(status: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [_identity_header(status), f"Ledger rows: {len(rows)}", ""]
    for row in rows:
        lines.append(
            f"{row.get('decision_timestamp') or row.get('created_at') or '(no timestamp)'} | "
            f"{row.get('ledger_id')} | trace={row.get('trace_id')} | "
            f"{row.get('league') or '(league?)'} | {row.get('market')} | "
            f"{row.get('selection_descriptor')} | status={row.get('status')} | "
            f"provenance={row.get('provenance')}"
        )
    if not rows:
        lines.append("(no ledger rows matched)")
    return "\n".join(lines)


def _query_traces(args: argparse.Namespace) -> list[dict[str, Any]]:
    store = TraceStore(db_path=args.db, read_only=True)
    try:
        if args.trace_id:
            trace = store.get_trace(args.trace_id)
            return [trace] if trace is not None else []
        if args.session_id:
            return store.query_by_session(args.session_id)[: args.limit]
        return store.query_traces(
            league=args.league,
            start=args.start,
            end=args.end,
            has_outcome=args.has_outcome,
            execution_mode=args.execution_mode,
            limit=args.limit,
        )
    finally:
        store.close()


def _query_ledger(args: argparse.Namespace) -> list[dict[str, Any]]:
    store = TraceStore(db_path=args.db, read_only=True)
    try:
        return store.query_ledger(
            league=args.league,
            sport=args.sport,
            status=args.status,
            provenance=args.provenance,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
    finally:
        store.close()


def _emit_jsonl(meta: dict[str, Any], rows: list[dict[str, Any]], row_type: str) -> None:
    print(json.dumps({"type": "metadata", **meta}, sort_keys=True))
    for row in rows:
        print(json.dumps({"type": row_type, **row}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only TraceStore DB doctor.")
    parser.add_argument(
        "--db",
        default=None,
        help="Requested DB path (default resolver: OMEGA_TRACE_DB or var/omega_traces.db)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument(
        "--format",
        choices=["summary", "json", "jsonl"],
        default="summary",
        help="Output format for status and lookup modes",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="MUTATING: copy a valid non-empty source DB into an absent runtime DB",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--query-traces", action="store_true", help="Read persisted trace rows")
    mode.add_argument("--view-ledger", action="store_true", help="Read bet_ledger rows")
    parser.add_argument("--trace-id", help="Trace ID for --query-traces")
    parser.add_argument("--session-id", help="Session ID for --query-traces")
    parser.add_argument("--league", help="League filter")
    parser.add_argument("--sport", help="Sport filter for --view-ledger")
    parser.add_argument("--status", help="Ledger status filter")
    parser.add_argument("--provenance", help="Ledger provenance filter")
    parser.add_argument("--start", help="Inclusive ISO timestamp lower bound")
    parser.add_argument("--end", help="Inclusive ISO timestamp upper bound")
    parser.add_argument("--has-outcome", type=_parse_bool, help="true or false")
    parser.add_argument("--execution-mode", help="Trace execution mode filter")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to return")
    args = parser.parse_args(argv)
    output_format = "json" if args.json else args.format

    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.seed and (args.query_traces or args.view_ledger):
        parser.error("--seed cannot be combined with lookup modes")

    status = db_status(args.db)

    if args.seed:
        # Seed only makes sense when the effective path is a redirected runtime DB.
        source = status["repo_default_path"]
        runtime = status["would_be_runtime_path"]
        try:
            result = seed_runtime_db(source=source, runtime=runtime)
        except RuntimeError as exc:
            print(f"SEED REFUSED: {exc}", file=sys.stderr)
            return 1
        print(
            f"SEEDED {result['runtime']} from {result['source']} ({result['trace_count']} traces)"
        )
        status = db_status(args.db)

    if args.query_traces:
        traces = _query_traces(args)
        meta = {**_identity_meta(status), "query": "traces", "count": len(traces)}
        if output_format == "json":
            print(json.dumps({**meta, "traces": traces}, indent=2))
        elif output_format == "jsonl":
            _emit_jsonl(meta, traces, "trace")
        else:
            print(_render_traces(status, traces))
        return 0

    if args.view_ledger:
        rows = _query_ledger(args)
        meta = {**_identity_meta(status), "query": "ledger", "count": len(rows)}
        if output_format == "json":
            print(json.dumps({**meta, "ledger": rows}, indent=2))
        elif output_format == "jsonl":
            _emit_jsonl(meta, rows, "ledger")
        else:
            print(_render_ledger(status, rows))
        return 0

    if output_format == "json":
        print(json.dumps(status, indent=2))
    elif output_format == "jsonl":
        print(json.dumps({"type": "status", **status}, sort_keys=True))
    else:
        print(_render(status))

    # Non-zero only when an existing effective DB is actually corrupt.
    if status["effective_exists"] and status["effective_integrity_ok"] is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
