#!/usr/bin/env python
"""run_slate.py - run a built entries file through omega_run_batch.

The run half of the retired scratch runners (run_today_mlb_batch.py,
run_slate_YYYYMMDD.py, ...), minus their defects:

- Session id, bankroll, and entries path are arguments, not per-day edits.
- The session sidecar must already exist (open it via ``omega-session-run`` /
  the session bootstrap first); running without one requires an explicit
  ``--allow-missing-sidecar``. The scratch runners appended audit events to a
  hardcoded sidecar path and failed only at append time.
- Per-entry statuses are read from the envelope's real vocabulary
  (``ok`` / ``skipped`` / ``error``). The June scratch runner filtered on
  ``status == "success"`` — a value the batch tool never emits — and so always
  collected zero trace ids from the per-entry rows.
- The full result envelope is persisted next to the entries file, so the
  trace ids and skip reasons survive the console session.

Usage
-----
    python tools/run_slate.py --entries var/slates/mlb-2026-07-05.entries.json \
        --session-id sess-20260705-XXXXXXXXXX --bankroll 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from omega.mcp.server import omega_run_batch
    from omega.trace.session_sidecar import append_audit_events
except ImportError:  # running outside an installed env — fall back to src layout
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from omega.mcp.server import omega_run_batch
    from omega.trace.session_sidecar import append_audit_events


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(sidecar: Path | None, status: str, notes: str, trace_ids: list[str]) -> None:
    if sidecar is None:
        return
    try:
        append_audit_events(
            sidecar,
            [
                {
                    "ts": _now(),
                    "event_type": "engine_run",
                    "step": "run_slate",
                    "status": status,
                    "notes": notes,
                    "trace_ids": trace_ids,
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 — an audit failure must not eat the run result
        print(f"WARNING: sidecar audit append failed: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run built slate entries through omega_run_batch.")
    parser.add_argument("--entries", required=True, help="Entries JSON from build_slate_entries.py")
    parser.add_argument("--session-id", required=True, help="Open session id (sess-...)")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll (default 1000)")
    parser.add_argument(
        "--results-out",
        help="Result envelope path (default: <entries>.result.json)",
    )
    parser.add_argument(
        "--allow-missing-sidecar",
        action="store_true",
        help="Proceed without a session sidecar (audit events are then not recorded)",
    )
    args = parser.parse_args(argv)

    entries_path = Path(args.entries)
    try:
        entries = json.loads(entries_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot load entries: {exc}")
        return 2
    if not isinstance(entries, list) or not entries:
        print(f"ERROR: {entries_path} is not a non-empty JSON list of entries")
        return 2

    sidecar: Path | None = REPO_ROOT / "var" / "inbox" / "sessions" / f"{args.session_id}.json"
    if not sidecar.exists():
        if not args.allow_missing_sidecar:
            print(
                f"ERROR: session sidecar not found: {sidecar}\n"
                "Open the session first (omega-session-run / session bootstrap), or pass "
                "--allow-missing-sidecar to run without audit events."
            )
            return 2
        print("WARNING: running without a session sidecar; no audit events will be recorded.")
        sidecar = None

    n_games = sum(1 for e in entries if e.get("kind") == "game")
    n_props = len(entries) - n_games
    print(
        f"Running {len(entries)} entries ({n_games} games, {n_props} props) "
        f"for session {args.session_id}..."
    )
    _audit(
        sidecar,
        "ok",
        f"run_slate start: {len(entries)} entries from {entries_path.name}",
        [],
    )

    result = omega_run_batch(entries=entries, bankroll=args.bankroll, session_id=args.session_id)

    print("\n=== BATCH RESULT ===")
    print(
        f"status={result.get('status')}  total={result.get('entries_total')}  "
        f"ok={result.get('entries_ok')}  skipped={result.get('entries_skipped')}  "
        f"error={result.get('entries_error')}"
    )
    for row in result.get("results", []):
        status = row.get("status")
        line = f"- [{status}] {row.get('identifier')}"
        if status == "ok":
            line += f"  trace={row.get('trace_id')}"
            if row.get("rsvg_status"):
                line += f"  rsvg={row['rsvg_status']}"
        elif row.get("reason"):
            line += f"  reason={row['reason']}"
        elif row.get("error"):
            line += f"  error={row['error']}"
        print(line)

    trace_ids = result.get("trace_ids") or []
    overall = result.get("status")
    _audit(
        sidecar,
        "ok" if overall == "ok" else ("warn" if overall == "partial" else "fail"),
        (
            f"run_slate done: status={overall} ok={result.get('entries_ok')} "
            f"skipped={result.get('entries_skipped')} error={result.get('entries_error')}"
        ),
        trace_ids,
    )

    results_out = (
        Path(args.results_out) if args.results_out else entries_path.with_suffix(".result.json")
    )
    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\n{len(trace_ids)} trace id(s); full envelope -> {results_out}")

    return {"ok": 0, "partial": 1}.get(overall, 2)


if __name__ == "__main__":
    sys.exit(main())
