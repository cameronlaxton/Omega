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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.store import db_status, seed_runtime_db  # noqa: E402


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only TraceStore DB doctor.")
    parser.add_argument(
        "--db",
        default=None,
        help="Requested DB path (default resolver: OMEGA_TRACE_DB or var/omega_traces.db)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="MUTATING: copy a valid non-empty source DB into an absent runtime DB",
    )
    args = parser.parse_args(argv)

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
        print(f"SEEDED {result['runtime']} from {result['source']} ({result['trace_count']} traces)")
        status = db_status(args.db)

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(_render(status))

    # Non-zero only when an existing effective DB is actually corrupt.
    if status["effective_exists"] and status["effective_integrity_ok"] is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())




