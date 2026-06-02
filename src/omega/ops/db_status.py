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
        "trace_store_db_source": status["sourc