"""Shared path resolvers for the Omega repository layout."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the Omega repository root from the src/ package layout."""
    return Path(__file__).resolve().parents[2]


def runtime_root() -> Path:
    """Return the default runtime-state root.

    ``OMEGA_RUNTIME_DIR`` changes only default runtime locations. Explicit DB
    arguments and ``OMEGA_TRACE_DB`` keep their higher precedence in TraceStore.
    """
    override = os.environ.get("OMEGA_RUNTIME_DIR")
    if override:
        return Path(override).expanduser()
    return repo_root() / "var"


def runtime_path(*parts: str) -> Path:
    """Return a path under the active runtime root."""
    return runtime_root().joinpath(*parts)


def var_dir() -> Path:
    """Return the active runtime-state directory.

    Kept for existing callers; new code should prefer ``runtime_root()`` when
    the environment override matters semantically.
    """
    return runtime_root()


def inbox_dir() -> Path:
    """Return the canonical runtime inbox directory."""
    return runtime_path("inbox")


def trace_inbox_dir() -> Path:
    """Return the canonical trace-export inbox directory."""
    return runtime_path("inbox", "traces")


def session_inbox_dir() -> Path:
    """Return the canonical session-sidecar inbox directory."""
    return runtime_path("inbox", "sessions")


def closing_lines_inbox_dir() -> Path:
    """Return the canonical closing-line snapshot inbox directory."""
    return runtime_path("inbox", "closing_lines")


def action_plan_dir() -> Path:
    """Return the canonical runtime action-plan directory."""
    return runtime_path("inbox", "action_plans")


def reports_dir() -> Path:
    """Return the canonical generated reports directory."""
    return runtime_path("reports")


def latest_report_path() -> Path:
    """Return the canonical generated calibration report path."""
    return runtime_path("reports", "latest.md")


def run_audits_dir() -> Path:
    """Return the canonical generated session-audit reports directory."""
    return runtime_path("reports", "run_audits")


def trace_db_path() -> Path:
    """Return the default SQLite trace store path under the runtime root."""
    return runtime_path("omega_traces.db")


def default_trace_db_path() -> Path:
    """Return the default SQLite trace store path under the runtime root.

    NOTE: runtime_root() already returns <repo>/var by default, so the filename
    here must NOT be prefixed with another "var/". Doing so produced a stray
    <repo>/var/var/omega_traces.db that silently captured writes from any
    no-explicit-path caller (see the 2026-06-01 stale-dup cleanup).
    """
    return trace_db_path()
