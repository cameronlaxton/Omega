"""Shared path resolvers for the Omega repository layout."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the Omega repository root from the src/ package layout."""
    return Path(__file__).resolve().parents[2]


def var_dir() -> Path:
    """Return the runtime-state directory under the repository root."""
    return repo_root() / "var"


def default_trace_db_path() -> Path:
    """Return the default SQLite trace store path under var/.

    NOTE: var_dir() already returns <repo>/var, so the filename here must NOT be
    prefixed with another "var/" — doing so produced a stray <repo>/var/var/
    omega_traces.db that silently captured writes from any no-explicit-path
    caller (see the 2026-06-01 stale-dup cleanup).
    """
    return var_dir() / "omega_traces.db"
