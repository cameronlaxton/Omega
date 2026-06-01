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
    """Return the default SQLite trace store path under var/."""
    return var_dir() / "var/omega_traces.db"
