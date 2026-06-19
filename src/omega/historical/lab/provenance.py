"""Git provenance capture for a lab run.

The historical path records ``current_code_version()`` (package metadata) but
never the git commit or working-tree state. Auto-promotion needs both: a clean,
attributable tree is a precondition for flipping a profile to production. These
thin helpers mirror the subprocess idiom already used in
``omega.ops.cowork_preflight`` (``git status --porcelain`` / ``rev-parse HEAD``)
and fail **closed** — an unreadable git state is treated as "dirty" so promotion
refuses rather than proceeding blind.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from omega.historical.contracts import current_code_version

# repo root = .../src/omega/historical/lab/provenance.py → parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _git(repo_root: Path | str, args: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode, (result.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def git_commit(repo_root: Path | str | None = None) -> str:
    """Resolved HEAD commit sha, or ``"unknown"`` when git is unavailable."""
    rc, out = _git(repo_root or _REPO_ROOT, ["rev-parse", "--verify", "HEAD"])
    return out if rc == 0 and out else "unknown"


def working_tree_dirty(repo_root: Path | str | None = None) -> bool:
    """True when the tree has uncommitted changes — or when git state is unreadable.

    Fail-closed: a non-zero ``git status`` (corrupt repo, not a repo, locked
    index) returns ``True`` so an armed auto-promote will refuse.
    """
    rc, out = _git(repo_root or _REPO_ROOT, ["status", "--porcelain"])
    if rc != 0:
        return True
    return bool(out)


def capture(repo_root: Path | str | None = None) -> dict[str, object]:
    """Provenance triple for a :class:`HistoricalLabRun`."""
    root = repo_root or _REPO_ROOT
    return {
        "code_version": current_code_version(),
        "git_commit": git_commit(root),
        "working_tree_dirty": working_tree_dirty(root),
    }
