"""
scripts/validate_artifact_policy.py — enforce Omega's artifact-segregation policy.

Grounds the "Source-of-Truth Authority" rules in PROJECT_STATE.md in an
executable check so runtime artifacts cannot silently pollute source review.
It owns no opinions of its own: the source of truth for "what is a runtime
artifact" is the repo's own .gitignore (plus the user/global git excludes).

Checks:
  1. No committed-but-ignored files. `git ls-files --cached --ignored
     --exclude-standard` must be empty — if a tracked file matches an ignore
     rule, the repo is committing something it already declared transient
     (runtime DB sidecars, tmp probe journals, inbox/traces/processed/, local
     tool settings, etc.).
  2. No scratch artifacts at the repo root. PROJECT_STATE/.gitignore forbid
     RUN_AUDIT.md and RUN_TRACE.jsonl from reappearing at the root.

Usage:
    python scripts/validate_artifact_policy.py

Exit codes:
    0 — clean
    1 — at least one policy violation (with per-file reasons)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Scratch artifacts that must never sit at the repo root (see .gitignore).
_FORBIDDEN_ROOT_FILES = ("RUN_AUDIT.md", "RUN_TRACE.jsonl")


def _committed_but_ignored() -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files", "--cached", "--ignored", "--exclude-standard"],
        cwd=_REPO_ROOT,
        text=True,
    )
    return [line for line in out.splitlines() if line.strip()]


def _root_scratch_present() -> list[str]:
    return [name for name in _FORBIDDEN_ROOT_FILES if (_REPO_ROOT / name).exists()]


def main() -> int:
    print("Running Artifact-Policy Validation...")
    failed = False

    ignored_tracked = _committed_but_ignored()
    if ignored_tracked:
        failed = True
        print(
            f"ERROR: {len(ignored_tracked)} tracked file(s) match a git ignore rule "
            "(runtime artifacts must not be committed). Untrack with "
            "`git rm --cached <file>`:"
        )
        for f in ignored_tracked:
            print(f"  - {f}")

    scratch = _root_scratch_present()
    if scratch:
        failed = True
        print("ERROR: forbidden scratch artifact(s) present at repo root (do not regenerate):")
        for f in scratch:
            print(f"  - {f}")

    if failed:
        print("FAILED: Artifact-policy checks failed.")
        return 1
    print("SUCCESS: Artifact-policy checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
