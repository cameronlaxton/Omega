"""
omega.ops.install_hooks â€” install tracked git hooks into .git/hooks/.

Copies every hook in tools/hooks/ into the repo's .git/hooks/ directory and
marks it executable. The tracked copies under tools/hooks/ are the source of
truth (they survive clone); .git/hooks/ is per-clone and not version-controlled,
so this installer wires them up.

Usage:
    omega-install-hooks
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HOOK_SRC = _REPO_ROOT / "tools" / "hooks"


def _git_hooks_dir() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--git-path", "hooks"], cwd=_REPO_ROOT, text=True
    ).strip()
    path = Path(out)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def main() -> int:
    if not _HOOK_SRC.is_dir():
        print(f"No hook source directory: {_HOOK_SRC}")
        return 1

    hooks_dir = _git_hooks_dir()
    hooks_dir.mkdir(parents=True, exist_ok=True)

    installed = 0
    for src in sorted(_HOOK_SRC.iterdir()):
        if not src.is_file():
            continue
        dst = hooks_dir / src.name
        shutil.copyfile(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"installed {src.name} -> {dst}")
        installed += 1

    if installed == 0:
        print("No hooks found to install.")
        return 1
    print(f"Done. Installed {installed} hook(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())



