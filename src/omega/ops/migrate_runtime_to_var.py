"""omega.ops.migrate_runtime_to_var — one-time migration of legacy root-level
runtime artifacts into the canonical ``var/`` runtime root.

Historically some CLIs defaulted to repo-root ``inbox/`` and ``reports/`` while
the canonical runtime root is ``var/`` (see docs/phase6/ARTIFACT_AUTHORITY.md).
This helper merge-moves any leftover root artifacts under the active runtime root
and deletes the now-empty root dirs, so ``var/`` is the single source of truth.

Mapping (recursive, preserves subfolders):
    <repo>/inbox/**   -> <runtime_root>/inbox/**
    <repo>/reports/** -> <runtime_root>/reports/**

Collision policy: keep the file with the newer mtime — a newer destination is
never silently clobbered; the stale root copy is dropped instead. Empty source
directories (and the root ``inbox/`` and ``reports/`` themselves) are removed
after a successful ``--apply``.

Usage:
    omega-migrate-runtime              # dry-run (default): print the plan
    omega-migrate-runtime --apply      # perform the move + delete root dirs

Exit codes:
    0 — dry-run printed, nothing to do, or apply completed with no errors
    1 — unsafe runtime root, or an error occurred during apply
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.paths import repo_root, runtime_root  # noqa: E402

logger = logging.getLogger("omega.ops.migrate_runtime_to_var")

# Root-level runtime dir names that move verbatim under the runtime root.
_LEGACY_DIRNAMES = ("inbox", "reports")


@dataclass
class MigrationPlan:
    moves: list[tuple[Path, Path]] = field(default_factory=list)  # (src, dst): move src -> dst
    skips: list[tuple[Path, Path]] = field(default_factory=list)  # (src, dst): dst newer -> drop src


def _newer(a: Path, b: Path) -> bool:
    return a.stat().st_mtime > b.stat().st_mtime


def build_plan(root: Path, runtime: Path) -> MigrationPlan:
    """Compute the file-level migration plan without touching the filesystem."""
    plan = MigrationPlan()
    for name in _LEGACY_DIRNAMES:
        src_root = root / name
        dst_root = runtime / name
        if not src_root.is_dir():
            continue
        for src in sorted(src_root.rglob("*")):
            if not src.is_file():
                continue
            dst = dst_root / src.relative_to(src_root)
            if dst.exists() and not _newer(src, dst):
                plan.skips.append((src, dst))  # destination is canonical + newer/equal
            else:
                plan.moves.append((src, dst))
    return plan


def _apply_moves(plan: MigrationPlan) -> None:
    for src, dst in plan.moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()  # os.rename won't overwrite on Windows; remove the older dst first
        shutil.move(str(src), str(dst))
        logger.info("moved %s -> %s", src, dst)
    for src, dst in plan.skips:
        src.unlink()  # newer copy already lives at the canonical destination
        logger.info("dropped stale %s (newer copy at %s)", src, dst)


def _rmdir_with_retry(path: Path, attempts: int = 5, delay: float = 0.2) -> bool:
    """rmdir an empty dir on Windows, clearing the read-only attribute first.

    Two distinct causes make ``rmdir`` raise ``WinError 5`` (PermissionError) on
    an empty directory:
      * the directory carries ``FILE_ATTRIBUTE_READONLY`` (common on dirs synced
        from a CIFS mount via robocopy) — cleared here with ``S_IWRITE``;
      * Defender / the search indexer briefly hold a handle right after files
        move out — handled by the retry/backoff.
    Returns True on success (or if the dir is already gone).
    """
    for i in range(attempts):
        try:
            os.chmod(path, stat.S_IWRITE)  # clear read-only; no-op if already writable
        except OSError:
            pass
        try:
            path.rmdir()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if i == attempts - 1:
                return False
            time.sleep(delay)
    return False


def _prune_empty_legacy_dirs(root: Path) -> list[Path]:
    """Remove empty legacy root dirs (and their now-empty subtrees). Idempotent.

    Best-effort: runs independently of any file move so leftover empty dirs from
    a prior partial run are still cleaned, and a stubborn transient lock on one
    dir never aborts the migration. Subdirs are removed deepest-first (by path
    depth, not lexicographic order) so a parent is only considered once its
    children are gone.
    """
    removed: list[Path] = []
    for name in _LEGACY_DIRNAMES:
        src_root = root / name
        if not src_root.is_dir():
            continue
        for d in sorted(src_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                _rmdir_with_retry(d)
        if not any(src_root.iterdir()) and _rmdir_with_retry(src_root):
            removed.append(src_root)
            logger.info("removed empty legacy dir %s", src_root)
        elif src_root.exists():
            logger.warning(
                "legacy dir %s could not be removed (locked or not empty); re-run --apply to retry",
                src_root,
            )
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy root inbox/ and reports/ into the canonical var/ runtime root."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Perform the move + delete root dirs (default: dry-run)."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    root = repo_root().resolve()
    runtime = runtime_root().resolve()

    # Safety: refuse if the runtime root is the repo root itself, which would map
    # inbox/ -> inbox/ and risk destroying the source while "migrating" it.
    if runtime == root:
        logger.error(
            "Runtime root %s coincides with repo root; nothing to migrate safely. "
            "Unset OMEGA_RUNTIME_DIR or point it away from the repo root.",
            runtime,
        )
        return 1

    plan = build_plan(root, runtime)
    legacy_dirs = [root / name for name in _LEGACY_DIRNAMES if (root / name).is_dir()]

    if not plan.moves and not plan.skips and not legacy_dirs:
        print("No legacy root runtime artifacts under inbox/ or reports/. Nothing to do.")
        return 0

    print(f"Runtime root: {runtime}")
    print(f"Planned moves: {len(plan.moves)}; stale-drops (dest newer): {len(plan.skips)}")
    for src, dst in plan.moves:
        print(f"  MOVE  {src.relative_to(root)} -> {dst}")
    for src, dst in plan.skips:
        print(f"  DROP  {src.relative_to(root)} (newer copy at {dst})")
    if legacy_dirs:
        print(f"  PRUNE empty legacy dirs after move: {[str(p.relative_to(root)) for p in legacy_dirs]}")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to perform the migration.")
        return 0

    try:
        _apply_moves(plan)
    except OSError as exc:
        logger.error("migration failed during file move: %s", exc)
        return 1
    removed = _prune_empty_legacy_dirs(root)  # best-effort; never aborts the move
    remaining = [d for d in legacy_dirs if d.exists()]
    if remaining:
        print(
            "\nFiles migrated. Could not remove now-empty legacy dir(s) "
            f"{[str(p.relative_to(root)) for p in remaining]} (transient Windows lock); "
            "re-run with --apply to retry the cleanup."
        )
    else:
        print(f"\nMigration complete. Removed legacy dirs: {[str(p) for p in removed]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
