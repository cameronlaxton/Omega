"""Preflight checks for Omega Cowork/local VM sessions.

This script intentionally checks the runtime before an agent attempts MCP or
direct engine execution. A missing dependency is a setup failure, not a reason
to downgrade away from the deterministic engine without first repairing setup.
"""

from __future__ import annotations

import argparse
import ast
import collections
import errno
import importlib
import importlib.metadata
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

MIN_PYTHON = (3, 10)
_REPAIR_TAINT_FILENAME = ".repair_taint"
_OMEGA_CACHE_DIRNAME = ".omega_cache"
_PREFLIGHT_SENTINEL = "# EOF: cowork_preflight_completed"


def _get_repair_taint_path(repo_root: Path) -> Path:
    return repo_root / _OMEGA_CACHE_DIRNAME / _REPAIR_TAINT_FILENAME


def _version_text(version_info: Sequence[int]) -> str:
    return ".".join(str(part) for part in version_info[:3])


def check_python(version_info: Sequence[int] | None = None) -> list[str]:
    current = tuple((version_info if version_info is not None else sys.version_info)[:3])
    if current[:2] < MIN_PYTHON:
        return [
            "Python 3.10+ is required. "
            f"Current interpreter is {_version_text(current)} at {sys.executable}."
        ]
    return []


def check_distribution() -> list[str]:
    try:
        importlib.metadata.version("omega")
    except importlib.metadata.PackageNotFoundError:
        return ["Omega is not installed in this interpreter. Run: python -m pip install -e .[mcp]"]
    return []


def check_import(module: str, install_hint: str) -> list[str]:
    try:
        importlib.import_module(module)
    except ImportError as exc:
        return [f"Missing import '{module}': {exc}. Run: {install_hint}"]
    return []


def _index_lock_backup_path(repo_root: Path) -> Path:
    git_dir = repo_root / ".git"
    candidate = git_dir / "index.lock.bak"
    if not candidate.exists():
        return candidate

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    stamped = git_dir / f"index.lock.bak.{stamp}"
    counter = 1
    while stamped.exists():
        stamped = git_dir / f"index.lock.bak.{stamp}.{counter}"
        counter += 1
    return stamped


def _relocate_stale_index_lock(repo_root: Path) -> tuple[Path | None, str | None]:
    """Move a stale git index lock without unlinking it."""
    index_lock = repo_root / ".git" / "index.lock"
    if not index_lock.exists():
        return None, None

    backup_path = _index_lock_backup_path(repo_root)
    try:
        os.rename(str(index_lock), str(backup_path))
    except OSError as exc:
        return None, (
            "GitEnvironmentCorrupt: .git/index.lock exists and FUSE-safe "
            f"rename to {backup_path} failed: {exc}. Close any active git "
            "processes or manually move the lock aside; avoid unlink/delete on "
            "the Cowork FUSE mount."
        )
    return backup_path, None


def check_git_health(repo_root: Path) -> list[str]:
    """Verify git environment is functional and not mid-operation.

    Detects index.lock (another git process is running) and git command failures
    that would make content-equality checks unreliable.
    """
    relocated_lock, relocate_error = _relocate_stale_index_lock(repo_root)
    if relocate_error:
        return [relocate_error]
    if relocated_lock is not None:
        print(f"[repair] Moved stale .git/index.lock to {relocated_lock}")

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        lock_note = (
            f" after moving .git/index.lock to {relocated_lock}"
            if relocated_lock is not None
            else ""
        )
        return [
            f"GitEnvironmentCorrupt: `git status` failed{lock_note} "
            f"(exit {result.returncode}): "
            f"{result.stderr.strip() or '(no stderr)'}. "
            "Resolve git environment issues before running engine checks."
        ]
    return []


def check_omega_import_binding(repo_root: Path) -> list[str]:
    """Ensure the active interpreter imports Omega from this checkout."""
    try:
        omega_mod = importlib.import_module("omega")
    except ImportError as exc:
        return [f"Missing import 'omega': {exc}. Run: python -m pip install -e .[mcp]"]

    omega_file = getattr(omega_mod, "__file__", None)
    if not omega_file:
        return ["Omega import binding check failed: omega.__file__ is missing."]

    resolved_repo = repo_root.resolve()
    resolved_omega = Path(omega_file).resolve()
    if resolved_omega == resolved_repo or resolved_repo in resolved_omega.parents:
        return []
    return [
        "Omega import binding mismatch: active interpreter imports "
        f"{resolved_omega}, expected a module under {resolved_repo}. "
        "Run from the active checkout and reinstall with: python -m pip install -e .[mcp]"
    ]


def clean_stale_bytecode(repo_root: Path) -> list[str]:
    """Remove stale .pyc and __pycache__ to prevent signature mismatches across Python versions."""
    errors: list[str] = []
    unlink_failures: collections.defaultdict[int, int] = collections.defaultdict(int)

    for pyc_file in repo_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
        except OSError as exc:
            unlink_failures[exc.errno] += 1

    pycache_dirs = sorted(
        repo_root.rglob("__pycache__"),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for pycache_dir in pycache_dirs:
        try:
            pycache_dir.rmdir()
        except Exception:
            # Directory removal is best-effort on Windows: import scanners and
            # antivirus can briefly hold handles or recreate pyc files while the
            # preflight process is still healthy. Stale bytecode files above are
            # the actual correctness risk, and pyc unlink errors are reported.
            pass

    for err_num, count in sorted(unlink_failures.items()):
        err_name = errno.errorcode.get(err_num, f"Errno {err_num}")
        errors.append(f"Skipped {count} stale .pyc file(s) ({err_name}); likely host-locked.")

    return errors


def repair_source_integrity(repo_root: Path) -> list[str]:
    """Strip null bytes from .py files and verify AST parse.

    OneDrive mount snapshots can produce files with trailing null bytes (incomplete
    sync) or hard-truncated files. Null stripping is auto-applied; truncated files
    that still fail AST parse after stripping are reported as failures.
    """
    failures: list[str] = []
    repaired: list[str] = []

    for py_file in sorted(repo_root.rglob("*.py")):
        if ".venv" in py_file.parts or "venv" in py_file.parts:
            continue
        try:
            raw = py_file.read_bytes()
        except OSError as e:
            failures.append(f"Cannot read {py_file}: {e}")
            continue

        if b"\x00" in raw:
            cleaned = raw.replace(b"\x00", b"")
            try:
                py_file.write_bytes(cleaned)
                repaired.append(str(py_file))
            except OSError as e:
                failures.append(f"Cannot repair null bytes in {py_file}: {e}")
                continue

        source = py_file.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(source, filename=str(py_file))
        except SyntaxError as e:
            failures.append(
                f"Source corrupt (AST parse failed) {py_file}:{e.lineno} — {e.msg}. "
                "File may be hard-truncated; manual repair needed."
            )

    if repaired:
        print(f"[repair] Stripped null bytes from {len(repaired)} file(s):")
        for path in repaired:
            print(f"  {path}")

    return failures


def _git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


_CRITICAL_FILES = frozenset({
    "omega/core/contracts/service.py",  # Service layer; repeated corruption across sessions
})


# Tiered repair targets (BUG-PREFLIGHT-3): test-tier divergence must never
# block engine-source repair. Anything under tests/ is the test tier; every
# other tracked .py file is the core tier (engine code, scripts, MCP).
def _is_test_tier(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    return normalized.startswith("tests/") or normalized == "tests"


def _split_tiers(rel_paths: list[str]) -> tuple[list[str], list[str]]:
    """Return (core, test) partition of the given tracked-path list."""
    core: list[str] = []
    test: list[str] = []
    for rel in rel_paths:
        (test if _is_test_tier(rel) else core).append(rel)
    return core, test


def _check_critical_files(repo_root: Path, diverged: list[str]) -> list[str]:
    """Flag critical files separately with stricter warnings.

    These files have demonstrated repeated mount corruption and merit explicit
    attention before any Omega execution proceeds.
    """
    failures: list[str] = []
    critical_diverged = [p for p in diverged if p in _CRITICAL_FILES]
    if critical_diverged:
        for path in critical_diverged:
            failures.append(
                f"CRITICAL: {path} diverges from git HEAD (mount corruption risk). "
                "Re-run preflight with --repair-from-git before engine execution."
            )
    return failures


def _tracked_python_files(repo_root: Path) -> tuple[list[str], list[str]]:
    """Return tracked Python files from HEAD, plus any git inspection failures."""
    head_check = _git(repo_root, ["rev-parse", "--verify", "HEAD"])
    if head_check.returncode != 0:
        return [], [
            "verify_against_git: no HEAD commit available "
            "(detached/empty repo); skipping git-parity check."
        ]

    ls_files = _git(repo_root, ["ls-tree", "-r", "-z", "--name-only", "HEAD"])
    if ls_files.returncode != 0:
        return [], [f"verify_against_git: `git ls-tree` failed: {ls_files.stderr.strip()}"]

    return [
        rel_path
        for rel_path in ls_files.stdout.split("\x00")
        if rel_path and rel_path.endswith(".py")
    ], []


def _diverged_tracked_files(repo_root: Path, tracked_files: list[str]) -> list[str]:
    diverged: list[str] = []
    for rel_path in tracked_files:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            continue
        index_hash = _git(repo_root, ["rev-parse", f"HEAD:{rel_path}"]).stdout.strip()
        if not index_hash:
            continue
        wt_hash = _git(repo_root, ["hash-object", "--", str(abs_path)]).stdout.strip()
        if wt_hash and wt_hash != index_hash:
            diverged.append(rel_path)
    return diverged


def _syntax_corrupt_tracked_files(repo_root: Path, tracked_files: list[str]) -> list[str]:
    corrupt: list[str] = []
    for rel_path in tracked_files:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            continue
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
            ast.parse(source, filename=str(abs_path))
        except (OSError, SyntaxError):
            corrupt.append(rel_path)
    return corrupt


def _sync_filesystem() -> None:
    sync = getattr(os, "sync", None)
    if callable(sync):
        sync()


def _ast_parse_file(path: Path) -> str | None:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        return str(exc)
    return None


def verify_against_git(
    repo_root: Path, *, repair: bool = False, force_repair: bool = False
) -> list[str]:
    """Detect silent semantic truncation (Pattern C) by comparing tracked .py files to git HEAD.

    The sandbox mount has been observed to deliver source files that are
    syntactically valid but missing trailing function/class definitions. AST
    parsing cannot catch these; only content equality with the git blob does.

    Tiering (BUG-PREFLIGHT-3): divergence is split into a core tier (engine,
    scripts, MCP) and a test tier (anything under ``tests/``). ``--repair-from-git``
    is allowed to proceed when only test-tier files are dirty outside the
    repair set; test-tier divergence on its own is a warning, not a failure.
    Core-tier divergence still blocks unless every dirty core file is a repair
    target or ``--force-repair`` is supplied.

    In --repair-from-git mode, syntax-corrupt tracked files and known critical
    corruption targets are restored via ``git checkout HEAD --``, which writes
    through the mount cache (Write-tool writes do not — see
    docs/session_bugs_20260521_mount_corruption.md).

    Caution: --force-repair clobbers every divergent tracked Python file across
    both tiers.
    """
    failures: list[str] = []

    tracked_files, git_failures = _tracked_python_files(repo_root)
    if git_failures:
        return git_failures

    diverged = _diverged_tracked_files(repo_root, tracked_files)

    if not diverged:
        return []

    core_diverged, test_diverged = _split_tiers(diverged)

    if repair:
        corrupt = set(_syntax_corrupt_tracked_files(repo_root, tracked_files))
        critical = set(_CRITICAL_FILES).intersection(diverged)
        if force_repair:
            repair_targets = set(diverged)
        else:
            # Default repair set: every corrupt or critical file in EITHER tier
            # gets restored. Test-tier corruption is restored too — that's
            # still a real corruption event we want to heal — but a *clean*
            # core tree with dirty test edits no longer blocks repair.
            repair_targets = corrupt.union(critical)

        # The blocking guard now only fires when CORE files are dirty without
        # being repair targets. Test-tier divergence outside the repair set
        # is intentionally non-blocking (BUG-PREFLIGHT-3): the engine is fine,
        # only the test fixtures have local edits.
        non_target_core_dirty = sorted(set(core_diverged) - repair_targets)
        if non_target_core_dirty and not force_repair:
            return [
                "verify_against_git: Refusing to --repair-from-git because tracked "
                "core Python files outside repair targets are dirty. Uncommitted "
                f"core files: {', '.join(non_target_core_dirty)}. "
                "Use --force-repair to clobber."
            ]
        if not repair_targets:
            return [
                "verify_against_git: No repair targets found. "
                "Use --force-repair to restore all divergent tracked Python files."
            ]

        restored: list[str] = []
        for rel_path in sorted(repair_targets):
            result = _git(repo_root, ["checkout", "HEAD", "--", rel_path])
            if result.returncode == 0:
                restored.append(rel_path)
            else:
                failures.append(
                    f"verify_against_git: failed to restore {rel_path}: "
                    f"{result.stderr.strip()}"
                )
        if restored:
            _sync_filesystem()
            print(f"[repair] Restored {len(restored)} file(s) from git HEAD:")
            for path in restored:
                print(f"  {path}")
            print("[repair] Re-parsing restored file(s):")
            for path in restored:
                parse_error = _ast_parse_file(repo_root / path)
                if parse_error is None:
                    print(f"  OK {path}")
                else:
                    failures.append(f"verify_against_git: AST parse failed after repair {path}: {parse_error}")
            taint_path = _get_repair_taint_path(repo_root)
            try:
                taint_path.parent.mkdir(parents=True, exist_ok=True)
                taint_path.write_text("repair_tainted\n", encoding="utf-8")
                print(f"[repair] Taint lockfile written: {taint_path}")
            except OSError as exc:
                failures.append(f"[repair] Could not write repair taint lockfile: {exc}")
        remaining_diverged = _diverged_tracked_files(repo_root, tracked_files)
        failures.extend(_check_critical_files(repo_root, remaining_diverged))
        # Full parse sweep of all tracked Python files to catch any remaining
        # truncation after repair (BUG-PREFLIGHT-4: partial repair blind spots).
        if not failures:
            post_repair_corrupt = _syntax_corrupt_tracked_files(repo_root, tracked_files)
            for rel_path in post_repair_corrupt:
                failures.append(
                    f"Source corrupt after repair (AST parse failed): {rel_path}. "
                    "Manual restoration may be required."
                )
        return failures

    # Non-repair report path. Tier the failures.
    # Critical-file divergences (highest Pattern C risk) are hard failures.
    # Non-critical core divergences are surfaced as warnings only — they are
    # most commonly intentional edits. The operator may run --repair-from-git
    # to investigate further; blocking here prevents legitimate engine use.
    critical_failures = _check_critical_files(repo_root, core_diverged)
    if critical_failures:
        failures.extend(critical_failures)

    for rel_path in core_diverged:
        if rel_path not in _CRITICAL_FILES:
            print(
                f"[warning] verify_against_git: {rel_path} diverges from git HEAD. "
                "If this is mount corruption (truncation), re-run with --repair-from-git. "
                "If this is your intentional edit, this is informational only."
            )

    # Test-tier divergence is a warning, never a failure. Surface it on stdout
    # so the operator notices but does not block engine execution.
    if test_diverged:
        print(
            f"[warning] verify_against_git: {len(test_diverged)} test file(s) "
            "diverge from git HEAD (not blocking engine execution): "
            + ", ".join(test_diverged)
        )

    return failures


def run_checks(
    *,
    require_mcp: bool = True,
    repo_root: Path | None = None,
    repair_from_git: bool = False,
    force_repair: bool = False,
) -> list[str]:
    failures: list[str] = []
    failures.extend(check_python())
    if failures:
        return failures

    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    # Fail fast on git environment problems before any content equality checks.
    failures.extend(check_git_health(repo_root))
    if failures:
        return failures

    if repair_from_git:
        # Restore known-corrupt tracked files before AST parsing so hard
        # truncations do not block the git-based repair path.
        git_failures = verify_against_git(
            repo_root, repair=True, force_repair=force_repair
        )
        if git_failures:
            failures.extend(git_failures)
            return failures

    # Source integrity must run before any import attempt. Null bytes or
    # hard-truncated files produce cryptic ImportError/SyntaxError otherwise.
    integrity_failures = repair_source_integrity(repo_root)
    if integrity_failures:
        failures.extend(integrity_failures)
        failures.append(
            "Source integrity check failed. Fix the files above before running "
            "the engine. Do not emit formal Omega numeric outputs."
        )
        return failures

    if not repair_from_git:
        # Pattern C: silent trailing truncation that leaves the file AST-valid.
        # Only git content equality catches it.
        git_failures = verify_against_git(repo_root)
        if git_failures:
            failures.extend(git_failures)
            return failures

    # Clean stale bytecode after source is verified clean (BUG-2026-05-20-003).
    clean_errors = clean_stale_bytecode(repo_root)
    if clean_errors:
        for error in clean_errors:
            print(f"[warning] {error}")

    # Parse critical runtime files before import (catches silent content-valid truncation
    # that AST-parity and null-byte checks may miss on files not yet diverged from HEAD).
    for critical_rel in sorted(_CRITICAL_FILES):
        crit_path = repo_root / critical_rel
        if crit_path.exists():
            err = _ast_parse_file(crit_path)
            if err is not None:
                failures.append(
                    f"Critical runtime file AST parse failed: {critical_rel}: {err}"
                )
    if failures:
        return failures

    failures.extend(check_distribution())
    failures.extend(check_omega_import_binding(repo_root))
    failures.extend(check_import("pydantic", "python -m pip install -e .[mcp]"))
    failures.extend(check_import("numpy", "python -m pip install -e .[mcp]"))
    failures.extend(
        check_import(
            "omega.core.contracts.service",
            "python -m pip install -e .[mcp]",
        )
    )
    if require_mcp:
        failures.extend(
            check_import(
                "mcp.server.fastmcp",
                "python -m pip install -e .[mcp]",
            )
        )
    return failures


def check_formal_output_smoke() -> list[str]:
    """Run a deterministic direct-engine smoke for formal output readiness."""
    try:
        from omega.core.contracts.service import analyze

        trace = analyze(
            {
                "home_team": "Boston Celtics",
                "away_team": "Indiana Pacers",
                "league": "NBA",
                "n_iterations": 100,
                "seed": 20260528,
                "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
                "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
                "game_context": {"is_playoff": False, "rest_days": 2},
                "odds": {"moneyline_home": -150, "moneyline_away": 130},
            },
            session_id="sess-preflight-smoke",
            bankroll=1000.0,
        )
    except Exception as exc:  # noqa: BLE001
        return [f"Formal output smoke failed: {exc}"]

    result = trace.get("result") or {}
    if result.get("status") != "success":
        return [f"Formal output smoke returned status={result.get('status')!r}"]
    if not trace.get("trace_id", "").startswith("sandbox-"):
        return ["Formal output smoke did not return an engine-minted sandbox trace_id"]
    if not result.get("edges"):
        return ["Formal output smoke returned no deterministic edge rows"]
    return []


def run_formal_output_gate(
    *,
    require_mcp: bool = True,
    repo_root: Path | None = None,
    repair_from_git: bool = False,
    force_repair: bool = False,
) -> list[str]:
    """Hard gate for betting-grade output: clean preflight + engine smoke."""
    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    failures = run_checks(
        require_mcp=require_mcp,
        repo_root=repo_root,
        repair_from_git=repair_from_git,
        force_repair=force_repair,
    )
    if failures:
        return failures

    tracked_files, git_failures = _tracked_python_files(repo_root)
    if git_failures:
        return git_failures
    diverged = _diverged_tracked_files(repo_root, tracked_files)
    if diverged:
        return [
            "Formal output gate requires clean source parity with git HEAD. "
            "Run non-formal research only or repair/commit intentional source "
            f"changes first. Diverged tracked Python files: {', '.join(diverged)}"
        ]

    smoke_failures = check_formal_output_smoke()
    if smoke_failures:
        return smoke_failures

    # Source is verified clean. If a repair-taint lockfile is present from a prior
    # repair run, clear it now and require one additional clean invocation to confirm
    # the gate is open. This prevents formal output from being emitted in the same
    # session as a repair without an independent confirmation pass.
    taint_path = _get_repair_taint_path(repo_root)
    if taint_path.exists():
        try:
            taint_path.unlink()
        except OSError:
            pass
        return [
            "TAINT_CLEARED: Source was repaired in a prior run and has now been verified "
            "clean. Re-run preflight --formal-output-gate to confirm and open the gate."
        ]

    return []


def _run_bug_sentinel(repo_root: Path) -> None:
    """Run the known-bug sentinel as an informational preflight step.

    Failures are printed but never block preflight; sentinel issues are
    advisory. Call before run_checks() so gate status is visible first.
    """
    sentinel = repo_root / "scripts" / "bug_sentinel.py"
    if not sentinel.exists():
        print("[sentinel] scripts/bug_sentinel.py not found — skipping bug sentinel")
        return
    result = subprocess.run(
        [sys.executable, str(sentinel)],
        cwd=repo_root,
        check=False,
        capture_output=False,
    )
    if result.returncode not in (0, 1):
        print(f"[sentinel] bug_sentinel.py exited with code {result.returncode} — continuing preflight")



def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Omega Cowork runtime dependencies before engine execution.",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Skip the optional MCP SDK check for direct analyze() smoke tests.",
    )
    parser.add_argument(
        "--repair-from-git",
        action="store_true",
        help=(
            "Restore syntax-corrupt tracked Python files and known critical "
            "corruption targets via `git checkout`. Used to recover from "
            "silent mount truncation."
        ),
    )
    parser.add_argument(
        "--force-repair",
        action="store_true",
        help="Allow --repair-from-git to clobber all divergent tracked Python files.",
    )
    parser.add_argument(
        "--skip-bug-sentinel",
        action="store_true",
        help="Skip the known-bug sentinel advisory check (use in CI or automated loops).",
    )
    parser.add_argument(
        "--formal-output-gate",
        action="store_true",
        help="Require clean preflight plus deterministic smoke before Bet Cards.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).parent.parent

    # Bug sentinel: run before preflight checks so known issues are visible
    # even if the checks pass. Skippable for speed in CI or automated loops.
    if not getattr(args, "skip_bug_sentinel", False):
        _run_bug_sentinel(repo_root)

    if args.formal_output_gate:
        failures = run_formal_output_gate(
            require_mcp=not args.direct_only,
            repo_root=repo_root,
            repair_from_git=args.repair_from_git,
            force_repair=args.force_repair,
        )
    else:
        failures = run_checks(
            require_mcp=not args.direct_only,
            repo_root=repo_root,
            repair_from_git=args.repair_from_git,
            force_repair=args.force_repair,
        )
    if failures:
        print("cowork_preflight_failed:")
        for failure in failures:
            print(f"- {failure}")
        print("- Do not emit formal Omega numeric outputs until preflight passes.")
        return 1

    # Banner: if any tracked files diverge from HEAD after all checks passed,
    # the divergences are either non-critical core edits (warned above, not
    # blocked) or test-tier edits.  Both are safe for engine execution —
    # critical-file corruption and import failures were already blocked above.
    # Emit cowork_preflight_core_ready so automated runs can distinguish this
    # state from a hard failure.
    tracked_files, git_failures = _tracked_python_files(repo_root)
    if not git_failures:
        diverged = _diverged_tracked_files(repo_root, tracked_files)
        core_diverged, test_diverged = _split_tiers(diverged)
        if core_diverged or test_diverged:
            dirty_parts = []
            if core_diverged:
                dirty_parts.append(f"core_dirty={len(core_diverged)}")
            if test_diverged:
                dirty_parts.append(f"test_tier_diverged={len(test_diverged)}")
            print(
                "cowork_preflight_core_ready: python="
                f"{_version_text(tuple(sys.version_info[:3]))} "
                f"({'  '.join(dirty_parts)})"
            )
            return 0

    print(f"cowork_preflight_ready: python={_version_text(tuple(sys.version_info[:3]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# EOF: cowork_preflight_completed
