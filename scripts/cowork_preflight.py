"""Preflight checks for Omega Cowork/local VM sessions.

This script intentionally checks the runtime before an agent attempts MCP or
direct engine execution. A missing dependency is a setup failure, not a reason
to downgrade away from the deterministic engine without first repairing setup.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

MIN_PYTHON = (3, 10)


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


def clean_stale_bytecode(repo_root: Path) -> list[str]:
    """Remove stale .pyc and __pycache__ to prevent signature mismatches across Python versions."""
    errors: list[str] = []

    for pycache_dir in repo_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
        except Exception as e:
            errors.append(f"Failed to remove {pycache_dir}: {e}")

    for pyc_file in repo_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
        except Exception as e:
            errors.append(f"Failed to remove {pyc_file}: {e}")

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


def run_checks(*, require_mcp: bool = True, repo_root: Path | None = None) -> list[str]:
    failures: list[str] = []
    failures.extend(check_python())
    if failures:
        return failures

    if repo_root is None:
        repo_root = Path(__file__).parent.parent

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

    # Clean stale bytecode after source is verified clean (BUG-2026-05-20-003).
    clean_errors = clean_stale_bytecode(repo_root)
    if clean_errors:
        for error in clean_errors:
            print(f"[warning] {error}")

    failures.extend(check_distribution())
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Omega Cowork runtime dependencies before engine execution.",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Skip the optional MCP SDK check for direct analyze() smoke tests.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).parent.parent
    failures = run_checks(require_mcp=not args.direct_only, repo_root=repo_root)
    if failures:
        print("cowork_preflight_failed:")
        for failure in failures:
            print(f"- {failure}")
        print("- Do not emit formal Omega numeric outputs until preflight passes.")
        return 1

    print(f"cowork_preflight_ready: python={_version_text(tuple(sys.version_info[:3]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
