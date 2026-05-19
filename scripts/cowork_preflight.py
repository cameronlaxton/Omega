"""Preflight checks for Omega Cowork/local VM sessions.

This script intentionally checks the runtime before an agent attempts MCP or
direct engine execution. A missing dependency is a setup failure, not a reason
to downgrade away from the deterministic engine without first repairing setup.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from collections.abc import Sequence

MIN_PYTHON = (3, 12)


def _version_text(version_info: Sequence[int]) -> str:
    return ".".join(str(part) for part in version_info[:3])


def check_python(version_info: Sequence[int] | None = None) -> list[str]:
    current = tuple((version_info if version_info is not None else sys.version_info)[:3])
    if current[:2] < MIN_PYTHON:
        return [
            "Python 3.12+ is required. "
            f"Current interpreter is {_version_text(current)} at {sys.executable}."
        ]
    return []


def check_distribution() -> list[str]:
    try:
        importlib.metadata.version("omega")
    except importlib.metadata.PackageNotFoundError:
        return [
            "Omega is not installed in this interpreter. "
            "Run: python -m pip install -e .[mcp]"
        ]
    return []


def check_import(module: str, install_hint: str) -> list[str]:
    try:
        importlib.import_module(module)
    except ImportError as exc:
        return [f"Missing import '{module}': {exc}. Run: {install_hint}"]
    return []


def run_checks(*, require_mcp: bool = True) -> list[str]:
    failures: list[str] = []
    failures.extend(check_python())
    if failures:
        return failures

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

    failures = run_checks(require_mcp=not args.direct_only)
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
