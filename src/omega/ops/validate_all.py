"""
omega.ops.validate_all â€” one-shot Omega validation harness.

Runs the existing validation checks together and reports a single aggregated
PASS/FAIL with one exit code. This is the orchestrator for Phase 6h exit
criterion #7 ("CI or local validation checks cover tests, sidecars, export
shapes, report metadata, and artifact policy"). It owns no validation logic of
its own â€” it dispatches the individual validators, exactly as
fetch_outcomes_all.py dispatches the per-sport outcome scripts.

Checks (in order):
  1. tests            -> pytest -q            (skippable; ~2 min)
  2. repo-state       -> validate_repo_state.py      (report metadata + terminology)
  3. artifact-policy  -> validate_artifact_policy.py (no committed runtime artifacts)
  4. session-sidecars -> validate_session_sidecars.py --sessions-inbox <dir>
  5. trace-export     -> validate_trace_export.py <dir>

Path-based checks (3, 4) are SKIPPED â€” not failed â€” when their inbox directory
is absent or has no *.json, so a clean checkout with no runtime artifacts still
passes.

Usage:
    omega-validate-all
    omega-validate-all --skip-tests        # policy/shape checks only (fast)
    omega-validate-all --sessions-inbox var/inbox/sessions --traces var/inbox/traces

Exit codes:
    0 â€” every executed check passed
    1 â€” at least one executed check failed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from omega.paths import repo_root, session_inbox_dir, trace_inbox_dir

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Canonical runtime inboxes live under ``var/`` (docs/phase6/ARTIFACT_AUTHORITY.md).
# A default missing the ``var/`` segment points at a stale/empty directory, so the
# path-based checks below SKIP (or validate the wrong files) and the harness reports
# a false green while the live sidecars/exports go unchecked.
_DEFAULT_SESSIONS_INBOX = session_inbox_dir()
_DEFAULT_TRACES = trace_inbox_dir()

_PASS, _FAIL, _SKIP, _WARN = "PASS", "FAIL", "SKIP", "WARN"


@dataclass
class StepResult:
    name: str
    status: str
    detail: str = ""


def _has_json(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.json"))


def _legacy_runtime_dirs() -> list[Path]:
    root = repo_root()
    return [path for path in (root / "inbox", root / "reports") if path.exists()]


def _run(name: str, cmd: list[str]) -> StepResult:
    print(f"\n=== {name}: {' '.join(cmd)} ===", flush=True)
    rc = subprocess.run(cmd, cwd=_REPO_ROOT).returncode
    return StepResult(name, _PASS if rc == 0 else _FAIL, f"exit={rc}")


def _run_module(name: str, module: str, *args: str) -> StepResult:
    return _run(name, [sys.executable, "-m", module, *args])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all Omega validation checks with one verdict.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the (slow) pytest run")
    parser.add_argument(
        "--sessions-inbox",
        type=Path,
        default=_DEFAULT_SESSIONS_INBOX,
        help="Session sidecar directory (default: var/inbox/sessions)",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=_DEFAULT_TRACES,
        help="Exported trace directory (default: var/inbox/traces)",
    )
    args = parser.parse_args(argv)

    results: list[StepResult] = []

    legacy_dirs = _legacy_runtime_dirs()
    if legacy_dirs:
        detail = (
            "legacy root runtime dir(s) exist; repo root is workspace and var/ is runtime: "
            + ", ".join(str(path) for path in legacy_dirs)
        )
        results.append(StepResult("legacy-runtime-dirs", _WARN, detail))

    # 1. Test suite
    if args.skip_tests:
        results.append(StepResult("tests", _SKIP, "--skip-tests"))
    else:
        results.append(_run("tests", [sys.executable, "-m", "pytest", "-q"]))

    # 2. Repo-state policy (report metadata + banned terminology). No paths needed.
    results.append(_run_module("repo-state", "omega.ops.validate_repo_state"))

    # 3. Artifact-segregation policy (no committed runtime artifacts; no root scratch).
    results.append(_run_module("artifact-policy", "omega.ops.validate_artifact_policy"))

    # 4. Documentation lint (reference files exist, links resolve, no duplicate definitions).
    results.append(_run_module("doc-lint", "omega.ops.validate_docs"))

    # 3. Session sidecars â€” skip if the inbox has no sidecars yet.
    if _has_json(args.sessions_inbox):
        results.append(
            _run_module(
                "session-sidecars",
                "omega.ops.validate_session_sidecars",
                "--sessions-inbox",
                str(args.sessions_inbox),
            )
        )
    else:
        results.append(StepResult("session-sidecars", _SKIP, f"no *.json in {args.sessions_inbox}"))

    # 4. Trace export shapes â€” skip if there are no exported traces to check.
    if _has_json(args.traces):
        results.append(
            _run_module("trace-export", "omega.ops.validate_trace_export", str(args.traces))
        )
    else:
        results.append(StepResult("trace-export", _SKIP, f"no *.json in {args.traces}"))

    # Summary
    print("\n" + "=" * 56)
    print("VALIDATION HARNESS SUMMARY")
    print("=" * 56)
    for r in results:
        print(f"  {r.status:4}  {r.name:<18} {r.detail}")
    print("=" * 56)

    failed = [r.name for r in results if r.status == _FAIL]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("SUCCESS: all executed checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

