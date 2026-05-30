"""
scripts/validate_all.py — one-shot Omega validation harness.

Runs the existing validation checks together and reports a single aggregated
PASS/FAIL with one exit code. This is the orchestrator for Phase 6h exit
criterion #7 ("CI or local validation checks cover tests, sidecars, export
shapes, report metadata, and artifact policy"). It owns no validation logic of
its own — it dispatches the individual validators, exactly as
fetch_outcomes_all.py dispatches the per-sport outcome scripts.

Checks (in order):
  1. tests            -> pytest -q            (skippable; ~2 min)
  2. repo-state       -> validate_repo_state.py      (report metadata + terminology)
  3. artifact-policy  -> validate_artifact_policy.py (no committed runtime artifacts)
  4. session-sidecars -> validate_session_sidecars.py --sessions-inbox <dir>
  5. trace-export     -> validate_trace_export.py <dir>

Path-based checks (3, 4) are SKIPPED — not failed — when their inbox directory
is absent or has no *.json, so a clean checkout with no runtime artifacts still
passes.

Usage:
    python scripts/validate_all.py
    python scripts/validate_all.py --skip-tests        # policy/shape checks only (fast)
    python scripts/validate_all.py --sessions-inbox inbox/sessions --traces inbox/traces

Exit codes:
    0 — every executed check passed
    1 — at least one executed check failed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"

_PASS, _FAIL, _SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class StepResult:
    name: str
    status: str
    detail: str = ""


def _has_json(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.json"))


def _run(name: str, cmd: list[str]) -> StepResult:
    print(f"\n=== {name}: {' '.join(cmd)} ===", flush=True)
    rc = subprocess.run(cmd, cwd=_REPO_ROOT).returncode
    return StepResult(name, _PASS if rc == 0 else _FAIL, f"exit={rc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all Omega validation checks with one verdict.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the (slow) pytest run")
    parser.add_argument(
        "--sessions-inbox",
        type=Path,
        default=_REPO_ROOT / "inbox" / "sessions",
        help="Session sidecar directory (default: inbox/sessions)",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=_REPO_ROOT / "inbox" / "traces",
        help="Exported trace directory (default: inbox/traces)",
    )
    args = parser.parse_args(argv)

    results: list[StepResult] = []

    # 1. Test suite
    if args.skip_tests:
        results.append(StepResult("tests", _SKIP, "--skip-tests"))
    else:
        results.append(_run("tests", [sys.executable, "-m", "pytest", "-q"]))

    # 2. Repo-state policy (report metadata + banned terminology). No paths needed.
    results.append(_run("repo-state", [sys.executable, str(_SCRIPTS / "validate_repo_state.py")]))

    # 3. Artifact-segregation policy (no committed runtime artifacts; no root scratch).
    results.append(
        _run("artifact-policy", [sys.executable, str(_SCRIPTS / "validate_artifact_policy.py")])
    )

    # 3. Session sidecars — skip if the inbox has no sidecars yet.
    if _has_json(args.sessions_inbox):
        results.append(
            _run(
                "session-sidecars",
                [
                    sys.executable,
                    str(_SCRIPTS / "validate_session_sidecars.py"),
                    "--sessions-inbox",
                    str(args.sessions_inbox),
                ],
            )
        )
    else:
        results.append(StepResult("session-sidecars", _SKIP, f"no *.json in {args.sessions_inbox}"))

    # 4. Trace export shapes — skip if there are no exported traces to check.
    if _has_json(args.traces):
        results.append(
            _run("trace-export", [sys.executable, str(_SCRIPTS / "validate_trace_export.py"), str(args.traces)])
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
