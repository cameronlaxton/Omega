"""
scripts/run_action_plan.py — dispatch the LLM's action-plan JSON to deterministic scripts.

The LLM, at session start, emits an action plan per system_prompt.txt §13. This
script reads that JSON, enforces a strict allowlist of action types and arg
schemas, and dispatches to the relevant Phase 6 scripts via subprocess.

The LLM is forbidden from emitting calibration parameters, Brier targets, or
shrink factors. This runner enforces that boundary: only high-level orchestration
arguments are accepted. Anything else fails closed.

Allowlist:
    type=fit_calibration
        args.league: str (required)
        args.method: "isotonic" | "shrinkage" | "both" (default "both")
        args.min_samples: int (default 100)
    type=promote_profile
        args.candidate_id: str (required)
        args.auto: bool (default false)
    type=report_calibration
        args.league: str (required)
        args.window_days: int (default 30)
    type=fetch_outcomes
        args.leagues: list[str] (default ["nba", "mlb", "props"])
        args.since: str YYYY-MM-DD (optional)
        args.until: str YYYY-MM-DD (optional)

Usage:
    python scripts/run_action_plan.py inbox/action_plans/<session_id>.json
    python scripts/run_action_plan.py <file> --dry-run

Exit codes:
    0 — all actions dispatched successfully (or dry-run completed)
    1 — at least one action failed
    2 — fatal validation error (file missing, unknown action type, bad args)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("run_action_plan")


def _validate_fit_calibration(args: dict[str, Any]) -> list[str]:
    if "league" not in args or not isinstance(args["league"], str):
        raise ValueError("fit_calibration.args.league is required and must be str")
    method = args.get("method", "both")
    if method not in ("isotonic", "shrinkage", "both"):
        raise ValueError(f"fit_calibration.args.method invalid: {method!r}")
    min_samples = args.get("min_samples", 100)
    if not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("fit_calibration.args.min_samples must be a positive int")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "fit_calibration.py"),
        "--league",
        args["league"],
        "--method",
        method,
        "--min-samples",
        str(min_samples),
    ]
    return cmd


def _validate_promote_profile(args: dict[str, Any]) -> list[str]:
    if "candidate_id" not in args or not isinstance(args["candidate_id"], str):
        raise ValueError("promote_profile.args.candidate_id is required and must be str")
    auto = args.get("auto", False)
    if not isinstance(auto, bool):
        raise ValueError("promote_profile.args.auto must be bool")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "promote_profile.py"),
        "--candidate-id",
        args["candidate_id"],
    ]
    if auto:
        cmd.append("--auto")
    return cmd


def _validate_report_calibration(args: dict[str, Any]) -> list[str]:
    if "league" not in args or not isinstance(args["league"], str):
        raise ValueError("report_calibration.args.league is required and must be str")
    window_days = args.get("window_days", 30)
    if not isinstance(window_days, int) or window_days < 1:
        raise ValueError("report_calibration.args.window_days must be a positive int")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "report_calibration.py"),
        "--league",
        args["league"],
        "--window-days",
        str(window_days),
    ]
    return cmd


def _validate_fetch_outcomes(args: dict[str, Any]) -> list[str]:
    _VALID_LEAGUES = {"nba", "mlb", "props"}
    leagues = args.get("leagues", ["nba", "mlb", "props"])
    if not isinstance(leagues, list) or not all(isinstance(league, str) for league in leagues):
        raise ValueError("fetch_outcomes.args.leagues must be a list of strings")
    unknown = set(leagues) - _VALID_LEAGUES
    if unknown:
        raise ValueError(f"fetch_outcomes.args.leagues contains unknown leagues: {unknown}")
    since = args.get("since")
    until = args.get("until")
    if since is not None and not isinstance(since, str):
        raise ValueError("fetch_outcomes.args.since must be a date string YYYY-MM-DD")
    if until is not None and not isinstance(until, str):
        raise ValueError("fetch_outcomes.args.until must be a date string YYYY-MM-DD")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "fetch_outcomes_all.py"),
        "--leagues",
        *leagues,
    ]
    if since:
        cmd += ["--since", since]
    if until:
        cmd += ["--until", until]
    return cmd


# Strict allowlist. Adding a key here is a deliberate boundary change.
_DISPATCH: dict[str, Any] = {
    "fit_calibration": _validate_fit_calibration,
    "promote_profile": _validate_promote_profile,
    "report_calibration": _validate_report_calibration,
    "fetch_outcomes": _validate_fetch_outcomes,
}


def _load_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        plan = json.load(fh)
    if not isinstance(plan, dict):
        raise ValueError("Top-level JSON must be an object")
    if "actions" not in plan or not isinstance(plan["actions"], list):
        raise ValueError("plan.actions must be an array (may be empty)")
    return plan


def _validate_all(plan: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Validate every action up-front and return (type, cmd) pairs. Raises on bad input."""
    out: list[tuple[str, list[str]]] = []
    for i, action in enumerate(plan["actions"]):
        if not isinstance(action, dict):
            raise ValueError(f"actions[{i}] must be an object")
        atype = action.get("type")
        if atype not in _DISPATCH:
            raise ValueError(f"actions[{i}].type={atype!r} not in allowlist {sorted(_DISPATCH)}")
        args = action.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"actions[{i}].args must be an object")
        try:
            cmd = _DISPATCH[atype](args)
        except ValueError as exc:
            raise ValueError(f"actions[{i}] ({atype}): {exc}") from exc
        out.append((atype, cmd))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch an action-plan JSON file.")
    parser.add_argument("plan", type=Path, help="Path to the action-plan JSON file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print commands but do not execute"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.plan.exists():
        logger.error("Action plan file not found: %s", args.plan)
        return 2

    try:
        plan = _load_plan(args.plan)
        validated = _validate_all(plan)
    except ValueError as exc:
        logger.error("Plan validation failed: %s", exc)
        return 2

    session_id = plan.get("session_id", "?")
    if not validated:
        logger.info("Plan session=%s contains no actions; nothing to do.", session_id)
        return 0

    logger.info("Plan session=%s: %d action(s) validated.", session_id, len(validated))
    if args.dry_run:
        for atype, cmd in validated:
            logger.info("DRY-RUN %s: %s", atype, " ".join(cmd))
        return 0

    failures = 0
    for atype, cmd in validated:
        logger.info("Running %s: %s", atype, " ".join(cmd))
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        if result.returncode != 0:
            failures += 1
            logger.error("%s FAILED (exit=%d)", atype, result.returncode)
        else:
            logger.info("%s OK", atype)

    if failures:
        logger.error("Action plan completed with %d failure(s).", failures)
        return 1
    logger.info("Action plan completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
