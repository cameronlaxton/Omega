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
        args.plane: "game" | "prop" (default "game")
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
    type=ingest_traces
        args.verbose: bool (default false)
    type=fetch_closing_lines
        args.league: str (optional; must be mapped in omega.integrations.odds_api)
        args.verbose: bool (default false)
    type=score_evidence_signals
        args.league: str (optional)
        args.window_days: int (optional)
        args.verbose: bool (default false)
    type=fit_adjustment_policy
        args.league: str (required)
        args.mode: "shadow" (default "shadow"; live is manual-only)
        args.min_samples: int (default 30)
        args.verbose: bool (default false)
    type=render_audit
        args.session_ids: list[str]   (optional; explicit list to render)
        args.all_open: bool           (optional; render every sidecar in inbox/sessions/)
        args.verbose: bool (default false)
        Exactly one of session_ids or all_open must be provided.

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
from datetime import date
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("run_action_plan")


def _reject_unknown_args(action: str, args: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ValueError(f"{action}.args contains unknown keys: {unknown}")


def _require_nonempty_str(action: str, args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{action}.args.{key} is required and must be non-empty str")
    return value


def _optional_nonempty_str(action: str, args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{action}.args.{key} must be non-empty str")
    return value


def _optional_bool(action: str, args: dict[str, Any], key: str, default: bool = False) -> bool:
    value = args.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{action}.args.{key} must be bool")
    return value


def _positive_int(action: str, args: dict[str, Any], key: str, default: int) -> int:
    value = args.get(key, default)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{action}.args.{key} must be a positive int")
    return value


def _optional_date(action: str, args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{action}.args.{key} must be a date string YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{action}.args.{key} must be a date string YYYY-MM-DD") from exc
    return value


def _validate_fit_calibration(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args(
        "fit_calibration",
        args,
        {"league", "plane", "method", "min_samples"},
    )
    league = _require_nonempty_str("fit_calibration", args, "league")
    plane = args.get("plane", "game")
    if plane not in ("game", "prop"):
        raise ValueError(f"fit_calibration.args.plane invalid: {plane!r}")
    method = args.get("method", "both")
    if method not in ("isotonic", "shrinkage", "both"):
        raise ValueError(f"fit_calibration.args.method invalid: {method!r}")
    min_samples = _positive_int("fit_calibration", args, "min_samples", 100)

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "fit_calibration.py"),
        "--league",
        league,
        "--plane",
        plane,
        "--method",
        method,
        "--min-samples",
        str(min_samples),
    ]
    return cmd


def _validate_promote_profile(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args("promote_profile", args, {"candidate_id", "auto"})
    candidate_id = _require_nonempty_str("promote_profile", args, "candidate_id")
    auto = _optional_bool("promote_profile", args, "auto")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "promote_profile.py"),
        "--candidate-id",
        candidate_id,
    ]
    if auto:
        cmd.append("--auto")
    return cmd


def _validate_report_calibration(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args("report_calibration", args, {"league", "window_days"})
    league = _require_nonempty_str("report_calibration", args, "league")
    window_days = _positive_int("report_calibration", args, "window_days", 30)

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "report_calibration.py"),
        "--league",
        league,
        "--window-days",
        str(window_days),
    ]
    return cmd


def _validate_fetch_outcomes(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args("fetch_outcomes", args, {"leagues", "since", "until"})
    _VALID_LEAGUES = {"nba", "wnba", "mlb", "props"}
    leagues = args.get("leagues", ["nba", "mlb", "props"])
    if not isinstance(leagues, list) or not all(isinstance(league, str) for league in leagues):
        raise ValueError("fetch_outcomes.args.leagues must be a list of strings")
    leagues = [league.lower() for league in leagues]
    unknown = set(leagues) - _VALID_LEAGUES
    if unknown:
        raise ValueError(f"fetch_outcomes.args.leagues contains unknown leagues: {unknown}")
    since = _optional_date("fetch_outcomes", args, "since")
    until = _optional_date("fetch_outcomes", args, "until")

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


def _validate_ingest_traces(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args("ingest_traces", args, {"verbose"})
    verbose = _optional_bool("ingest_traces", args, "verbose")

    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "ingest_traces.py")]
    if verbose:
        cmd.append("--verbose")
    return cmd


def _validate_fetch_closing_lines(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args("fetch_closing_lines", args, {"league", "verbose"})
    league = _optional_nonempty_str("fetch_closing_lines", args, "league")
    verbose = _optional_bool("fetch_closing_lines", args, "verbose")

    if league:
        from omega.integrations.odds_api import SPORT_KEY_MAP  # noqa: PLC0415

        league = league.upper()
        if league not in SPORT_KEY_MAP:
            raise ValueError(
                f"fetch_closing_lines.args.league={league!r} has no Odds API sport mapping"
            )

    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "fetch_closing_lines.py")]
    if league:
        cmd += ["--league", league]
    if verbose:
        cmd.append("--verbose")
    return cmd


def _validate_score_evidence_signals(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args("score_evidence_signals", args, {"league", "window_days", "verbose"})
    league = _optional_nonempty_str("score_evidence_signals", args, "league")
    window_days = args.get("window_days")
    if window_days is not None and (not isinstance(window_days, int) or window_days < 1):
        raise ValueError("score_evidence_signals.args.window_days must be a positive int")
    verbose = _optional_bool("score_evidence_signals", args, "verbose")

    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "score_evidence_signals.py")]
    if league:
        cmd += ["--league", league.upper()]
    if window_days is not None:
        cmd += ["--window-days", str(window_days)]
    if verbose:
        cmd.append("--verbose")
    return cmd


def _validate_fit_adjustment_policy(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args(
        "fit_adjustment_policy",
        args,
        {"league", "mode", "min_samples", "verbose"},
    )
    league = _require_nonempty_str("fit_adjustment_policy", args, "league")
    mode = args.get("mode", "shadow")
    if mode != "shadow":
        raise ValueError("fit_adjustment_policy.args.mode must be 'shadow'; live is manual-only")
    min_samples = _positive_int("fit_adjustment_policy", args, "min_samples", 30)
    verbose = _optional_bool("fit_adjustment_policy", args, "verbose")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "fit_adjustment_policy.py"),
        "--league",
        league.upper(),
        "--mode",
        "shadow",
        "--min-samples",
        str(min_samples),
    ]
    if verbose:
        cmd.append("--verbose")
    return cmd


def _validate_render_audit(args: dict[str, Any]) -> list[str]:
    _reject_unknown_args("render_audit", args, {"session_ids", "all_open", "verbose"})
    session_ids = args.get("session_ids")
    all_open = args.get("all_open", False)

    if not isinstance(all_open, bool):
        raise ValueError("render_audit.args.all_open must be bool")
    if session_ids is not None:
        if not isinstance(session_ids, list) or not all(
            isinstance(sid, str) and sid.strip() for sid in session_ids
        ):
            raise ValueError(
                "render_audit.args.session_ids must be a list of non-empty strings"
            )
    if bool(session_ids) == bool(all_open):
        raise ValueError(
            "render_audit requires exactly one of session_ids or all_open=true"
        )
    verbose = _optional_bool("render_audit", args, "verbose")

    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "render_session_audits.py")]
    if all_open:
        cmd.append("--all-open")
    elif session_ids:
        cmd += ["--session-ids", *session_ids]
    if verbose:
        cmd.append("--verbose")
    return cmd


# Strict allowlist. Adding a key here is a deliberate boundary change.
_DISPATCH: dict[str, Any] = {
    "fit_calibration": _validate_fit_calibration,
    "promote_profile": _validate_promote_profile,
    "report_calibration": _validate_report_calibration,
    "fetch_outcomes": _validate_fetch_outcomes,
    "ingest_traces": _validate_ingest_traces,
    "fetch_closing_lines": _validate_fetch_closing_lines,
    "score_evidence_signals": _validate_score_evidence_signals,
    "fit_adjustment_policy": _validate_fit_adjustment_policy,
    "render_audit": _validate_render_audit,
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

    # Sub-scripts inherit the default DB path (cwd=_REPO_ROOT). Report which DB
    # they will actually hit once, up front, so an empty/redirected DB is visible
    # before any action runs.
    try:
        from omega.trace.store import db_status

        st = db_status()
        logger.info(
            "Effective DB: %s (source=%s, traces=%s, integrity_ok=%s, EMPTY_HISTORY_MODE=%s). %s",
            st["effective_path"],
            st["source"],
            st["effective_trace_count"],
            st["effective_integrity_ok"],
            str(st["empty_history_mode"]).lower(),
            st["recommended_action"],
        )
    except Exception as exc:  # noqa: BLE001 — never let diagnostics break the run
        logger.warning("db_status summary unavailable: %s", exc)

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
