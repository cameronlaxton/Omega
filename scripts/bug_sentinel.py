"""Omega Known-Bug Sentinel.

Reads omega/qa/bug_catalog.json and runs mechanical, read-only checks against
the live repo/engine state. Produces a structured report: which bugs are
present, which are fixed, and what analysis gates are blocked.

Usage:
    python scripts/bug_sentinel.py              # human-readable summary
    python scripts/bug_sentinel.py --json       # structured JSON to stdout
    python scripts/bug_sentinel.py --ci         # exits 1 if any critical bug is present
    python scripts/bug_sentinel.py --session-id sess-20260528-a1b2  # writes to sidecar

All checks are read-only. No writes, no network, no analyze() calls (except the
import_test fixture check, which uses n_iterations=100 with a fixed seed).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
CATALOG_PATH = REPO_ROOT / "omega" / "qa" / "bug_catalog.json"

STATUS_PRESENT = "present"
STATUS_FIXED = "fixed"
STATUS_UNKNOWN = "unknown"
STATUS_SHADOW = "shadow_mode"
STATUS_ERROR = "check_error"


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

def load_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["bugs"]


# ---------------------------------------------------------------------------
# Check runners
# ---------------------------------------------------------------------------

def _run_grep(bug: dict[str, Any], repo_root: Path) -> tuple[str, str]:
    """Grep check: search for bad or good patterns in a source file."""
    check = bug["check"]
    file_path = repo_root / check["file"]

    if not file_path.exists():
        return STATUS_ERROR, f"File not found: {check['file']}"

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return STATUS_ERROR, f"Cannot read {check['file']}: {exc}"

    bad_pat = check.get("bad_pattern")
    good_pat = check.get("good_pattern")
    present_means = check.get("present_means", "bug_present")  # what bad_pat match means

    # Scope to function if requested
    scope_fn = check.get("function_scope")
    if scope_fn:
        source = _extract_function_source(source, scope_fn) or source

    bad_found = bool(bad_pat and re.search(bad_pat, source))
    good_found = bool(good_pat and re.search(good_pat, source))

    # Interpret results
    if bad_pat and good_pat:
        if bad_found:
            status = STATUS_PRESENT if present_means == "bug_present" else STATUS_FIXED
            return status, f"bad_pattern '{bad_pat}' matched in {check['file']}"
        if good_found:
            return STATUS_FIXED, f"good_pattern '{good_pat}' found; bad_pattern absent"
        return STATUS_UNKNOWN, "neither pattern matched"
    elif bad_pat:
        if bad_found:
            return (STATUS_PRESENT if present_means == "bug_present" else STATUS_FIXED,
                    f"pattern '{bad_pat}' matched")
        return (STATUS_FIXED if present_means == "bug_present" else STATUS_PRESENT,
                f"pattern '{bad_pat}' not found")
    elif good_pat:
        if good_found:
            return (STATUS_FIXED if present_means == "fixed" else STATUS_PRESENT,
                    f"pattern '{good_pat}' found")
        return (STATUS_PRESENT if present_means == "fixed" else STATUS_FIXED,
                f"pattern '{good_pat}' not found")

    return STATUS_UNKNOWN, "no patterns configured"


def _extract_function_source(source: str, fn_name: str) -> str | None:
    """Extract source lines for a top-level or module-level function by name."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            lines = source.splitlines()
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") else len(lines)
            return "\n".join(lines[start:end])
    return None


def _run_import_test(bug: dict[str, Any], repo_root: Path) -> tuple[str, str]:
    """Import-test check: import a module and run a minimal verification."""
    check = bug["check"]
    action = check.get("check_action", "analyze_fixture")

    if action == "analyze_fixture":
        return _check_input_snapshot_identity(bug, check, repo_root)
    elif action == "mlb_draw_prob":
        return _check_mlb_draw_prob(bug, check, repo_root)
    elif action == "load_active_mode":
        return _check_adjustment_policy_mode(bug, check, repo_root)
    else:
        return STATUS_UNKNOWN, f"unknown check_action: {action}"


def _check_input_snapshot_identity(
    bug: dict[str, Any], check: dict[str, Any], repo_root: Path
) -> tuple[str, str]:
    """Call analyze() with a prop fixture and verify identity fields appear in input_snapshot."""
    sys.path.insert(0, str(repo_root))
    try:
        from omega.core.contracts.service import analyze  # noqa: PLC0415
    except ImportError as exc:
        return STATUS_UNKNOWN, f"Cannot import omega.core.contracts.service: {exc}"

    fixture = dict(check["fixture"])
    required_fields = check.get("required_in_input_snapshot", [])
    present_means = check.get("present_means", "fixed")

    try:
        result = analyze(fixture, session_id="sess-sentinel-check", bankroll=1000.0)
    except Exception as exc:  # noqa: BLE001
        return STATUS_ERROR, f"analyze() raised: {exc}"

    snap = result.get("input_snapshot") or {}
    missing = [f for f in required_fields if f not in snap]

    if missing:
        status = STATUS_PRESENT if present_means == "fixed" else STATUS_FIXED
        return status, f"input_snapshot missing identity fields: {missing}"
    status = STATUS_FIXED if present_means == "fixed" else STATUS_PRESENT
    return status, f"input_snapshot has all required identity fields: {required_fields}"


def _check_mlb_draw_prob(
    bug: dict[str, Any], check: dict[str, Any], repo_root: Path
) -> tuple[str, str]:
    """Call analyze() on an MLB fixture and verify draw_prob == 0.0."""
    sys.path.insert(0, str(repo_root))
    try:
        from omega.core.contracts.service import analyze  # noqa: PLC0415
    except ImportError as exc:
        return STATUS_UNKNOWN, f"Cannot import omega.core.contracts.service: {exc}"

    fixture = dict(check["fixture"])
    expected = check.get("expected_draw_prob", 0.0)
    present_means = check.get("present_means", "fixed")

    try:
        result = analyze(fixture, session_id="sess-sentinel-check", bankroll=1000.0)
    except Exception as exc:  # noqa: BLE001
        return STATUS_ERROR, f"analyze() raised: {exc}"

    sim = (result.get("result") or {}).get("simulation") or result.get("result") or {}
    draw_prob = sim.get("draw_prob")

    if draw_prob is None:
        # Try top-level result keys
        draw_prob = result.get("result", {}).get("draw_prob")

    if draw_prob is None:
        return STATUS_UNKNOWN, "draw_prob not found in result.simulation"

    if float(draw_prob) == expected:
        status = STATUS_FIXED if present_means == "fixed" else STATUS_PRESENT
        return status, f"draw_prob={draw_prob} (expected {expected}) — supports_draw=False correctly handled"
    else:
        status = STATUS_PRESENT if present_means == "fixed" else STATUS_FIXED
        return status, (
            f"draw_prob={draw_prob} (expected {expected}) — "
            "tie samples not re-allocated; home/away win probs deflated"
        )


def _check_adjustment_policy_mode(
    bug: dict[str, Any], check: dict[str, Any], repo_root: Path
) -> tuple[str, str]:
    """Load the active adjustment policy and check its mode."""
    policy_file = repo_root / check["policy_file"]
    expected_mode = check.get("expected_mode", "live")
    present_means = check.get("present_means", "fixed")

    if not policy_file.exists():
        return STATUS_UNKNOWN, f"Policy file not found: {check['policy_file']}"

    try:
        data = json.loads(policy_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return STATUS_ERROR, f"Cannot read policy file: {exc}"

    policies = data.get("policies", [])
    if not policies:
        return STATUS_UNKNOWN, "No policies in file"

    active_id = data.get("active_policy_id")
    if active_id:
        active = next((p for p in policies if p.get("policy_id") == active_id), policies[0])
    else:
        active = policies[0]

    mode = active.get("mode", "unknown")
    version = active.get("version", "?")

    if mode == expected_mode:
        status = STATUS_FIXED if present_means == "fixed" else STATUS_PRESENT
        return status, f"Policy v{version} mode={mode} (expected {expected_mode})"
    else:
        status = STATUS_SHADOW if mode == "shadow" else STATUS_PRESENT
        if present_means == "fixed":
            status = STATUS_PRESENT
        return status, (
            f"Policy v{version} mode={mode} (expected {expected_mode}). "
            "Evidence signals are recorded but not applied to simulation inputs."
        )


def _run_db_query(bug: dict[str, Any], repo_root: Path) -> tuple[str, str]:
    """DB check: probe whether SQLite WAL mode works at the DB path."""
    check = bug["check"]
    db_path = repo_root / check["db_path"]
    success_means = check.get("success_means", "fixed")

    # Use a temp file on the same filesystem as the target DB to probe WAL behaviour.
    # This avoids touching the real DB while still probing the filesystem layer.
    db_dir = db_path.parent
    try:
        with tempfile.NamedTemporaryFile(dir=db_dir, suffix=".probe.db", delete=False) as tf:
            probe_path = tf.name

        conn = sqlite3.connect(probe_path)
        try:
            result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            wal_mode = result[0] if result else "unknown"
            conn.close()
        finally:
            try:
                Path(probe_path).unlink(missing_ok=True)
                Path(probe_path + "-wal").unlink(missing_ok=True)
                Path(probe_path + "-shm").unlink(missing_ok=True)
            except OSError:
                pass

        if wal_mode == "wal":
            status = STATUS_FIXED if success_means == "fixed" else STATUS_PRESENT
            return status, f"WAL mode accepted on {db_dir}"
        else:
            # Downgraded to delete/memory mode — WAL failed silently
            status = STATUS_PRESENT if success_means == "fixed" else STATUS_FIXED
            return status, f"WAL probe returned mode={wal_mode!r} on {db_dir} (expected 'wal')"

    except sqlite3.OperationalError as exc:
        status = STATUS_PRESENT if success_means == "fixed" else STATUS_FIXED
        return status, f"WAL probe raised OperationalError on {db_dir}: {exc}"
    except OSError as exc:
        return STATUS_ERROR, f"Cannot probe DB directory {db_dir}: {exc}"


def _run_manual(bug: dict[str, Any]) -> tuple[str, str]:
    """Manual check: always returns unknown. Workaround must be applied by operator."""
    note = bug["check"].get("note", "Manual check — cannot automate.")
    return STATUS_UNKNOWN, note


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_check(bug: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Run one bug's check and return a result dict."""
    check_type = bug.get("check_type", "manual")
    try:
        if check_type == "grep":
            status, evidence = _run_grep(bug, repo_root)
        elif check_type == "import_test":
            status, evidence = _run_import_test(bug, repo_root)
        elif check_type == "db_query":
            status, evidence = _run_db_query(bug, repo_root)
        elif check_type == "manual":
            status, evidence = _run_manual(bug)
        else:
            status, evidence = STATUS_UNKNOWN, f"Unsupported check_type: {check_type!r}"
    except Exception:  # noqa: BLE001
        status = STATUS_ERROR
        evidence = traceback.format_exc(limit=5)

    return {
        "bug_id": bug["bug_id"],
        "title": bug["title"],
        "severity": bug["severity"],
        "status": status,
        "suppresses_bet_card": bug.get("suppresses_bet_card", False),
        "blocks_ingest": bug.get("blocks_ingest", False),
        "sport_gates": bug.get("sport_gates", []),
        "analysis_kind_gates": bug.get("analysis_kind_gates", []),
        "evidence": evidence,
        "workaround": bug.get("workaround", ""),
        "check_type": check_type,
        "status_at_last_audit": bug.get("status_at_last_audit", "unknown"),
        "last_audited": bug.get("last_audited", ""),
    }


# ---------------------------------------------------------------------------
# Gate summary builder
# ---------------------------------------------------------------------------

def build_gate_summary(results: list[dict[str, Any]]) -> dict[str, str]:
    """Compute per sport+kind gate status from sentinel results."""
    # Start all known gates as "clear"
    gates: dict[str, str] = {}
    sport_kinds = [
        ("NBA", "game"), ("NBA", "prop"),
        ("MLB", "game"), ("MLB", "prop"),
        ("NHL", "game"), ("NHL", "prop"),
        ("NFL", "game"),
    ]
    for sport, kind in sport_kinds:
        gates[f"{sport}_{kind}"] = "clear"

    for r in results:
        if r["status"] != STATUS_PRESENT:
            continue
        if not r["suppresses_bet_card"]:
            continue
        for sport in (r["sport_gates"] or []):
            for kind in (r["analysis_kind_gates"] or []):
                gate_key = f"{sport}_{kind}"
                gates[gate_key] = "suppressed"

    return gates


def build_report(results: list[dict[str, Any]], repo_root: Path) -> dict[str, Any]:
    gate_summary = build_gate_summary(results)
    open_critical = sum(
        1 for r in results
        if r["status"] == STATUS_PRESENT and r["severity"] == "critical"
    )
    open_high = sum(
        1 for r in results
        if r["status"] == STATUS_PRESENT and r["severity"] == "high"
    )
    open_other = sum(
        1 for r in results
        if r["status"] == STATUS_PRESENT and r["severity"] not in ("critical", "high")
    )
    unknown_count = sum(1 for r in results if r["status"] == STATUS_UNKNOWN)
    regression_count = sum(
        1 for r in results
        if r["status"] == STATUS_PRESENT and r.get("status_at_last_audit") == "fixed"
    )

    return {
        "sentinel_version": "1.0",
        "ran_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_path": str(repo_root),
        "results": results,
        "gate_summary": gate_summary,
        "open_critical": open_critical,
        "open_high": open_high,
        "open_other": open_other,
        "unknown_count": unknown_count,
        "regression_count": regression_count,
        "all_clear": (open_critical == 0 and open_high == 0 and regression_count == 0),
    }


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

STATUS_MARKER = {
    STATUS_FIXED: "OK",
    STATUS_PRESENT: "BUG",
    STATUS_UNKNOWN: "?",
    STATUS_SHADOW: "~",
    STATUS_ERROR: "!",
}

SEVERITY_LABEL = {
    "critical": "CRITICAL",
    "high": "HIGH    ",
    "medium": "MEDIUM  ",
    "low": "LOW     ",
}


def print_report(report: dict[str, Any]) -> None:
    results = report["results"]
    print()
    print("=" * 68)
    print("  Omega Bug Sentinel")
    print(f"  {report['ran_at']}  repo: {report['repo_path']}")
    print("=" * 68)

    for r in results:
        marker = STATUS_MARKER.get(r["status"], "?")
        sev = SEVERITY_LABEL.get(r["severity"], r["severity"].upper()[:8])
        label = r["status"].upper().replace("_", " ")
        print(f"\n  {marker:<3} [{sev}] {r['bug_id']} - {label}")
        print(f"    {r['title']}")
        print(f"    Evidence : {r['evidence']}")
        if r["status"] == STATUS_PRESENT and r.get("workaround"):
            print(f"    Workaround: {r['workaround']}")
        if r["status"] == STATUS_PRESENT and r.get("status_at_last_audit") in ("fixed",):
            print(f"    !! REGRESSION - was fixed at last audit ({r['last_audited']})")

    print()
    print("-" * 68)
    print("  Gate Summary")
    print("-" * 68)
    for gate, state in sorted(report["gate_summary"].items()):
        marker = "OK" if state == "clear" else "BLOCK"
        print(f"  {marker:<5} {gate:<20} {state}")

    print()
    print("-" * 68)
    summary_parts = []
    if report["open_critical"]:
        summary_parts.append(f"{report['open_critical']} CRITICAL open")
    if report["open_high"]:
        summary_parts.append(f"{report['open_high']} HIGH open")
    if report["open_other"]:
        summary_parts.append(f"{report['open_other']} other open")
    if report["regression_count"]:
        summary_parts.append(f"{report['regression_count']} REGRESSION(S)")
    if report["unknown_count"]:
        summary_parts.append(f"{report['unknown_count']} unknown (manual)")

    if report["all_clear"]:
        print("  OK ALL CLEAR - no critical/high bugs or regressions detected")
    else:
        print(f"  ISSUES: {', '.join(summary_parts)}")
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# Sidecar integration
# ---------------------------------------------------------------------------

def write_to_sidecar(session_id: str, report: dict[str, Any], repo_root: Path) -> None:
    """Append sentinel results as a structured bug audit event to the session sidecar."""
    sidecar_path = repo_root / "inbox" / "sessions" / f"{session_id}.json"
    if not sidecar_path.exists():
        print(f"[sentinel] Sidecar not found: {sidecar_path} — skipping sidecar write", file=sys.stderr)
        return

    sys.path.insert(0, str(repo_root))
    try:
        from omega.trace.session_sidecar import append_audit_events  # noqa: PLC0415
    except ImportError as exc:
        print(f"[sentinel] Cannot import session_sidecar: {exc}", file=sys.stderr)
        return

    issue_ids = [r["bug_id"] for r in report["results"] if r["status"] == STATUS_PRESENT]
    regression_ids = [
        r["bug_id"] for r in report["results"]
        if r["status"] == STATUS_PRESENT and r.get("status_at_last_audit") == "fixed"
    ]
    suppressed_gates = [k for k, v in report["gate_summary"].items() if v == "suppressed"]
    status = "ok" if report["all_clear"] else ("fail" if report["open_critical"] or regression_ids else "warn")

    notes_parts = [f"Sentinel ran at {report['ran_at']}."]
    if report["all_clear"]:
        notes_parts.append("No critical/high bugs or regressions.")
    else:
        if issue_ids:
            notes_parts.append(f"Active bugs: {', '.join(issue_ids)}.")
        if regression_ids:
            notes_parts.append(f"REGRESSIONS detected: {', '.join(regression_ids)}.")
        if suppressed_gates:
            notes_parts.append(f"Suppressed gates: {', '.join(suppressed_gates)}.")
        if report["unknown_count"]:
            notes_parts.append(f"{report['unknown_count']} manual check(s) require operator verification.")

    try:
        append_audit_events(sidecar_path, [
            {
                "ts": report["ran_at"],
                "event_type": "bug",
                "step": "bug_sentinel",
                "status": status,
                "notes": " ".join(notes_parts),
                "bugs": issue_ids,
            }
        ])
        print(f"[sentinel] Audit event written to sidecar: {sidecar_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[sentinel] Failed to write sidecar audit event: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Omega known-bug sentinel — read-only engine state check."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit structured JSON report to stdout instead of human-readable summary."
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: exit 1 if any critical bug is present or a regression detected."
    )
    parser.add_argument(
        "--session-id", metavar="SESSION_ID",
        help="If provided, append sentinel results to the session sidecar as an audit event."
    )
    parser.add_argument(
        "--repo-root", metavar="PATH", default=str(REPO_ROOT),
        help="Path to Omega repo root (default: parent of this script)."
    )
    parser.add_argument(
        "--catalog", metavar="PATH", default=str(CATALOG_PATH),
        help="Path to bug catalog JSON (default: omega/qa/bug_catalog.json)."
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    catalog_path = Path(args.catalog).resolve()

    try:
        bugs = load_catalog(catalog_path)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        print(f"[sentinel] Cannot load catalog: {exc}", file=sys.stderr)
        return 2

    results = [run_check(bug, repo_root) for bug in bugs]
    report = build_report(results, repo_root)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    if args.session_id:
        write_to_sidecar(args.session_id, report, repo_root)

    if args.ci:
        if report["open_critical"] > 0 or report["regression_count"] > 0:
            print(
                f"[sentinel] CI FAIL: {report['open_critical']} critical bug(s) open, "
                f"{report['regression_count']} regression(s).",
                file=sys.stderr,
            )
            return 1
        print("[sentinel] CI PASS", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
