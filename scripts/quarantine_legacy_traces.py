"""Quarantine legacy trace rows that are unsafe for calibration.

This script never deletes rows. It marks affected trace JSON blobs with
``trace_quality.calibration_eligible=false`` and explicit exclusion reasons so
the audit ledger remains intact while calibration consumers can default-deny
legacy or poisoned rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEFAULT_DB = _REPO_ROOT / "omega_traces.db"

_PROP_IDENTITY_FIELDS = ("player_name", "home_team", "away_team", "game_date", "line")


def _reasons(trace: dict[str, Any]) -> list[str]:
    reasons: set[str] = set()
    kind = trace.get("kind")
    result = trace.get("result") or {}
    tq = trace.get("trace_quality") or trace.get("quality_gate") or {}

    context_source = result.get("context_source") or tq.get("context_source")
    baseline_used = bool(result.get("baseline_used") or tq.get("baseline_used"))
    if not context_source:
        reasons.add("legacy_missing_context_source")
    elif context_source == "league_default" or baseline_used:
        reasons.add("baseline_default_context")

    if result.get("status") == "skipped" or "engine_skipped" in (trace.get("downgrades") or []):
        reasons.add("engine_skipped")

    if kind == "prop":
        snap = trace.get("input_snapshot") or {}
        if any(not snap.get(field) for field in _PROP_IDENTITY_FIELDS):
            reasons.add("legacy_missing_identity")

    return sorted(reasons)


def _quarantined_trace(trace: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    updated = dict(trace)
    tq = dict(updated.get("trace_quality") or updated.get("quality_gate") or {})
    existing = set(tq.get("calibration_exclusion_reasons") or [])
    tq["calibration_eligible"] = False
    tq["calibration_exclusion_reasons"] = sorted(existing | set(reasons))
    if "baseline_default_context" in reasons:
        tq["context_source"] = "league_default"
        tq["baseline_used"] = True
    elif "legacy_missing_context_source" in reasons:
        tq.setdefault("context_source", None)
        tq.setdefault("baseline_used", False)
    if "legacy_missing_identity" in reasons:
        tq["identity_status"] = "missing"
    updated["trace_quality"] = tq
    return updated


def quarantine(db_path: Path, apply: bool = False) -> tuple[int, dict[str, int]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    counts: dict[str, int] = {}
    changed = 0
    try:
        rows = conn.execute("SELECT trace_id, full_trace FROM traces").fetchall()
        for row in rows:
            trace = json.loads(row["full_trace"])
            reasons = _reasons(trace)
            if not reasons:
                continue
            changed += 1
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
            if apply:
                updated = _quarantined_trace(trace, reasons)
                conn.execute(
                    "UPDATE traces SET full_trace = ? WHERE trace_id = ?",
                    (json.dumps(updated, default=str, sort_keys=False), row["trace_id"]),
                )
        if apply:
            conn.commit()
    finally:
        conn.close()
    return changed, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Quarantine legacy calibration-unsafe traces")
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--apply", action="store_true", help="Write quarantine metadata")
    args = parser.parse_args()

    changed, counts = quarantine(args.db, apply=args.apply)
    mode = "applied" if args.apply else "dry_run"
    print(f"mode={mode} traces_matched={changed}")
    for reason, count in sorted(counts.items()):
        print(f"{reason}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
