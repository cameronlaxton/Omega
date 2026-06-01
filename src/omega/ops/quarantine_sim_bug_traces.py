"""Quarantine traces affected by known simulation bugs.

The script never deletes rows or rewrites predictions. It only marks matching
trace JSON blobs as calibration-ineligible with auditable exclusion metadata.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

_DEFAULT_DB = _REPO_ROOT / "var" / "var/omega_traces.db"
_REASON = "sim_def_rating_inverted_pre_20260525"
UTC = timezone.utc


def _timestamp_before(value: str | None, cutoff: str) -> bool:
    if not value:
        return True
    return value < cutoff


def _quarantine_trace(trace: dict[str, Any], cutoff: str) -> dict[str, Any]:
    updated = dict(trace)
    tq = dict(updated.get("trace_quality") or updated.get("quality_gate") or {})
    reasons = set(tq.get("calibration_exclusion_reasons") or [])
    reasons.add(_REASON)
    tq["calibration_eligible"] = False
    tq["calibration_exclusion_reasons"] = sorted(reasons)
    updated["trace_quality"] = tq
    updated["sim_bug_quarantine_meta"] = {
        "reason": _REASON,
        "cutoff": cutoff,
        "quarantined_at": datetime.now(UTC).isoformat(),
        "script": "quarantine_sim_bug_traces.py",
    }
    return updated


def _rollback_trace(trace: dict[str, Any]) -> dict[str, Any]:
    updated = dict(trace)
    tq = dict(updated.get("trace_quality") or updated.get("quality_gate") or {})
    reasons = [r for r in (tq.get("calibration_exclusion_reasons") or []) if r != _REASON]
    tq["calibration_exclusion_reasons"] = reasons
    if not reasons:
        tq["calibration_eligible"] = True
    updated["trace_quality"] = tq
    if (updated.get("sim_bug_quarantine_meta") or {}).get("reason") == _REASON:
        updated.pop("sim_bug_quarantine_meta", None)
    return updated


def quarantine(
    db_path: Path,
    *,
    cutoff: str,
    apply: bool = False,
    rollback: bool = False,
) -> tuple[int, int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    matched = 0
    changed = 0
    try:
        rows = conn.execute(
            "SELECT trace_id, timestamp, full_trace FROM traces WHERE league = 'MLB'"
        ).fetchall()
        for row in rows:
            trace = json.loads(row["full_trace"])
            if not rollback and not _timestamp_before(row["timestamp"], cutoff):
                continue
            tq = trace.get("trace_quality") or trace.get("quality_gate") or {}
            reasons = tq.get("calibration_exclusion_reasons") or []
            has_reason = _REASON in reasons
            if rollback:
                if not has_reason:
                    continue
                updated = _rollback_trace(trace)
            else:
                if has_reason:
                    continue
                updated = _quarantine_trace(trace, cutoff)
            matched += 1
            if apply:
                conn.execute(
                    "UPDATE traces SET full_trace = ? WHERE trace_id = ?",
                    (json.dumps(updated, default=str, sort_keys=False), row["trace_id"]),
                )
                changed += 1
        if apply:
            conn.commit()
    finally:
        conn.close()
    return matched, changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Quarantine MLB traces affected by sim bugs")
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--cutoff", default="2026-05-25T00:00:00+00:00")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    matched, changed = quarantine(
        args.db, cutoff=args.cutoff, apply=args.apply, rollback=args.rollback
    )
    mode = "rollback" if args.rollback else "quarantine"
    write_mode = "applied" if args.apply else "dry_run"
    print(f"mode={mode} write_mode={write_mode} matched={matched} changed={changed}")
    print(f"reason={_REASON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




