"""
omega.ops.backfill_trace_quality â€” repopulate calibration-eligibility metadata
on persisted traces whose `trace_quality` block was written by an older producer.

Background:
    Some traces (notably sandbox exports produced before the eligibility schema
    landed) persisted `trace_quality` as only `{"aggregate_quality": <x>}`. The
    calibration-eligibility gate in omega-report-calibration /
    omega-fit-calibration reads
    `trace_quality.calibration_eligible / context_source / identity_status`, so
    those traces are invisible to calibration even when the underlying `result`
    block carries the needed values (context_source, status). This migration
    recomputes the eligibility sub-block from the data already present in each
    trace's `result` + `input_snapshot` and merges it back into `trace_quality`.

Policy reuse (no drift):
    Eligibility and identity are derived via the SAME functions the live engine
    uses â€” `omega.core.contracts.service.derive_calibration_eligibility` and
    `identity_status_for_fields`. This script never reimplements the rules.

Provenance:
    Recomputed blocks are stamped `trace_quality.eligibility_source =
    "backfill_v1"` so calibration fits can distinguish backfilled eligibility
    from natively-engine-computed eligibility. This is a DATA migration only â€”
    it does not change DDL, so it does not bump the schema_versions ladder
    (bumping CURRENT_VERSION would mislabel newly persisted traces).

Idempotent:
    Only traces whose `trace_quality.calibration_eligible` is absent (NULL) are
    touched. Re-running is a no-op. A trace already carrying a native eligibility
    block is left untouched.

Usage:
    omega-backfill-trace-quality --dry-run          # report only
    omega-backfill-trace-quality                    # apply
    omega-backfill-trace-quality --league NBA       # filter
    omega-backfill-trace-quality --db <path> -v

Rollback:
    omega_traces.db is git-tracked: `git checkout omega_traces.db` reverts all
    rewrites. (Or re-run with a restored DB.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.contracts.service import (  # noqa: E402
    derive_calibration_eligibility,
    identity_status_for_fields,
)
from omega.trace.store import TraceStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_trace_quality")

_ELIGIBILITY_SOURCE = "backfill_v1"


def _downgrades_for_status(status: str | None) -> list[str]:
    """Mirror service._result_downgrades for the persisted (dict) result."""
    if status == "skipped":
        return ["engine_skipped"]
    if status == "error":
        return ["engine_error"]
    return []


def _recompute_trace_quality(full_trace: dict) -> dict:
    """Return a new trace_quality block recomputed from the trace's own data.

    Preserves existing keys (e.g. aggregate_quality) and overlays the
    eligibility sub-block plus audit fields, mirroring engine_trace_quality.
    """
    existing_tq = full_trace.get("trace_quality") or {}
    result = full_trace.get("result") or {}
    snap = full_trace.get("input_snapshot") or {}
    kind = str(full_trace.get("kind", "unknown"))

    status = result.get("status")
    context_source = result.get("context_source")
    baseline_used = bool(result.get("baseline_used"))
    identity_status = identity_status_for_fields(kind, snap)
    downgrades = _downgrades_for_status(status)

    eligibility = derive_calibration_eligibility(
        status=status,
        context_source=context_source,
        baseline_used=baseline_used,
        identity_status=identity_status,
        downgrades=downgrades,
    )

    return {
        **existing_tq,
        "downgrades": downgrades,
        "passed": len(downgrades) == 0,
        "evidence_status": "present" if snap.get("evidence") else "empty",
        **eligibility,
        "eligibility_source": _ELIGIBILITY_SOURCE,
    }


def _needs_backfill(full_trace: dict) -> bool:
    tq = full_trace.get("trace_quality") or {}
    return tq.get("calibration_eligible") is None


def run(*, db: str | None, league: str | None, dry_run: bool) -> dict[str, int]:
    store = TraceStore(db_path=db)
    sql = "SELECT trace_id, league, full_trace FROM traces"
    params: list = []
    if league:
        sql += " WHERE league = ?"
        params.append(league.upper())
    rows = store.conn.execute(sql, params).fetchall()

    scanned = 0
    skipped_present = 0
    updated = 0
    now_eligible = 0
    for row in rows:
        scanned += 1
        full = json.loads(row["full_trace"])
        if not _needs_backfill(full):
            skipped_present += 1
            continue
        new_tq = _recompute_trace_quality(full)
        if new_tq.get("calibration_eligible"):
            now_eligible += 1
        if dry_run:
            logger.info(
                "[dry-run] %s: -> calibration_eligible=%s context_source=%s identity=%s reasons=%s",
                row["trace_id"],
                new_tq.get("calibration_eligible"),
                new_tq.get("context_source"),
                new_tq.get("identity_status"),
                new_tq.get("calibration_exclusion_reasons"),
            )
        else:
            store.rewrite_trace_quality(row["trace_id"], new_tq)
            updated += 1

    summary = {
        "scanned": scanned,
        "skipped_already_present": skipped_present,
        "needing_backfill": scanned - skipped_present,
        "updated": 0 if dry_run else updated,
        "would_be_eligible_after": now_eligible,
    }
    logger.info(
        "%s: scanned=%d, already_present=%d, needing_backfill=%d, %s=%d, eligible_after=%d",
        "DRY-RUN" if dry_run else "DONE",
        summary["scanned"],
        summary["skipped_already_present"],
        summary["needing_backfill"],
        "would_update" if dry_run else "updated",
        summary["needing_backfill"] if dry_run else summary["updated"],
        summary["would_be_eligible_after"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill calibration-eligibility metadata into persisted trace_quality blocks."
    )
    parser.add_argument("--db", default=None, help="DB path (default: var/omega_traces.db)")
    parser.add_argument("--league", default=None, help="Restrict to one league (e.g. NBA)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report changes without writing."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.verbose:
        logging.getLogger("backfill_trace_quality").setLevel(logging.DEBUG)
    run(db=args.db, league=args.league, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





