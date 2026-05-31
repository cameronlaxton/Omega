"""
scripts/backfill_evidence_signals.py — re-explode evidence_signals from frozen
pre-decision trace snapshots.

This is RE-DERIVATION, not recovery. Structured evidence is never lost: it lives
in ``full_trace.input_snapshot.evidence``. ``TraceStore._write_evidence_signals``
explodes it into the queryable ``evidence_signals`` table, but only on a trace's
first insert and only for non-empty evidence. Traces persisted before that table
existed (schema < V9) therefore carry evidence in their blob with zero rows in
the table, so retrospective scoring silently skips them.

This script finds those traces and re-explodes their *own* frozen snapshot into
the table, reusing the exact same explosion path as persist() so the rows are
identical to a first-insert. Because the only source is the trace's own
pre-decision ``input_snapshot``, provenance is ``original`` — there is no separate
recovery source and no ``recovered_predecision`` class here.

PROVENANCE SAFETY: the only source read is ``input_snapshot.evidence``. Outcomes,
box scores, closing lines, engine predictions, EV/edge/Kelly, and settlement
results are NEVER read — they cannot manufacture evidence. A trace whose snapshot
evidence is genuinely empty is left untouched and counted as unrecoverable; no
fake signals are invented.

Usage:
    python scripts/backfill_evidence_signals.py                 # dry-run (default)
    python scripts/backfill_evidence_signals.py --dry-run --verbose
    python scripts/backfill_evidence_signals.py --apply         # writes
    python scripts/backfill_evidence_signals.py --trace-id sandbox-...

Exit codes:
    0 — scan completed (dry-run or apply)
    1 — fatal error (DB missing/unreadable)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("backfill_evidence_signals")


@dataclass
class BackfillSummary:
    traces_scanned: int = 0
    graded_traces: int = 0
    original_evidence_in_blob: int = 0
    already_exploded: int = 0
    would_explode_rows: int = 0
    unrecoverable_empty_count: int = 0
    invalid_schema_count: int = 0
    applied_traces: int = 0
    applied_rows: int = 0


def _iter_trace_rows(store: TraceStore, trace_id: str | None):
    """Yield (trace_id, full_trace_dict) for one or all persisted traces."""
    if trace_id:
        rows = store.conn.execute(
            "SELECT trace_id, full_trace FROM traces WHERE trace_id = ?",
            (trace_id,),
        ).fetchall()
    else:
        rows = store.conn.execute(
            "SELECT trace_id, full_trace FROM traces ORDER BY timestamp"
        ).fetchall()
    for row in rows:
        try:
            trace = json.loads(row["full_trace"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("trace %s has unreadable full_trace; skipping", row["trace_id"])
            continue
        yield row["trace_id"], trace


def _has_outcome(store: TraceStore, trace_id: str) -> bool:
    row = store.conn.execute(
        "SELECT 1 FROM outcomes WHERE trace_id = ? "
        "UNION ALL SELECT 1 FROM prop_outcomes WHERE trace_id = ? LIMIT 1",
        (trace_id, trace_id),
    ).fetchone()
    return row is not None


def _mark_evidence_present(store: TraceStore, trace_id: str, trace: dict) -> None:
    """Update the stored blob's trace_quality to reflect re-exploded evidence.

    Targeted, additive edit: only trace_quality.evidence_status /
    evidence_provenance change. The original input_snapshot.evidence is never
    touched. Provenance is 'original' — the data came from the trace's own frozen
    pre-decision snapshot.
    """
    tq = dict(trace.get("trace_quality") or {})
    tq["evidence_status"] = "present"
    tq["evidence_provenance"] = "original"
    trace["trace_quality"] = tq
    store.conn.execute(
        "UPDATE traces SET full_trace = ? WHERE trace_id = ?",
        (json.dumps(trace, default=str), trace_id),
    )


def run_backfill(
    store: TraceStore,
    *,
    apply: bool = False,
    trace_id: str | None = None,
) -> BackfillSummary:
    """Scan traces and (optionally) re-explode evidence from their snapshots."""
    summary = BackfillSummary()

    for tid, trace in _iter_trace_rows(store, trace_id):
        summary.traces_scanned += 1
        if _has_outcome(store, tid):
            summary.graded_traces += 1

        input_snap = trace.get("input_snapshot") or {}
        evidence = input_snap.get("evidence")

        # Genuinely empty / absent snapshot evidence is unrecoverable. We never
        # fabricate signals from outcomes or engine math.
        if evidence is None or (isinstance(evidence, list) and not evidence):
            summary.unrecoverable_empty_count += 1
            continue
        if not isinstance(evidence, list):
            summary.invalid_schema_count += 1
            logger.warning("trace %s input_snapshot.evidence is not a list; skipping", tid)
            continue

        summary.original_evidence_in_blob += 1

        # Already materialized? Leave it; we only re-derive the missing rows.
        if store.get_evidence_signals(tid):
            summary.already_exploded += 1
            continue

        valid_signals = [s for s in evidence if isinstance(s, dict)]
        if not valid_signals:
            summary.invalid_schema_count += 1
            logger.warning("trace %s evidence carries no valid signal objects; skipping", tid)
            continue

        summary.would_explode_rows += len(valid_signals)
        if not apply:
            logger.debug("would re-explode %d signals for %s", len(valid_signals), tid)
            continue

        written = store._write_evidence_signals(tid, trace)
        _mark_evidence_present(store, tid, trace)
        store.conn.commit()
        summary.applied_traces += 1
        summary.applied_rows += written
        logger.info("re-exploded %d evidence signals for %s (provenance=original)", written, tid)

    return summary


def _print_summary(summary: BackfillSummary, *, apply: bool) -> None:
    logger.info("Evidence backfill summary")
    logger.info("-------------------------")
    logger.info("Mode:                          %s", "APPLY" if apply else "DRY-RUN")
    logger.info("Traces scanned:                %d", summary.traces_scanned)
    logger.info("Graded traces:                 %d", summary.graded_traces)
    logger.info("Original evidence in blob:     %d", summary.original_evidence_in_blob)
    logger.info("Already exploded:              %d", summary.already_exploded)
    logger.info("Unrecoverable (empty):         %d", summary.unrecoverable_empty_count)
    logger.info("Invalid schema:                %d", summary.invalid_schema_count)
    logger.info("Would re-explode rows:         %d", summary.would_explode_rows)
    if apply:
        logger.info("Applied traces:                %d", summary.applied_traces)
        logger.info("Applied evidence rows:         %d", summary.applied_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-explode evidence_signals from frozen pre-decision trace snapshots "
            "(input_snapshot.evidence only). Re-derivation, not recovery."
        )
    )
    parser.add_argument("--db", type=str, default=None, help="SQLite path")
    parser.add_argument("--trace-id", type=str, default=None, help="Limit to one trace_id")
    parser.add_argument(
        "--apply", action="store_true", help="Write evidence rows (default: dry-run)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Scan and report only (the default)"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Default to dry-run; --apply is required to write. --dry-run always wins.
    apply = bool(args.apply) and not args.dry_run

    try:
        store = TraceStore(db_path=args.db)
        log_effective_db(store, logger)
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot open trace store: %s", exc)
        return 1

    summary = run_backfill(store, apply=apply, trace_id=args.trace_id)
    _print_summary(summary, apply=apply)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
