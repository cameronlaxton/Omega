"""
scripts/score_evidence_signals.py — retrospective scoring of structured evidence.

After outcomes attach to traces, this script measures whether each reasoning
signal was predictive: it JOINs ``evidence_signals`` rows to their trace's
realized outcome, scores direction-correctness and confidence calibration, and
upserts per-key aggregates into the ``signal_performance`` table.

``report_calibration.py`` then surfaces those aggregates to the agent at session
start, closing the reasoning-improvement loop: the agent learns which of its own
signal types and sources are worth trusting.

This script is read-only with respect to predictions and outcomes — it only
writes the signal_performance aggregate table.

Determinism: ``dataset_hash`` is the sha256 of the sorted scored observations,
so re-running on the same DB snapshot replaces the same rows rather than
duplicating them.

Usage:
    python scripts/score_evidence_signals.py
    python scripts/score_evidence_signals.py --league NBA
    python scripts/score_evidence_signals.py --league NBA --window-days 60
    python scripts/score_evidence_signals.py --dry-run --verbose

Exit codes:
    0 — scoring completed (even if zero signals were scoreable)
    1 — fatal error (DB missing/unreadable)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.strategy.signal_performance import (  # noqa: E402
    ScoredSignal,
    accumulate_signal_performance,
    score_trace_signals,
)
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("score_evidence_signals")


def _dataset_hash(scored: list[ScoredSignal]) -> str:
    """Deterministic hash of the scored observation set."""
    payload = sorted(
        (s.signal_type, s.source, s.obs_window, s.league,
         round(s.confidence, 6), s.direction_correct)
        for s in scored
    )
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score structured evidence signals against realized outcomes."
    )
    parser.add_argument("--league", default=None, help="Filter to one league (default: all)")
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Only score traces newer than N days (default: all history)",
    )
    parser.add_argument("--db", type=str, default=None, help="SQLite path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Score and report but do not write the table"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        store = TraceStore(db_path=args.db)
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot open trace store: %s", exc)
        return 1

    cutoff = None
    if args.window_days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=args.window_days)).isoformat()

    graded = store.query_traces(
        league=args.league,
        start=cutoff,
        has_outcome=True,
        limit=100_000,
    )
    logger.info("Loaded %d graded traces.", len(graded))

    all_scored: list[ScoredSignal] = []
    traces_with_evidence = 0
    for trace in graded:
        trace_id = trace.get("trace_id")
        if not trace_id:
            continue
        evidence_rows = store.get_evidence_signals(str(trace_id))
        if not evidence_rows:
            continue
        traces_with_evidence += 1
        all_scored.extend(score_trace_signals(trace, evidence_rows))

    logger.info(
        "%d graded traces carried evidence; %d directional signals were scoreable.",
        traces_with_evidence,
        len(all_scored),
    )

    if not all_scored:
        logger.info("No scoreable evidence signals — nothing to write.")
        store.close()
        return 0

    rows = accumulate_signal_performance(all_scored)
    dataset_hash = _dataset_hash(all_scored)

    if args.dry_run:
        logger.info("DRY-RUN — %d signal-performance rows (dataset_hash=%s):",
                    len(rows), dataset_hash[:12])
    else:
        written = store.upsert_signal_performance(rows, dataset_hash)
        logger.info("Wrote %d signal-performance rows (dataset_hash=%s).",
                    written, dataset_hash[:12])

    for r in rows:
        logger.info(
            "  %-22s src=%-18s win=%-7s league=%-6s n=%-3d acc=%.2f "
            "conf=%.2f gap=%+.2f brier=%.3f",
            r.signal_type, r.source, r.obs_window, r.league, r.sample_size,
            r.direction_accuracy, r.mean_confidence, r.calibration_gap, r.brier,
        )

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
