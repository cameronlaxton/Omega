"""
omega-score-evidence-signals â€” retrospective scoring of structured evidence.

After outcomes attach to traces, this script measures whether each reasoning
signal was predictive: it JOINs ``evidence_signals`` rows to their trace's
realized outcome, scores direction-correctness and confidence calibration, and
upserts per-key aggregates into the ``signal_performance`` table.

``report_calibration.py`` then surfaces those aggregates to the agent at session
start, closing the reasoning-improvement loop: the agent learns which of its own
signal types and sources are worth trusting.

This script is read-only with respect to predictions and outcomes â€” it only
writes the signal_performance aggregate table.

Determinism: ``dataset_hash`` is the sha256 of the sorted scored observations,
so re-running on the same DB snapshot replaces the same rows rather than
duplicating them.

Usage:
    omega-score-evidence-signals
    omega-score-evidence-signals --league NBA
    omega-score-evidence-signals --league NBA --window-days 60
    omega-score-evidence-signals --dry-run --verbose

Exit codes:
    0 â€” scoring completed (even if zero signals were scoreable)
    1 â€” fatal error (DB missing/unreadable)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.strategy.signal_performance import (  # noqa: E402
    ScoredSignal,
    accumulate_signal_performance,
    score_trace_signals,
)
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("score_evidence_signals")


def _dataset_hash(scored: list[ScoredSignal]) -> str:
    """Deterministic hash of the scored observation set."""
    payload = sorted(
        (s.signal_type, s.source, s.obs_window, s.league,
         round(s.confidence, 6), s.direction_correct)
        for s in scored
    )
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


@dataclass
class ScoreSummary:
    """Evidence-coverage breakdown for one scoring run.

    Distinguishes *why* a graded trace did not contribute signals so empty
    evidence is reported as an evidence-learning gap â€” NOT a probability-
    calibration failure.
    """

    graded_traces: int = 0
    evidence_present: int = 0
    skipped_empty: int = 0
    skipped_qa_failed: int = 0
    rows_produced: int = 0


def collect_scores(
    store: TraceStore, graded: list[dict]
) -> tuple[list[ScoredSignal], ScoreSummary]:
    """Score every graded trace's evidence, recording skip reasons by status.

    Skips QA-failed traces (evidence learning is blocked for them) and traces
    with no evidence rows (an evidence-learning gap, not a calibration problem).
    """
    summary = ScoreSummary(graded_traces=len(graded))
    all_scored: list[ScoredSignal] = []
    for trace in graded:
        trace_id = trace.get("trace_id")
        if not trace_id:
            continue
        qa_row = store.get_qa_verdict(str(trace_id))
        if qa_row and qa_row.get("verdict") == "fail":
            summary.skipped_qa_failed += 1
            continue
        evidence_rows = store.get_evidence_signals(str(trace_id))
        if not evidence_rows:
            summary.skipped_empty += 1
            continue
        summary.evidence_present += 1
        all_scored.extend(score_trace_signals(trace, evidence_rows))
    return all_scored, summary


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
        log_effective_db(store, logger)
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

    all_scored, summary = collect_scores(store, graded)
    rows = accumulate_signal_performance(all_scored) if all_scored else []
    summary.rows_produced = len(rows)

    # Always print the coverage summary. Empty evidence is an evidence-learning
    # gap (these traces may still be probability-calibration eligible); it is
    # NOT a probability-calibration failure.
    logger.info("Evidence scoring summary")
    logger.info("------------------------")
    logger.info("Graded traces:                 %d", summary.graded_traces)
    logger.info("Evidence-eligible (present):   %d", summary.evidence_present)
    logger.info("Skipped: empty evidence:       %d", summary.skipped_empty)
    logger.info("Skipped: QA failed:            %d", summary.skipped_qa_failed)
    logger.info("Signal-performance rows:       %d", summary.rows_produced)

    if not all_scored:
        # Not a failure: traces without evidence simply cannot feed signal
        # scoring. Available evidence (if any) was still scored above.
        logger.info(
            "No scoreable evidence signals among %d graded traces "
            "(%d skipped: empty evidence). Nothing to write.",
            summary.graded_traces,
            summary.skipped_empty,
        )
        store.close()
        return 0

    dataset_hash = _dataset_hash(all_scored)

    if args.dry_run:
        logger.info("DRY-RUN â€” %d signal-performance rows (dataset_hash=%s):",
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




