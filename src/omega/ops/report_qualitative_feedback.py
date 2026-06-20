"""
omega-report-qualitative-feedback — the Issue #22 Phase 6 feedback gate.

Reads persisted traces and reports, per qualitative evidence signal, coverage
across the dimensions the closed-loop feedback gate must distinguish (present /
applied / evidence mode / backend path / market type / calibration eligibility /
outcome resolution). Traces produced before evidence enrichment lack the
normalized fields and are labeled *insufficient* — counted and listed, never
folded into a signal's aggregates.

This script is read-only: it never writes predictions, outcomes, or the
signal_performance table. It is the safe readiness view that tells the operator
which signals are mature enough to feed ``omega-score-evidence-signals``.

Usage:
    omega-report-qualitative-feedback
    omega-report-qualitative-feedback --league NBA --window-days 60
    omega-report-qualitative-feedback --session <id> --out var/reports

Exit codes:
    0 — report rendered (even if zero traces matched)
    1 — fatal error (DB missing/unreadable)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.strategy.qualitative_feedback import (  # noqa: E402
    build_report,
    render_report_markdown,
)
from omega.trace._atomic import atomic_write_text  # noqa: E402
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("report_qualitative_feedback")


def _load_traces(store: TraceStore, args: argparse.Namespace) -> list[dict]:
    if args.session:
        return store.query_by_session(args.session)
    cutoff = None
    if args.window_days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=args.window_days)).isoformat()
    # has_outcome=None: include pending traces so the report can show which
    # applied signals are still awaiting outcome resolution.
    return store.query_traces(
        league=args.league.upper() if args.league else None,
        start=cutoff,
        has_outcome=None,
        limit=args.limit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qualitative evidence-signal feedback gate report."
    )
    parser.add_argument("--league", default=None, help="Filter to one league (default: all)")
    parser.add_argument(
        "--window-days", type=int, default=None,
        help="Only classify traces newer than N days (default: all history)",
    )
    parser.add_argument("--session", default=None, help="Restrict to one session_id")
    parser.add_argument("--db", default=None, help="SQLite path")
    parser.add_argument("--limit", type=int, default=100_000, help="Max traces to load")
    parser.add_argument(
        "--out", default=None,
        help="Directory to write the Markdown report (default: print only)",
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

    try:
        traces = _load_traces(store, args)
    finally:
        store.close()

    report = build_report(traces)
    markdown = render_report_markdown(report)

    logger.info(
        "Classified %d traces: %d sufficient, %d insufficient, %d no-evidence.",
        report.total_traces, report.sufficient, report.insufficient, report.no_evidence,
    )
    print(markdown)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        scope = args.session or args.league or "all"
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scope)
        target = out_dir / f"{ts}_qualitative_feedback_{safe}.md"
        atomic_write_text(target, markdown)
        logger.info("Wrote %s", target)

    return 0


if __name__ == "__main__":
    sys.exit(main())
