"""Render derived Markdown reports for Omega operator sessions."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.paths import reports_dir  # noqa: E402
from omega.trace._atomic import atomic_write_text  # noqa: E402
from omega.trace.db import require_sqlite_backend  # noqa: E402
from omega.trace.session_report.context_bundle import (  # noqa: E402
    ContextBundleError,
    load_context_bundle,
)
from omega.trace.session_report.extractors import extract_intake_report  # noqa: E402
from omega.trace.session_report.markdown import render_intake_markdown  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("render_session_report")
UTC = timezone.utc


def _target_path(out_dir: Path, *, kind: str, league: str | None, session_id: str | None) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    scope = session_id or league or "all"
    safe_scope = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scope)
    return out_dir / f"{ts}_{kind}_{safe_scope}.md"


def _render_one(args: argparse.Namespace, kind: str):
    if kind != "intake":
        raise NotImplementedError(f"{kind} report is not implemented in the intake vertical slice")
    bundle = (
        load_context_bundle(args.context_bundle) if args.context_mode == "persisted+cited" else None
    )
    if args.context_mode == "persisted+cited" and bundle is None:
        raise ValueError("persisted+cited requires --context-bundle")
    store = TraceStore(db_path=args.db)
    try:
        data = extract_intake_report(
            store,
            session_id=args.session_id,
            league=args.league.upper() if args.league else None,
            since=args.since,
            until=args.until,
            context_mode=args.context_mode,
            context_bundle=bundle,
        )
        if args.fail_on_context_mismatch and data.ignored_context_entries:
            raise ValueError(
                f"context bundle had {len(data.ignored_context_entries)} ignored entrie(s)"
            )
        markdown = render_intake_markdown(data)
    finally:
        store.close()

    out = _target_path(args.out_dir, kind=kind, league=args.league, session_id=args.session_id)
    if args.dry_run:
        logger.info("DRY-RUN would write %s", out)
        return data
    atomic_write_text(out, markdown)
    logger.info("wrote %s", out)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render derived Omega session reports.")
    parser.add_argument(
        "--kind", required=True, choices=["intake", "closing-lines", "portfolio", "all"]
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--league", default=None)
    parser.add_argument("--out-dir", type=Path, default=reports_dir() / "session_reports")
    parser.add_argument(
        "--context-mode", choices=["persisted", "persisted+cited"], default="persisted"
    )
    parser.add_argument("--context-bundle", type=Path, default=None)
    parser.add_argument("--fail-on-context-mismatch", action="store_true")
    parser.add_argument(
        "--allow-zero-evidence",
        action="store_true",
        help="Render the report but do NOT fail the run when too many traces are "
        "zero-evidence/empty-context (the blocker section still appears).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    require_sqlite_backend("render_session_report.py")
    if args.context_mode == "persisted+cited" and args.context_bundle is None:
        logger.error("persisted+cited requires --context-bundle")
        return 2
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    kinds = ["intake"] if args.kind == "all" else [args.kind]
    failures = 0
    blocked = False
    for kind in kinds:
        try:
            data = _render_one(args, kind)
            if getattr(data, "zero_evidence_blocked", False):
                blocked = True
                logger.error(
                    "ZERO-EVIDENCE BLOCKER (%s report):\n%s",
                    kind,
                    data.zero_evidence_diagnostic,
                )
        except (ContextBundleError, ValueError, NotImplementedError) as exc:
            logger.error("%s report failed: %s", kind, exc)
            failures += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s report failed unexpectedly: %s", kind, exc)
            failures += 1
    if failures:
        return 1
    # The run summary is failed when the session reasons blind, unless an
    # operator explicitly opts out. Distinct exit code (3) so automation can
    # tell a zero-evidence block from a render error (1) or arg error (2).
    if blocked and not args.allow_zero_evidence:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
