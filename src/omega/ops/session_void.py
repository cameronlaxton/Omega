"""Typed, audited replacement for ad hoc raw-SQL session cleanup.

Antigravity/Cowork sessions have repeatedly reached for a hand-written
sqlite3 script to purge a session's traces/ledger/evidence/QA rows before a
same-session rerun (e.g. after fixing a gate that produced bad traces). That
pattern has no audit trail, no backup, and bypasses the typed store entirely
-- exactly the raw-SQL mutation `AGENTS.md` says must stay on typed MCP/CLI
paths.

``omega-session-void --session-id <sid> --reason "<text>" --apply``:

1. Exports every trace for the session (via ``TraceStore.query_by_session``,
   which already carries attached outcomes/prop_outcomes/bet_records) to
   ``var/void_archive/<session_id>_<timestamp>.json`` BEFORE deleting anything
   -- a void is recoverable by inspection, unlike a bare ``DELETE``.
2. Deletes the session's rows from evidence_signals, simulation_distributions,
   trace_qa_verdicts (children), then bet_ledger and traces (parents) -- the
   same FK-safe order manual cleanup scripts have used, now in one place.
3. Appends a ``command`` audit event to the session's sidecar (if it exists)
   recording the reason, row counts, and the export path, so the void is part
   of the session's own audit trail, not an invisible side effect.

Dry-run by default (report only); nothing is exported or deleted without
``--apply``, matching the other mutating ops CLIs in this repo.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.session_sidecar import append_audit_events  # noqa: E402
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("session_void")

_DEFAULT_SESSIONS_INBOX = _REPO_ROOT / "var" / "inbox" / "sessions"
_DEFAULT_VOID_ARCHIVE = _REPO_ROOT / "var" / "void_archive"

# Children first (FK-safe delete order), then the two parent tables.
_CHILD_TABLES_BY_TRACE_ID = ("evidence_signals", "simulation_distributions", "trace_qa_verdicts")


def _counts(store: TraceStore, session_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    counts["traces"] = store.conn.execute(
        "SELECT COUNT(*) FROM traces WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    counts["bet_ledger"] = store.conn.execute(
        "SELECT COUNT(*) FROM bet_ledger WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    for table in _CHILD_TABLES_BY_TRACE_ID:
        counts[table] = store.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE trace_id IN "  # noqa: S608 - table from fixed tuple, not input
            "(SELECT trace_id FROM traces WHERE session_id = ?)",
            (session_id,),
        ).fetchone()[0]
    return counts


def _export(store: TraceStore, session_id: str, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = archive_dir / f"{session_id}_{ts}.json"
    traces = store.query_by_session(session_id)
    out_path.write_text(
        json.dumps({"session_id": session_id, "exported_at": ts, "traces": traces}, indent=2),
        encoding="utf-8",
    )
    return out_path


def _delete(store: TraceStore, session_id: str) -> None:
    with store.conn:
        for table in _CHILD_TABLES_BY_TRACE_ID:
            store.conn.execute(
                f"DELETE FROM {table} WHERE trace_id IN "  # noqa: S608 - table from fixed tuple, not input
                "(SELECT trace_id FROM traces WHERE session_id = ?)",
                (session_id,),
            )
        store.conn.execute("DELETE FROM bet_ledger WHERE session_id = ?", (session_id,))
        store.conn.execute("DELETE FROM traces WHERE session_id = ?", (session_id,))


def _record_audit_event(
    sidecar_path: Path, session_id: str, reason: str, counts: dict[str, int], export_path: Path
) -> bool:
    if not sidecar_path.exists():
        return False
    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    try:
        export_label = export_path.relative_to(_REPO_ROOT)
    except ValueError:
        export_label = export_path  # a custom --archive-dir outside the repo
    notes = f"session_void: reason={reason!r} counts={counts} export={export_label}"
    append_audit_events(
        sidecar_path,
        [
            {
                "ts": now,
                "event_type": "command",
                "step": "session_void",
                "status": "ok",
                "notes": notes,
            }
        ],
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True, help="Session ID to void, e.g. sess-20260709-abcd")
    parser.add_argument("--reason", required=True, help="Why this session's data is being voided")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=_DEFAULT_SESSIONS_INBOX,
        help="Directory containing <session_id>.json sidecars (default: var/inbox/sessions)",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=_DEFAULT_VOID_ARCHIVE,
        help="Where to write the pre-delete export (default: var/void_archive)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually export+delete; default is dry-run report only"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = TraceStore(db_path=args.db)
    log_effective_db(store, logger)

    counts = _counts(store, args.session_id)
    total = sum(counts.values())
    logger.info("Session %s row counts: %s (total=%d)", args.session_id, counts, total)

    if total == 0:
        logger.info("Nothing to void for session %s.", args.session_id)
        store.close()
        return 0

    if not args.apply:
        logger.info("Dry-run only. Re-run with --apply to export+delete.")
        store.close()
        return 0

    export_path = _export(store, args.session_id, args.archive_dir)
    logger.info("Exported %d trace(s) to %s before deleting.", counts["traces"], export_path)

    _delete(store, args.session_id)
    logger.info("Deleted session %s rows: %s", args.session_id, counts)

    sidecar_path = args.sidecar_dir / f"{args.session_id}.json"
    recorded = _record_audit_event(sidecar_path, args.session_id, args.reason, counts, export_path)
    if recorded:
        logger.info("Recorded session_void audit event on sidecar %s.", sidecar_path.name)
    else:
        logger.warning(
            "No sidecar at %s -- void completed but was NOT recorded in a session audit trail.",
            sidecar_path,
        )

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
