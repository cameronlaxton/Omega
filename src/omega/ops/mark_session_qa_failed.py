"""Append a QA-failed quality_gate audit event to session sidecars.

This is a quarantine helper: it records that a session's exported traces are
not betting-grade and must not be ingested for calibration learning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.paths import session_inbox_dir  # noqa: E402
from omega.trace.session_sidecar import SessionSidecar, append_audit_events  # noqa: E402

_DEFAULT_0528_SESSIONS = (
    "sess-20260528-nba1",
    "sess-20260528-prp1",
    "sess-20260528-mlb1",
    "sess-20260528-wnb1",
)

_DEFAULT_REASON = (
    "QA-failed quarantine: unverified context, injury translation gaps, "
    "engine parameter mismatches, WNBA total score-unit mismatch, estimated "
    "prop-line usage, and trace/summary drift. Do not ingest for calibration."
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def mark_sidecar(path: Path, *, reason: str, force_duplicate: bool = False) -> bool:
    """Append the QA-failed audit event. Returns True when an event was written."""
    sidecar = SessionSidecar.from_path(path)
    if not force_duplicate:
        for event in sidecar.audit_events:
            if (
                event.event_type == "quality_gate"
                and event.step == "qa_failed_quarantine_0528"
                and event.status == "fail"
            ):
                return False

    append_audit_events(
        path,
        [
            {
                "ts": _utc_now(),
                "event_type": "quality_gate",
                "step": "qa_failed_quarantine_0528",
                "status": "fail",
                "notes": reason,
                "trace_ids": [],
                "bugs": [
                    "0528: unverified injuries and context translation gaps",
                    "0528: invalid/proxy team metrics supplied to engine",
                    "0528: WNBA total markets unsafe due Markov score-unit mismatch",
                    "0528: estimated/milestone prop-line handling was not betting-grade",
                ],
            }
        ],
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=list(_DEFAULT_0528_SESSIONS),
        help="Session IDs to mark. Defaults to the four 2026-05-28 sessions.",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=session_inbox_dir(),
        help="Directory containing <session_id>.json sidecars (default: var/inbox/sessions).",
    )
    parser.add_argument(
        "--reason",
        default=_DEFAULT_REASON,
        help="Audit event notes text.",
    )
    parser.add_argument(
        "--force-duplicate",
        action="store_true",
        help="Append even if the quarantine event already exists.",
    )
    args = parser.parse_args(argv)

    written = 0
    skipped = 0
    for session_id in args.sessions:
        path = args.sessions_dir / f"{session_id}.json"
        if not path.exists():
            print(f"MISSING {path}")
            return 1
        did_write = mark_sidecar(
            path,
            reason=args.reason,
            force_duplicate=args.force_duplicate,
        )
        if did_write:
            written += 1
            print(f"QA_FAILED {session_id}")
        else:
            skipped += 1
            print(f"SKIP already marked {session_id}")

    print(f"Done. written={written} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
