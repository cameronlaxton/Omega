"""
omega-ingest-closing-lines â€” drain `var/inbox/closing_lines/*.json` into var/omega_traces.db.

This is the stable file-to-DB bridge for closing-line snapshots. The snapshot may
come from agent WebFetch, live Odds API capture, or paid historical Odds API
backfill. This script ingests those files into the closing_lines table.

No HTTP. No network calls. The agent is responsible for sourcing the data; this
script is a pure file-to-DB bridge mirroring omega-ingest-traces.

Expected file shape:
    {
      "trace_id": "sandbox-...",
      "captured_at": "2026-05-15T23:55:00Z",   # ISO 8601 UTC; required
      "source": "draftkings.com",              # required
      "lines": [
        {
          "market": "spread",                  # spread | total | moneyline | player_prop:<stat>
          "selection_descriptor": "lakers_-3.5",
          "closing_line": -3.5,                # optional for moneyline
          "closing_odds": -110                 # American; required
        },
        ...
      ]
    }

Atomicity: each file's `lines` array is applied as a single transaction. If any
entry fails validation, the entire file is rejected and moved to failed/ with a
sibling .error.txt â€” no partial writes.

Idempotency: TraceStore.attach_closing_line() is UNIQUE(trace_id, market,
selection_descriptor); re-ingesting the same file is a no-op for already-attached
selections.

Workflow:
    1. Scan var/inbox/closing_lines/*.json (non-recursive; processed/ and failed/ skipped).
    2. For each file: parse â†’ validate â†’ apply each `lines` entry via TraceStore â†’
       move file to processed/.
    3. On any error: move file to failed/ with .error.txt sidecar.

Usage:
    omega-ingest-closing-lines
    omega-ingest-closing-lines --inbox <path> --db <path>
    omega-ingest-closing-lines --dry-run

Exit codes:
    0 â€” all files processed (some may have failed; check failed/)
    1 â€” fatal error before scanning (bad args, inbox missing)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.paths import closing_lines_inbox_dir  # noqa: E402
from omega.trace.db import require_sqlite_backend  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("ingest_closing_lines")

_REQUIRED_FILE_FIELDS = ("trace_id", "captured_at", "source", "lines")
_REQUIRED_LINE_FIELDS = ("market", "selection_descriptor", "closing_odds")
_ALLOWED_MARKET_PREFIXES = ("moneyline", "spread", "total", "player_prop:")


def _validate_payload(payload: dict[str, Any]) -> None:
    """Reject malformed payloads before any DB write. Raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON must be an object, got {type(payload).__name__}")
    for field in _REQUIRED_FILE_FIELDS:
        if field not in payload:
            raise ValueError(f"Missing required top-level field: {field!r}")
    if not isinstance(payload["lines"], list) or not payload["lines"]:
        raise ValueError("'lines' must be a non-empty array")

    for i, entry in enumerate(payload["lines"]):
        if not isinstance(entry, dict):
            raise ValueError(f"lines[{i}] must be an object")
        for field in _REQUIRED_LINE_FIELDS:
            if field not in entry:
                raise ValueError(f"lines[{i}] missing required field: {field!r}")
        market = str(entry["market"])
        if not any(market == p or market.startswith(p) for p in _ALLOWED_MARKET_PREFIXES):
            raise ValueError(
                f"lines[{i}].market={market!r} not in allowed set {_ALLOWED_MARKET_PREFIXES}"
            )
        # closing_line is required for spread/total/player_prop, optional for moneyline
        if market != "moneyline" and entry.get("closing_line") is None:
            raise ValueError(f"lines[{i}].closing_line is required for market={market!r}")
        try:
            float(entry["closing_odds"])
        except (TypeError, ValueError):
            raise ValueError(f"lines[{i}].closing_odds must be numeric")


def ingest_file(path: Path, store: TraceStore, dry_run: bool = False) -> tuple[str, int]:
    """Ingest one file. Returns (trace_id, n_lines_attached). Raises on error.

    The file's `lines` array is applied as a single transaction. attach_closing_line
    is itself idempotent on the UNIQUE constraint, so retrying after a partial
    success is safe; we still wrap in BEGIN/COMMIT to fail-fast on schema-level
    issues.
    """
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    _validate_payload(payload)

    trace_id = str(payload["trace_id"])
    captured_at = str(payload["captured_at"])
    source = str(payload["source"])
    lines: list[dict[str, Any]] = payload["lines"]

    # Confirm the trace exists before touching closing_lines; otherwise
    # attach_closing_line will raise per row and leave us with a half-applied file.
    row = store.conn.execute(
        "SELECT trace_id FROM traces WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No trace found with trace_id={trace_id!r}")

    if dry_run:
        return (trace_id, len(lines))

    attached = 0
    store.conn.execute("BEGIN")
    try:
        for entry in lines:
            market = str(entry["market"])
            selection_descriptor = str(entry["selection_descriptor"])
            closing_odds = float(entry["closing_odds"])
            closing_line = (
                float(entry["closing_line"]) if entry.get("closing_line") is not None else None
            )
            store.attach_closing_line(
                trace_id=trace_id,
                market=market,
                selection_descriptor=selection_descriptor,
                closing_odds=closing_odds,
                closing_line=closing_line,
                closing_timestamp=captured_at,
                source=source,
            )
            attached += 1
        store.conn.commit()
    except Exception:
        store.conn.rollback()
        raise

    return (trace_id, attached)


def _move_to(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.stem}.{uuid.uuid4().hex[:8]}{src.suffix}"
    shutil.move(str(src), str(dst))
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest agent-emitted closing-line snapshots into var/omega_traces.db"
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=closing_lines_inbox_dir(),
        help="Directory containing *.json closing-line files (default: var/inbox/closing_lines)",
    )
    parser.add_argument(
        "--db", type=str, default=None, help="SQLite path (default: var/omega_traces.db)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate but skip writes; do not move files",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    require_sqlite_backend("ingest_closing_lines.py")

    inbox: Path = args.inbox
    if not inbox.exists():
        logger.error("Inbox directory does not exist: %s", inbox)
        return 1

    processed_dir = inbox / "processed"
    failed_dir = inbox / "failed"

    files = sorted(p for p in inbox.glob("*.json") if p.is_file())
    if not files:
        logger.info("No new closing-line files in %s", inbox)
        return 0

    store = TraceStore(db_path=args.db)
    ok = 0
    failed = 0
    total_attached = 0

    for path in files:
        try:
            trace_id, n_attached = ingest_file(path, store, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning("FAILED %s: %s", path.name, exc)
            if not args.dry_run:
                moved = _move_to(path, failed_dir)
                error_path = moved.with_suffix(moved.suffix + ".error.txt")
                error_path.write_text(
                    f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
            continue

        ok += 1
        total_attached += n_attached
        logger.info("OK %s -> trace=%s lines=%d", path.name, trace_id, n_attached)
        if not args.dry_run:
            _move_to(path, processed_dir)

    logger.info("Done. %d files ingested (%d lines), %d failed.", ok, total_attached, failed)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())




