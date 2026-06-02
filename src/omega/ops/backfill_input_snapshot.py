"""
omega.ops.backfill_input_snapshot â€” one-time provenance-first historical fix.

Finds prop traces whose input_snapshot is missing any of the five required
identity fields (player_name, home_team, away_team, game_date, line) and
attempts to recover them from full_trace.result.

Recovery logic:
  - Checks all five target fields against known result aliases.
  - If any field is recoverable: patches input_snapshot, tags identity_status="backfilled".
  - If no fields are recoverable: tags identity_status="missing", leaves input_snapshot alone.
  - Every modified row gets a backfill_meta block embedded in full_trace for audit.

Usage:
    omega-backfill-input-snapshot          # dry-run (default)
    omega-backfill-input-snapshot --apply  # commit changes

Exit codes:
    0 â€” completed (even if some rows were unrecoverable â€” check output)
    1 â€” fatal error
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

UTC = timezone.utc
logger = logging.getLogger("backfill_input_snapshot")

_IDENTITY_FIELDS = ("player_name", "home_team", "away_team", "game_date", "line")

# Maps each target identity field to the aliases we'll look for in result.
_RESULT_ALIASES: dict[str, list[str]] = {
    "player_name": ["player_name", "player"],
    "home_team": ["home_team"],
    "away_team": ["away_team"],
    "game_date": ["game_date"],
    "line": ["line"],
}

_SELECTION_SQL = """
    SELECT trace_id, full_trace FROM traces
    WHERE json_extract(full_trace, '$.kind') = 'prop'
      AND (
          json_extract(full_trace, '$.input_snapshot.player_name') IS NULL
          OR json_extract(full_trace, '$.input_snapshot.home_team') IS NULL
          OR json_extract(full_trace, '$.input_snapshot.away_team') IS NULL
          OR json_extract(full_trace, '$.input_snapshot.game_date') IS NULL
          OR json_extract(full_trace, '$.input_snapshot.line') IS NULL
      )
"""


def _recover_from_result(
    result: dict[str, Any],
    input_snap: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Attempt to recover missing identity fields from result.

    Returns:
        patched_snap: input_snapshot with recovered fields copied in.
        recovered: list of field names successfully recovered.
        unrecoverable: list of fields that could not be found in result.
    """
    patched = dict(input_snap)
    recovered: list[str] = []
    unrecoverable: list[str] = []

    for field in _IDENTITY_FIELDS:
        if patched.get(field):
            continue  # already present â€” skip
        value = None
        for alias in _RESULT_ALIASES[field]:
            candidate = result.get(alias)
            if candidate is not None and candidate != "":
                value = candidate
                break
        if value is not None:
            patched[field] = value
            recovered.append(field)
        else:
            unrecoverable.append(field)

    return patched, recovered, unrecoverable


def _patch_trace(
    trace: dict[str, Any],
    patched_snap: dict[str, Any],
    recovered: list[str],
    unrecoverable: list[str],
) -> dict[str, Any]:
    """Return a copy of trace with patched input_snapshot, updated trace_quality, and backfill_meta."""
    updated = dict(trace)
    updated["input_snapshot"] = patched_snap

    # Determine identity_status
    identity_status = "backfilled" if recovered else "missing"

    # Update trace_quality â€” prefer existing trace_quality, fall back to quality_gate.
    tq = dict(trace.get("trace_quality") or trace.get("quality_gate") or {})
    tq["identity_status"] = identity_status
    updated["trace_quality"] = tq

    # Embed backfill_meta for audit provenance.
    updated["backfill_meta"] = {
        "backfill_script": "backfill_input_snapshot.py",
        "backfilled_at": datetime.now(UTC).isoformat(),
        "fields_recovered": recovered,
        "fields_unrecoverable": unrecoverable,
        "source": "full_trace.result",
    }

    return updated


def run(db_path: str, apply: bool) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    rows = conn.execute(_SELECTION_SQL).fetchall()
    if not rows:
        logger.info("No prop traces with missing identity fields found.")
        conn.close()
        return 0

    logger.info("Found %d prop trace(s) with missing identity fields.", len(rows))

    n_patched = 0
    n_unrecoverable = 0
    n_noop = 0
    updates: list[tuple[str, str]] = []

    for row in rows:
        trace_id: str = row["trace_id"]
        trace: dict[str, Any] = json.loads(row["full_trace"])
        input_snap: dict[str, Any] = trace.get("input_snapshot") or {}
        result: dict[str, Any] = trace.get("result") or {}

        patched_snap, recovered, unrecoverable = _recover_from_result(result, input_snap)

        if not recovered and not unrecoverable:
            # All fields were already present â€” shouldn't happen given the SQL filter.
            n_noop += 1
            logger.debug("trace %s â€” no-op (all fields already present)", trace_id)
            continue

        patched_trace = _patch_trace(trace, patched_snap, recovered, unrecoverable)

        if recovered:
            n_patched += 1
            logger.info(
                "trace %s â€” recovered %s; unrecoverable %s",
                trace_id, recovered, unrecoverable or "none",
            )
        else:
            n_unrecoverable += 1
            logger.warning(
                "trace %s â€” no recoverable fields (result also missing); tagged identity_status=missing",
                trace_id,
            )

        updates.append((json.dumps(patched_trace, default=str), trace_id))

    logger.info(
        "Summary: %d recoverable (patched), %d unrecoverable (tagged missing), %d no-op",
        n_patched, n_unrecoverable, n_noop,
    )

    if not apply:
        logger.info("Dry-run mode â€” no changes committed. Pass --apply to commit.")
        conn.close()
        return 0

    if updates:
        conn.executemany(
            "UPDATE traces SET full_trace = ? WHERE trace_id = ?",
            updates,
        )
        conn.commit()
        logger.info("Committed %d row update(s).", len(updates))

    conn.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-time backfill: copy identity fields from full_trace.result â†’ "
            "full_trace.input_snapshot for prop traces that were written without "
            "going through analyze(). Tags identity_status on all affected rows."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit changes to the database (default: dry-run only)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite database path (default: var/omega_traces.db)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.db:
        db_path = args.db
    else:
        db_path = str(_REPO_ROOT / "var" / "omega_traces.db")

    if not Path(db_path).exists():
        logger.error("Database not found: %s", db_path)
        return 1

    return run(db_path, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())




