"""Quarantine sink for rejected/ambiguous historical rows.

Bad rows (missing identity, duplicate event keys, props without a line) are
written to ``data/historical/quarantine/<LEAGUE>/rejected_rows.jsonl`` instead of
being silently dropped, so an operator can audit exactly what was excluded.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from omega.historical.contracts import HistoricalEvent

_DEFAULT_ROOT = Path("data/historical/quarantine")


def quarantine_path(league: str, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else _DEFAULT_ROOT
    return base / league.upper() / "rejected_rows.jsonl"


def write_rejected(rows: list[dict], league: str, root: str | Path | None = None) -> Path:
    """Append rejected rows (one JSON object per line). Returns the file path."""
    out = quarantine_path(league, root)
    if not rows:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with out.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({**row, "quarantined_at": ts}, default=str) + "\n")
    return out


def partition_events(
    events: Iterable[HistoricalEvent],
) -> tuple[list[HistoricalEvent], list[dict]]:
    """Split events into (clean, rejected).

    Rejects: missing identity + duplicate event_id (first occurrence kept). Each
    rejected row is a provenance dict with a ``reason_code`` ready for
    :func:`write_rejected`.
    """
    clean: list[HistoricalEvent] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for ev in events:
        if ev.identity_status == "missing":
            rejected.append(_reject(ev, "missing_identity", "unresolved home/away identity"))
            continue
        if ev.event_id in seen:
            rejected.append(
                _reject(ev, "duplicate_event_key", "duplicate event_id within dataset")
            )
            continue
        seen.add(ev.event_id)
        clean.append(ev)
    return clean, rejected


def _reject(ev: HistoricalEvent, reason_code: str, reason: str) -> dict:
    return {
        "source": ev.source_name,
        "source_row_ref": ev.source_row_ref,
        "league": ev.league,
        "event_id": ev.event_id,
        "raw_home": ev.raw_home or ev.home_team,
        "raw_away": ev.raw_away or ev.away_team,
        "reason_code": reason_code,
        "reason": reason,
    }
