"""Writable sidecar store for trace enrichments (the only mutation surface).

A standalone SQLite database, separate from the canonical trace store, holding
the Deep Dive enrichment artifacts and their operator feedback. The schema is
generic enough to later host other write-back tenants (postgame autopsy tags,
operator-action-queue state) without touching ``omega_traces.db``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega.enrichment.schemas import EnrichmentFeedback, EnrichmentRecord, EnrichmentResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_enrichments (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    trace_type TEXT,
    league TEXT,
    market TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    depth TEXT NOT NULL DEFAULT 'deep',
    provider TEXT,
    model TEXT,
    prompt_version TEXT,
    context_pack TEXT,
    result TEXT,
    narrative_md TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_enrichments_trace ON trace_enrichments(trace_id);
CREATE INDEX IF NOT EXISTS idx_trace_enrichments_status ON trace_enrichments(status);
CREATE TABLE IF NOT EXISTS trace_enrichment_feedback (
    id TEXT PRIMARY KEY,
    enrichment_id TEXT NOT NULL,
    user_rating INTEGER,
    feedback_text TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EnrichmentStore:
    """SQLite-backed enrichment sidecar. Writable unless ``read_only=True``."""

    def __init__(self, db_path: str | Path, *, read_only: bool = False) -> None:
        self.db_path = str(db_path)
        self.read_only = read_only
        if read_only:
            self.conn = sqlite3.connect(
                f"file:{Path(self.db_path).as_posix()}?mode=ro", uri=True
            )
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self.conn.executescript(_SCHEMA)
            self.conn.commit()
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    # -- writes ----------------------------------------------------------

    def create(
        self, *, trace_id: str, trace_type: str | None, league: str | None,
        market: str | None, depth: str = "deep",
    ) -> str:
        """Insert a queued enrichment row and return its id."""
        eid = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO trace_enrichments "
            "(id, trace_id, trace_type, league, market, status, depth, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
            (eid, trace_id, trace_type, league, market, depth, _now()),
        )
        self.conn.commit()
        return eid

    def set_running(self, enrichment_id: str) -> None:
        self.conn.execute(
            "UPDATE trace_enrichments SET status='running' WHERE id=?", (enrichment_id,)
        )
        self.conn.commit()

    def set_completed(
        self, enrichment_id: str, *, provider: str, model: str | None,
        prompt_version: str, context_pack: dict[str, Any], result: EnrichmentResult,
        narrative_md: str,
    ) -> None:
        self.conn.execute(
            "UPDATE trace_enrichments SET status='completed', provider=?, model=?, "
            "prompt_version=?, context_pack=?, result=?, narrative_md=?, completed_at=? "
            "WHERE id=?",
            (
                provider, model, prompt_version,
                json.dumps(context_pack), result.model_dump_json(), narrative_md,
                _now(), enrichment_id,
            ),
        )
        self.conn.commit()

    def set_failed(self, enrichment_id: str, error: str) -> None:
        self.conn.execute(
            "UPDATE trace_enrichments SET status='failed', error=?, completed_at=? WHERE id=?",
            (error[:2000], _now(), enrichment_id),
        )
        self.conn.commit()

    def add_feedback(self, enrichment_id: str, feedback: EnrichmentFeedback) -> str:
        fid = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO trace_enrichment_feedback "
            "(id, enrichment_id, user_rating, feedback_text, created_at) VALUES (?, ?, ?, ?, ?)",
            (fid, enrichment_id, feedback.user_rating, feedback.feedback_text, _now()),
        )
        self.conn.commit()
        return fid

    # -- reads -----------------------------------------------------------

    def get(self, enrichment_id: str) -> EnrichmentRecord | None:
        row = self.conn.execute(
            "SELECT * FROM trace_enrichments WHERE id=?", (enrichment_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def list_for_trace(self, trace_id: str, *, limit: int = 50) -> list[EnrichmentRecord]:
        rows = self.conn.execute(
            "SELECT * FROM trace_enrichments WHERE trace_id=? ORDER BY created_at DESC LIMIT ?",
            (trace_id, limit),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def latest_for_trace(self, trace_id: str) -> EnrichmentRecord | None:
        row = self.conn.execute(
            "SELECT * FROM trace_enrichments WHERE trace_id=? ORDER BY created_at DESC LIMIT 1",
            (trace_id,),
        ).fetchone()
        return _row_to_record(row) if row else None

    def trace_id_for(self, enrichment_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT trace_id FROM trace_enrichments WHERE id=?", (enrichment_id,)
        ).fetchone()
        return row["trace_id"] if row else None


def _row_to_record(row: sqlite3.Row) -> EnrichmentRecord:
    result = None
    if row["result"]:
        try:
            result = EnrichmentResult.model_validate_json(row["result"])
        except Exception:  # noqa: BLE001 — a stored artifact must never crash a read
            result = None
    context_pack = None
    if row["context_pack"]:
        try:
            context_pack = json.loads(row["context_pack"])
        except (json.JSONDecodeError, TypeError):
            context_pack = None
    return EnrichmentRecord(
        id=row["id"],
        trace_id=row["trace_id"],
        trace_type=row["trace_type"],
        league=row["league"],
        market=row["market"],
        status=row["status"],
        depth=row["depth"],
        provider=row["provider"],
        model=row["model"],
        prompt_version=row["prompt_version"],
        context_pack=context_pack,
        result=result,
        narrative_md=row["narrative_md"],
        error=row["error"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )
