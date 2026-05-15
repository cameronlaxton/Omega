"""
omega.trace.schema — DDL definitions and migration helpers.

Schema version 1:
- traces table: one row per ExecutionTrace (full JSON blob + denormalized query columns)
- outcomes table: attached after initial persistence, references traces(trace_id)
- schema_versions table: tracks applied migrations

Schema version 2 (additive):
- bet_records table: user-confirmed wagers tied to a trace_id (book, line, odds, stake,
  decision_timestamp). Required for CLV computation. One row per (trace_id, market,
  selection_descriptor) — a trace may have multiple bets if a slate.

Schema version 3 (additive):
- closing_lines table: market close snapshots tied to a trace_id + market + selection.
  Used for CLV computation by comparing odds_taken to closing_odds. One row per
  (trace_id, market, selection_descriptor) — same key as bet_records so they line up.

Schema version 4 (additive):
- traces.session_id (nullable TEXT): groups traces produced in one Claude Project
  chat session. Legacy traces stay NULL. Session metadata lives in a JSON sidecar
  under inbox/sessions/<session_id>.json — no separate sessions table.

Design rules:
- Full trace stored as JSON blob to decouple trace evolution from SQLite schema
- Denormalized columns exist for querying only — the blob is source of truth
- Outcomes are a separate table, never mutated into the trace record
- Schema migrations are forward-additive (CREATE TABLE IF NOT EXISTS, ALTER ADD COLUMN)
"""
from __future__ import annotations

CURRENT_VERSION = 4

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id          TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    league            TEXT,
    matchup           TEXT,
    execution_mode    TEXT,
    simulation_seed   INTEGER,
    aggregate_quality REAL,
    predictions       TEXT,
    recommendations   TEXT,
    odds_snapshot     TEXT,
    downgrades        TEXT,
    full_trace        TEXT NOT NULL,
    schema_version    INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_traces_league ON traces(league);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_matchup ON traces(matchup);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id  TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL REFERENCES traces(trace_id),
    home_score  INTEGER NOT NULL,
    away_score  INTEGER NOT NULL,
    result      TEXT NOT NULL,
    attached_at TEXT NOT NULL DEFAULT (datetime('now')),
    source      TEXT NOT NULL DEFAULT 'manual'
);

CREATE INDEX IF NOT EXISTS idx_outcomes_trace_id ON outcomes(trace_id);

CREATE TABLE IF NOT EXISTS schema_versions (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
"""

SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS bet_records (
    bet_id              TEXT PRIMARY KEY,
    trace_id            TEXT NOT NULL REFERENCES traces(trace_id),
    book                TEXT NOT NULL,
    market              TEXT NOT NULL,
    selection           TEXT NOT NULL,
    selection_descriptor TEXT NOT NULL,
    line_taken          REAL,
    odds_taken          REAL NOT NULL,
    stake_units         REAL NOT NULL,
    decision_timestamp  TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    recorded_at         TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (trace_id, market, selection_descriptor)
);

CREATE INDEX IF NOT EXISTS idx_bet_records_trace_id ON bet_records(trace_id);
CREATE INDEX IF NOT EXISTS idx_bet_records_status ON bet_records(status);
"""

SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS closing_lines (
    closing_id            TEXT PRIMARY KEY,
    trace_id              TEXT NOT NULL REFERENCES traces(trace_id),
    market                TEXT NOT NULL,
    selection_descriptor  TEXT NOT NULL,
    closing_line          REAL,
    closing_odds          REAL NOT NULL,
    closing_timestamp     TEXT NOT NULL,
    source                TEXT NOT NULL,
    captured_at           TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (trace_id, market, selection_descriptor)
);

CREATE INDEX IF NOT EXISTS idx_closing_lines_trace_id ON closing_lines(trace_id);
"""

# V4 cannot be expressed as a single executescript() because ALTER TABLE ADD
# COLUMN is non-idempotent in SQLite (errors if the column already exists).
# Apply via the V4 migration helper below.

V4_ADD_COLUMN_SQL = "ALTER TABLE traces ADD COLUMN session_id TEXT"
V4_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_traces_session_id ON traces(session_id)"


def apply_v4_migration(conn) -> None:
    """Idempotently apply V4: add traces.session_id and its index.

    SQLite has no `ADD COLUMN IF NOT EXISTS`; we probe PRAGMA table_info first
    so re-running this on a V4 DB is a no-op.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(traces)").fetchall()}
    if "session_id" not in cols:
        conn.execute(V4_ADD_COLUMN_SQL)
    conn.execute(V4_INDEX_SQL)
    conn.commit()
