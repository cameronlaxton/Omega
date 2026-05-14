"""
omega.trace.schema — DDL definitions and migration helpers.

Schema version 1:
- traces table: one row per ExecutionTrace (full JSON blob + denormalized query columns)
- outcomes table: attached after initial persistence, references traces(trace_id)
- schema_versions table: tracks applied migrations

Design rules:
- Full trace stored as JSON blob to decouple trace evolution from SQLite schema
- Denormalized columns exist for querying only — the blob is source of truth
- Outcomes are a separate table, never mutated into the trace record
"""
from __future__ import annotations

CURRENT_VERSION = 1

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
