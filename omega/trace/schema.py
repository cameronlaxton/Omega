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

Schema version 5 (additive):
- market_snapshots: provider market observations for line movement. Currently
  written by capture jobs but has no consumer in report_calibration.py — the
  compute_market_movement() helper exists but is not wired into any report.
  Captured for future use; wiring deserves its own design pass.

Schema version 6 (additive):
- prop_outcomes: player-prop grading rows, mirroring the outcomes table for
  game-level results. Separate table because the grading shape is different
  (player stat value vs. game score) and overloading outcomes would break
  the 1:1 LEFT JOIN assumed by calibration consumers. One row per
  (trace_id, player_name, stat_type). Source convention matches outcomes
  (e.g. "api:espn_boxscore", "manual:espn_boxscore_YYYYMMDD").

Schema version 7 (additive + cleanup):
- bet_records.session_id (nullable TEXT): mirrors traces.session_id so
  session-scoped audits can join bet_records directly without going through
  traces. Backfilled on migrate from traces via trace_id join.
- BUG-3 cleanup: removes binary 1/0 game-outcome rows that the manual
  backfill incorrectly attached to prop-kind traces. Predicate is narrow
  (home_score IN (0,1) AND away_score IN (0,1) AND linked trace.kind='prop')
  to avoid touching legitimate close-game outcomes.

Design rules:
- Full trace stored as JSON blob to decouple trace evolution from SQLite schema
- Denormalized columns exist for querying only — the blob is source of truth
- Outcomes are a separate table, never mutated into the trace record
- Schema migrations are forward-additive (CREATE TABLE IF NOT EXISTS, ALTER ADD COLUMN)
"""
from __future__ import annotations

import logging

CURRENT_VERSION = 7

_logger = logging.getLogger("omega.trace.schema")

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

SCHEMA_V5 = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id           TEXT PRIMARY KEY,
    league                TEXT NOT NULL,
    provider              TEXT NOT NULL,
    provider_event_id     TEXT NOT NULL,
    home_team             TEXT NOT NULL,
    away_team             TEXT NOT NULL,
    commence_time         TEXT,
    bookmaker             TEXT NOT NULL,
    market                TEXT NOT NULL,
    selection             TEXT NOT NULL,
    player                TEXT,
    point                 REAL,
    price                 REAL NOT NULL,
    snapshot_timestamp    TEXT NOT NULL,
    provider_last_update  TEXT,
    source                TEXT NOT NULL,
    schema_version        INTEGER NOT NULL DEFAULT 1,
    captured_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_event
    ON market_snapshots(league, provider_event_id, market, bookmaker);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_movement
    ON market_snapshots(provider_event_id, market, selection, bookmaker, snapshot_timestamp);
"""

SCHEMA_V6 = """
CREATE TABLE IF NOT EXISTS prop_outcomes (
    prop_outcome_id  TEXT PRIMARY KEY,
    trace_id         TEXT NOT NULL REFERENCES traces(trace_id),
    player_name      TEXT NOT NULL,
    stat_type        TEXT NOT NULL,
    stat_value       REAL NOT NULL,
    line             REAL NOT NULL,
    side             TEXT NOT NULL,
    result           TEXT NOT NULL,
    attached_at      TEXT NOT NULL DEFAULT (datetime('now')),
    source           TEXT NOT NULL DEFAULT 'manual',
    UNIQUE (trace_id, player_name, stat_type)
);

CREATE INDEX IF NOT EXISTS idx_prop_outcomes_trace_id ON prop_outcomes(trace_id);
"""


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


V7_ADD_COLUMN_SQL = "ALTER TABLE bet_records ADD COLUMN session_id TEXT"
V7_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_bet_records_session_id "
    "ON bet_records(session_id)"
)
V7_BACKFILL_SQL = (
    "UPDATE bet_records "
    "SET session_id = (SELECT t.session_id FROM traces t "
    "                  WHERE t.trace_id = bet_records.trace_id) "
    "WHERE session_id IS NULL"
)
# BUG-3 cleanup: the manual outcome backfill dropped placeholder 1/0 game
# scores onto prop-kind traces. The predicate is intentionally narrow — only
# rows where BOTH scores are in {0,1} AND the linked trace blob declares
# kind='prop' get removed. A legitimate 1-0 baseball game on a *game-kind*
# trace is untouched.
V7_CLEANUP_BAD_PROP_OUTCOMES_SQL = (
    "DELETE FROM outcomes "
    "WHERE home_score IN (0, 1) AND away_score IN (0, 1) "
    "  AND trace_id IN ("
    "    SELECT trace_id FROM traces "
    "    WHERE json_extract(full_trace, '$.kind') = 'prop'"
    "  )"
)


def apply_v7_migration(conn) -> int:
    """Idempotently apply V7: bet_records.session_id + BUG-3 outcome cleanup.

    Returns the number of bad outcome rows deleted (for audit logging).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(bet_records)").fetchall()}
    if "session_id" not in cols:
        conn.execute(V7_ADD_COLUMN_SQL)
    conn.execute(V7_INDEX_SQL)
    conn.execute(V7_BACKFILL_SQL)
    cur = conn.execute(V7_CLEANUP_BAD_PROP_OUTCOMES_SQL)
    deleted = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
    conn.commit()
    if deleted:
        _logger.warning(
            "V7 cleanup removed %d binary-placeholder outcome rows attached to "
            "prop-kind traces (BUG-3).", deleted,
        )
    return deleted
