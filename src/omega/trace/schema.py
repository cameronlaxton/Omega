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
  under var/inbox/sessions/<session_id>.json — no separate sessions table.

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

Schema version 8 (additive + cleanup):
- outcomes.trace_id uniqueness: game outcomes are one row per trace. Existing
  duplicate rows are collapsed to the earliest row before the unique index is
  added. Re-grading requires explicit row deletion first.

Schema version 9 (additive):
- evidence_signals: one row per structured EvidenceSignal carried on a trace.
  Exploded out of input_snapshot.evidence at persist time so retrospective
  scoring can JOIN reasoning signals to outcomes without parsing the full_trace
  JSON blob. The blob remains source of truth; this table is a query aid.
- signal_performance: retrospective scoring aggregates keyed by
  (signal_type, source, obs_window, league, dataset_hash). dataset_hash +
  scored_at make every scoring run an attributable, non-clobbering record.

Schema version 10 (additive):
- simulation_distributions: queryable summaries and generator provenance for
  deterministic simulation outputs. Universal summary columns stay typed;
  family-specific generator parameters live in versioned JSON so future
  distributions do not require schema churn.
- v_distribution_outcomes: raw join substrate for dynamic metric computation
  (CRPS/Brier/etc. are recomputed by versioned report code, not stored here).

Schema version 11 (additive):
- early_market_snapshots: low-liquidity early-line captures for leagues flagged
  liquidity_profile="low" (e.g. WNBA). Deliberately SEPARATE from closing_lines
  because early lines move violently on sharp action and do not reflect closing
  probability. The canonical CLV computation reads ONLY closing_lines and never
  joins this table, so blending phantom early EV into CLV is impossible by
  construction. The calibration fitter excludes early-market-tagged traces by
  default (opt-in forces a separate context_slice="early_market_low_liq"). No FK
  to traces — captures can land before a trace is persisted, matching the
  market_snapshots pattern. (Phase 7 red-team finding 4.)

Schema version 12 (additive):
- trace_qa_verdicts: one row per trace recording the trace-scoped quality-gate
  verdict computed at ingest (see omega/trace/session_sidecar.py
  quality_gate_verdict_for_trace). It is an audit/query aid only — the canonical
  calibration-eligibility flag stays trace_quality.calibration_eligible inside
  the full_trace JSON blob. A "fail" verdict is reconciled into that flag at
  ingest; this table records HOW the verdict was reached (trace_id match,
  timestamp window, pre-trace fatal, or conservative session fallback) so an
  operator can tell a trace-specific failure from a session-wide fallback.

Schema version 13 (additive):
- bet_ledger: dollar-denominated, flat-stake bet log for PnL/ROI tracking and a
  future betting dashboard. One row per (trace_id, market, selection_descriptor),
  recording the engine's RECOMMENDED selection (not a user-confirmed wager) as a
  bet at a fixed stake. Deliberately SEPARATE from bet_records: bet_records is the
  units-based CLV substrate meaning "what the user actually wagered" and drives
  CLV + live prop-outcome fetching, so it must never see auto-logged/backfilled
  phantom bets. bet_ledger carries dollar stake_amount, payout_amount, net_pnl,
  bankroll_at_open, and a provenance column (backfill | engine_auto |
  user_confirmed) so dashboard rows are attributable. Denormalized slice columns
  (league, sport, matchup, bookmaker, odds, line, bet_date) exist for dashboard
  querying only; the full_trace JSON blob remains the source of truth and is
  joined back via trace_id. v_bet_ledger_dashboard adds per-bet return_pct and an
  odds bucket for slicing.

Schema version 14 (consolidation, store-side migration):
- bet_records is REMOVED. Its role (the units-based "what the user actually
  wagered" CLV substrate) is absorbed by bet_ledger via the provenance column:
  the four legacy rows migrate in as provenance='user_confirmed', and all CLV /
  closing-line / prop-sweep / ingest consumers now read bet_ledger (filtering
  provenance='user_confirmed' where true-wager semantics matter). bet_ledger is
  the single source of truth for bets. The migrate-and-drop is performed by
  TraceStore._consolidate_legacy_bet_records() (it needs odds/Kelly helpers, so
  it lives in the store, not here). SCHEMA_V2/V7 remain in history but are only
  (re)applied on DBs that have not yet reached V14.

Design rules:
- Full trace stored as JSON blob to decouple trace evolution from SQLite schema
- Denormalized columns exist for querying only — the blob is source of truth
- Outcomes are a separate table, never mutated into the trace record
- Schema migrations are forward-additive (CREATE TABLE IF NOT EXISTS, ALTER ADD COLUMN)
"""

from __future__ import annotations

import logging

CURRENT_VERSION = 20

# ---------------------------------------------------------------------------
# Version lineage (applied in order by TraceStore._ensure_schema)
# ---------------------------------------------------------------------------
# Two shapes of version exist in this module. Both are forward-additive and
# converge a fresh or older DB to CURRENT_VERSION:
#
#   * SCHEMA_V{n}  — a string of CREATE TABLE/INDEX IF NOT EXISTS, applied via
#                    conn.executescript(). Idempotent by construction.
#   * apply_v{n}_migration(conn) — a Python helper for steps that CANNOT be a
#                    single idempotent executescript (SQLite has no
#                    `ALTER TABLE ADD COLUMN IF NOT EXISTS`, and dedup/DELETE +
#                    UNIQUE-index steps need row-level guards). These probe
#                    PRAGMA/table state first so re-running is a no-op.
#
#   V1  SCHEMA_V1            traces, outcomes, schema_versions
#   V2  SCHEMA_V2            bet_records (CLV)
#   V3  SCHEMA_V3            closing_lines (CLV)
#   V4  apply_v4_migration   traces.session_id column + index   (ALTER)
#   V5  SCHEMA_V5            market_snapshots
#   V6  SCHEMA_V6            prop_outcomes
#   V7  apply_v7_migration   bet_records.session_id + BUG-3 cleanup (ALTER+DELETE)
#   V8  apply_v8_migration   one-outcome-per-trace dedup + UNIQUE index
#   V9  SCHEMA_V9            evidence_signals + signal_performance
#   V10 SCHEMA_V10           simulation_distributions (+ dynamic outcome view)
#   V11 SCHEMA_V11           early_market_snapshots (segregated from CLV)
#   V12 SCHEMA_V12           trace_qa_verdicts (trace-scoped QA audit)
#   V13 SCHEMA_V13           bet_ledger (dollar/PnL bet log; separate from bet_records)
#   V14 (store-side)         consolidate bet_records -> bet_ledger, then DROP it
#   V15 apply_v15_migration  bet_ledger sizing-audit columns (staking/exposure/corr)
#   V16 SCHEMA_V16           priors_xg + priors_dixon_coles (soccer dynamic priors)
#   V17 SCHEMA_V17           priors_tennis + priors_tennis_pressure (tennis dynamic priors)
#   V18 SCHEMA_V18           priors_nfl_dispersion (NFL NB dispersion k w/ shrinkage provenance)
#   V19 SCHEMA_V19           parameter_profiles (governed backend structural-parameter profiles)
#   V20 apply_v20_migration  traces.parameter_profile_ref column (trace-level param provenance)
#
# There is intentionally no SCHEMA_V4/V7/V8 constant — those steps are the
# apply_v{n}_migration helpers above. Bump CURRENT_VERSION and add both the
# DDL/helper and a _record_version() call in store._ensure_schema together.

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
    -- Idempotent per recommendation; record_ledger_bet upgrades pending
    -- engine_auto/backfill rows when the same selection is user_confirmed.
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
V7_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_bet_records_session_id ON bet_records(session_id)"
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

    The bet_records.session_id portion is skipped when bet_records no longer
    exists (it is dropped at V14, consolidated into bet_ledger); the BUG-3
    outcome cleanup is independent of bet_records and always runs.
    """
    has_bet_records = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='bet_records'"
    ).fetchone()
    if has_bet_records:
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
            "prop-kind traces (BUG-3).",
            deleted,
        )
    return deleted


V8_DEDUP_OUTCOMES_SQL = (
    "DELETE FROM outcomes WHERE rowid NOT IN (  SELECT MIN(rowid) FROM outcomes GROUP BY trace_id)"
)
V8_UNIQUE_OUTCOME_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_trace_id_unique ON outcomes(trace_id)"
)


def apply_v8_migration(conn) -> int:
    """Idempotently apply V8: enforce one game outcome per trace.

    Returns the number of duplicate outcome rows deleted. The earliest row is
    preserved because it is the first attached grade and the only deterministic
    choice available without operator review.
    """
    cur = conn.execute(V8_DEDUP_OUTCOMES_SQL)
    deleted = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
    conn.execute(V8_UNIQUE_OUTCOME_SQL)
    conn.commit()
    if deleted:
        _logger.warning(
            "V8 cleanup removed %d duplicate game outcome rows; earliest "
            "outcome per trace was preserved.",
            deleted,
        )
    return deleted


# V9 is purely additive (two new tables) so it can be applied via executescript.
# obs_window is named to avoid SQLite's WINDOW keyword. The full_trace JSON blob
# remains the source of truth — these tables are query aids for retrospective
# evidence scoring.
SCHEMA_V9 = """
CREATE TABLE IF NOT EXISTS evidence_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL REFERENCES traces(trace_id),
    signal_type     TEXT NOT NULL,
    category        TEXT,
    plane           TEXT,
    source          TEXT,
    confidence      REAL,
    obs_window      TEXT,
    direction       TEXT,
    stat_key        TEXT,
    league          TEXT,
    value_json      TEXT,
    applied         INTEGER NOT NULL DEFAULT 0,
    applied_factor  REAL,
    policy_version  TEXT,
    evidence_mode   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_evidence_signals_trace_id ON evidence_signals(trace_id);
CREATE INDEX IF NOT EXISTS idx_evidence_signals_type ON evidence_signals(signal_type, league);

CREATE TABLE IF NOT EXISTS signal_performance (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type         TEXT NOT NULL,
    source              TEXT NOT NULL,
    obs_window          TEXT NOT NULL,
    league              TEXT NOT NULL,
    sample_size         INTEGER NOT NULL,
    direction_correct   INTEGER NOT NULL,
    direction_accuracy  REAL,
    mean_confidence     REAL,
    realized_hit_rate   REAL,
    calibration_gap     REAL,
    brier               REAL,
    dataset_hash        TEXT NOT NULL,
    scored_at           TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (signal_type, source, obs_window, league, dataset_hash)
);

CREATE INDEX IF NOT EXISTS idx_signal_performance_key
    ON signal_performance(signal_type, league);
"""


SCHEMA_V10 = """
CREATE TABLE IF NOT EXISTS simulation_distributions (
    distribution_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id              TEXT NOT NULL REFERENCES traces(trace_id),
    kind                  TEXT,
    league                TEXT,
    target                TEXT NOT NULL,
    market                TEXT,
    stat_key              TEXT,
    distribution_type     TEXT NOT NULL,
    distribution_params   TEXT NOT NULL,
    params_schema_version INTEGER NOT NULL DEFAULT 1,
    sample_mean           REAL,
    sample_std            REAL,
    p10                   REAL,
    p50                   REAL,
    p90                   REAL,
    n_iterations          INTEGER,
    seed                  INTEGER,
    context_hash          TEXT,
    component_version     TEXT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sim_distributions_trace_id
    ON simulation_distributions(trace_id);
CREATE INDEX IF NOT EXISTS idx_sim_distributions_lookup
    ON simulation_distributions(league, kind, market, stat_key);

CREATE VIEW IF NOT EXISTS v_distribution_outcomes AS
SELECT
    d.distribution_id,
    d.trace_id,
    d.kind,
    d.league,
    d.target,
    d.market,
    d.stat_key,
    d.distribution_type,
    d.distribution_params,
    d.params_schema_version,
    d.sample_mean,
    d.sample_std,
    d.p10,
    d.p50,
    d.p90,
    d.n_iterations,
    d.seed,
    d.context_hash,
    d.component_version,
    o.home_score,
    o.away_score,
    o.result AS game_result,
    p.player_name,
    p.stat_type,
    p.stat_value,
    p.line,
    p.side,
    p.result AS prop_result
FROM simulation_distributions d
LEFT JOIN outcomes o ON o.trace_id = d.trace_id
LEFT JOIN prop_outcomes p ON p.trace_id = d.trace_id
    AND (d.stat_key IS NULL OR p.stat_type = d.stat_key);
"""


# V11 is purely additive (one new table) so it can be applied via executescript.
# early_market_snapshots is intentionally a sibling of closing_lines, NOT an
# extension of it: keeping them in separate tables makes it structurally
# impossible for the CLV query (which reads only closing_lines) to pick up early
# low-liquidity captures. liquidity_profile is copied from the league config at
# capture time so a later analysis can slice on it without re-deriving.
SCHEMA_V11 = """
CREATE TABLE IF NOT EXISTS early_market_snapshots (
    early_id              TEXT PRIMARY KEY,
    trace_id              TEXT,
    league                TEXT NOT NULL,
    market                TEXT NOT NULL,
    selection_descriptor  TEXT NOT NULL,
    early_line            REAL,
    early_odds            REAL NOT NULL,
    liquidity_profile     TEXT NOT NULL,
    captured_at           TEXT NOT NULL,
    source                TEXT NOT NULL,
    recorded_at           TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (trace_id, league, market, selection_descriptor, captured_at)
);

CREATE INDEX IF NOT EXISTS idx_early_market_snapshots_trace_id
    ON early_market_snapshots(trace_id);
CREATE INDEX IF NOT EXISTS idx_early_market_snapshots_league
    ON early_market_snapshots(league, captured_at);
"""


# V12 is purely additive (one new table) so it can be applied via executescript.
# trace_qa_verdicts is an audit/query aid: the canonical calibration-eligibility
# flag remains trace_quality.calibration_eligible in the full_trace JSON blob.
# A row records the trace-scoped verdict (and the scope that produced it) so an
# operator can distinguish a trace-specific QA failure from a conservative
# session-wide fallback without parsing the blob.
SCHEMA_V12 = """
CREATE TABLE IF NOT EXISTS trace_qa_verdicts (
    trace_id          TEXT PRIMARY KEY REFERENCES traces(trace_id),
    session_id        TEXT,
    verdict           TEXT NOT NULL,
    scope             TEXT NOT NULL,
    gate_name         TEXT,
    reason            TEXT,
    event_id          TEXT,
    matched_trace_id  TEXT,
    ran_at            TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trace_qa_verdicts_session
    ON trace_qa_verdicts(session_id);
CREATE INDEX IF NOT EXISTS idx_trace_qa_verdicts_verdict
    ON trace_qa_verdicts(verdict, scope);
"""


# V13 is purely additive (one new table + one view) so it can be applied via
# executescript. bet_ledger is a sibling of bet_records, NOT an extension of it:
# keeping them separate makes it structurally impossible for the CLV query and
# the live prop-outcome sweep (both read bet_records) to pick up auto-logged or
# backfilled phantom bets. The money columns are dollar-denominated (bet_records
# is units-based) and recomputed at grade time; the JSON blob stays source of
# truth and is joined back via trace_id.
SCHEMA_V13 = """
CREATE TABLE IF NOT EXISTS bet_ledger (
    ledger_id            TEXT PRIMARY KEY,
    trace_id             TEXT NOT NULL REFERENCES traces(trace_id),
    bet_date             TEXT,
    league               TEXT,
    sport                TEXT,
    matchup              TEXT,
    market               TEXT NOT NULL,
    bookmaker            TEXT NOT NULL DEFAULT 'consensus',
    selection            TEXT NOT NULL,
    selection_descriptor TEXT NOT NULL,
    line                 REAL,
    odds                 REAL NOT NULL,
    stake_amount         REAL NOT NULL DEFAULT 25.0,
    payout_amount        REAL,
    net_pnl              REAL,
    bankroll_at_open     REAL DEFAULT 1000.0,
    status               TEXT NOT NULL DEFAULT 'pending',
    provenance           TEXT NOT NULL,
    decision_timestamp   TEXT NOT NULL,
    graded_at            TEXT,
    session_id           TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (trace_id, market, selection_descriptor)
);

CREATE INDEX IF NOT EXISTS idx_bet_ledger_trace_id ON bet_ledger(trace_id);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_status   ON bet_ledger(status);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_league   ON bet_ledger(league);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_sport    ON bet_ledger(sport);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_book     ON bet_ledger(bookmaker);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_date     ON bet_ledger(bet_date);

CREATE VIEW IF NOT EXISTS v_bet_ledger_dashboard AS
SELECT
    l.ledger_id,
    l.trace_id,
    l.bet_date,
    l.league,
    l.sport,
    l.matchup,
    l.market,
    l.bookmaker,
    l.selection,
    l.selection_descriptor,
    l.line,
    l.odds,
    l.stake_amount,
    l.payout_amount,
    l.net_pnl,
    l.bankroll_at_open,
    l.status,
    l.provenance,
    l.decision_timestamp,
    l.graded_at,
    l.session_id,
    l.created_at,
    CASE WHEN l.status IN ('won', 'lost', 'push', 'void') AND l.stake_amount > 0
         THEN l.net_pnl / l.stake_amount END                    AS return_pct,
    CASE WHEN l.odds > 0 THEN 'underdog' ELSE 'favorite' END     AS odds_side,
    CASE
        WHEN l.odds BETWEEN -110 AND 110 THEN 'pickem'
        WHEN l.odds < -110 THEN 'heavy_fav'
        ELSE 'plus_money' END                                    AS odds_bucket
FROM bet_ledger l;
"""


# V15 adds sizing-audit columns to bet_ledger. Like V4/V7 this is a column-add,
# which cannot use executescript (ALTER ADD COLUMN is non-idempotent in SQLite),
# so the migration probes PRAGMA table_info first.
V15_COLUMNS: tuple[tuple[str, str], ...] = (
    ("staking_policy_id", "TEXT"),
    ("staking_policy_version", "INTEGER"),
    ("exposure_limits_version", "INTEGER"),
    ("sizing_reasons", "TEXT"),  # JSON array of capped_by reasons
    ("correlation_group", "TEXT"),
)


def apply_v15_migration(conn) -> None:
    """Idempotently apply V15: add bet_ledger sizing-audit columns.

    Records which staking policy + exposure limits sized a recommended bet, why
    (the ``capped_by`` reasons as a JSON array), and its correlation group, so a
    bet's sizing is auditable. All columns are nullable — pre-V15 rows and bets
    logged without a portfolio sizing decision simply carry NULL. Re-running on a
    V15 DB is a no-op.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(bet_ledger)").fetchall()}
    for name, sql_type in V15_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE bet_ledger ADD COLUMN {name} {sql_type}")
    conn.commit()


# V16: soccer dynamic-prior tables (Phase 7 M2). priors_dixon_coles holds the
# per-competition Dixon-Coles rho fits produced by omega-fit-dixon-coles; the
# gatherer injects the production row's rho into request.prior_payload and the
# soccer backend fails closed when none exists. priors_xg holds team attack/
# defense xG aggregates from the StatsBomb/Understat/FBref adapters, keyed by
# source so redundancy disagreement is auditable. Both are append-friendly
# upsert targets; flushing them degrades to fail-closed, never to bad numbers.
SCHEMA_V16 = """
CREATE TABLE IF NOT EXISTS priors_dixon_coles (
    profile_id   TEXT NOT NULL,
    rho          REAL NOT NULL,
    n_matches    INTEGER NOT NULL,
    fit_loss     REAL,
    as_of_date   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'candidate',
    source       TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (profile_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_priors_dc_profile_status
    ON priors_dixon_coles(profile_id, status);

CREATE TABLE IF NOT EXISTS priors_xg (
    team         TEXT NOT NULL,
    competition  TEXT NOT NULL,
    season       TEXT NOT NULL,
    xg_for       REAL NOT NULL,
    xg_against   REAL NOT NULL,
    matches      INTEGER NOT NULL,
    source       TEXT NOT NULL,
    as_of_date   TEXT NOT NULL,
    last_updated TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (team, competition, season, source)
);

CREATE INDEX IF NOT EXISTS idx_priors_xg_competition
    ON priors_xg(competition, season);
"""


# V17: tennis dynamic-prior tables (Phase 7 M3). priors_tennis holds surface-
# segmented rolling serve/return point-win rates from the Sackmann match CSVs
# (12-month half-life, computed by omega-refresh-sackmann). priors_tennis_
# pressure holds per-player additive SPW% deltas for the six pressure states,
# fit from Match Charting Project point data by
# omega-fit-tennis-pressure-coefficients; players below the charted-point
# threshold carry source='group_fallback' rows (tour+surface group means) —
# never silent 0.0 deltas. Truncating the pressure table rolls tennis back to
# the flat IID closed-form model (design Part 9).
SCHEMA_V17 = """
CREATE TABLE IF NOT EXISTS priors_tennis (
    player       TEXT NOT NULL,
    tour         TEXT NOT NULL,
    surface      TEXT NOT NULL,
    spw_pct      REAL NOT NULL,
    rpw_pct      REAL NOT NULL,
    n_matches    INTEGER NOT NULL,
    as_of_date   TEXT NOT NULL,
    last_updated TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (player, tour, surface, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_priors_tennis_player
    ON priors_tennis(player, tour, surface);

CREATE TABLE IF NOT EXISTS priors_tennis_pressure (
    player       TEXT NOT NULL,
    tour         TEXT NOT NULL,
    surface      TEXT NOT NULL,
    state        TEXT NOT NULL,
    delta        REAL NOT NULL,
    n_points     INTEGER NOT NULL,
    source       TEXT NOT NULL,
    as_of_date   TEXT NOT NULL,
    last_updated TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (player, tour, surface, state, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_priors_tennis_pressure_player
    ON priors_tennis_pressure(player, tour, surface);
"""


# V18: NFL dispersion priors (Phase 7 M4). priors_nfl_dispersion holds the
# Negative-Binomial dispersion k per (entity, stat_type, season) fit by
# omega-fit-nfl-dispersion with mandatory hierarchical Bayesian shrinkage toward
# (position_group, stat_type) posteriors. nb_k_source (player|position_group|
# league) and nb_k_shrinkage_weight record whether a high-EV tail call was driven
# by genuine player signal or the group prior, so small-sample tail edges are
# auditable. The prop NB backend reads only nb_dispersion_k; all hierarchy lives
# in the offline fitter. Truncating the table degrades props to caller-supplied
# k (fail closed), never to bad numbers (design Part 9).
SCHEMA_V18 = """
CREATE TABLE IF NOT EXISTS priors_nfl_dispersion (
    entity                TEXT NOT NULL,
    stat_type             TEXT NOT NULL,
    season                TEXT NOT NULL,
    position_group        TEXT,
    nb_dispersion_k       REAL NOT NULL,
    nb_k_shrinkage_weight REAL NOT NULL,
    nb_k_source           TEXT NOT NULL,
    n_observations        INTEGER NOT NULL,
    as_of_date            TEXT NOT NULL,
    last_updated          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (entity, stat_type, season, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_priors_nfl_dispersion_entity
    ON priors_nfl_dispersion(entity, stat_type, season);
"""


# V19: backend parameter-profile governance (Phase 8). parameter_profiles holds
# versioned, attributable, promotable structural-parameter bundles for the
# deterministic simulation backends (soccer rho/hca/xg-mapping, NFL dispersion/
# correlation, ...). It generalizes the single-parameter priors_dixon_coles
# rho-profile pattern to a backend's full param set, and is promoted through the
# SAME fail-closed gate engine as calibration profiles
# (omega.core.governance.promotion_gates) — no second promotion path. params_json
# carries the structural knobs the gatherer injects into prior_payload;
# priors_as_of_date pins an immutable per-entity priors snapshot so a later refit
# cannot change what a promoted profile reads; metrics_json carries the RAW
# (pre-calibration) held-out ECE/Brier/log-loss the gates evaluate. The partial
# unique index enforces exactly one production profile per (backend_name,
# competition_bucket); flushing the table degrades a backend to fail-closed
# (status="skipped"), never to bad numbers (design Part 9).
SCHEMA_V19 = """
CREATE TABLE IF NOT EXISTS parameter_profiles (
    profile_id                TEXT PRIMARY KEY,
    schema_version            INTEGER NOT NULL DEFAULT 1,
    version                   INTEGER NOT NULL,
    backend_name              TEXT NOT NULL,
    backend_component_version TEXT NOT NULL,
    competition_bucket        TEXT NOT NULL,
    params_json               TEXT NOT NULL DEFAULT '{}',
    priors_as_of_date         TEXT,
    dataset_manifest_id       TEXT,
    dataset_hash              TEXT NOT NULL,
    sample_size               INTEGER NOT NULL DEFAULT 0,
    metrics_json              TEXT NOT NULL DEFAULT '{}',
    status                    TEXT NOT NULL DEFAULT 'candidate',
    incumbent_id              TEXT,
    promotion_gate_report     TEXT,
    created_at                TEXT NOT NULL DEFAULT (datetime('now')),
    promoted_at               TEXT,
    rejected_at               TEXT,
    reject_reason             TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_parameter_profiles_production
    ON parameter_profiles(backend_name, competition_bucket)
    WHERE status = 'production';

CREATE INDEX IF NOT EXISTS idx_parameter_profiles_lookup
    ON parameter_profiles(backend_name, competition_bucket, status);
"""


# V20: trace-level parameter-profile provenance (Phase 8). Adds a queryable
# traces.parameter_profile_ref column holding the BackendParameterProfile.trace_ref()
# JSON for the governed param set that priced a trace (backend, component version,
# param_profile_id, competition bucket, pinned priors_as_of_date, dataset_hash).
# The full_trace blob already carries these values; surfacing them as a single
# typed column makes a probability attributable to its exact parameter set and lets
# a replay/lab pin from the ref (or emit a loud freshness=unpinned audit when it is
# absent) instead of trusting a live re-read. ALTER ADD COLUMN is non-idempotent in
# SQLite, so apply via the helper which probes PRAGMA table_info first.
V20_ADD_COLUMN_SQL = "ALTER TABLE traces ADD COLUMN parameter_profile_ref TEXT"


def apply_v20_migration(conn) -> None:
    """Idempotently apply V20: add traces.parameter_profile_ref.

    Probes PRAGMA table_info first (SQLite has no ADD COLUMN IF NOT EXISTS) so
    re-running on a V20 DB is a no-op. No index: the column holds a JSON ref blob,
    not a query key, so an index would not help equality lookups inside the JSON.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(traces)").fetchall()}
    if "parameter_profile_ref" not in cols:
        conn.execute(V20_ADD_COLUMN_SQL)
    conn.commit()
