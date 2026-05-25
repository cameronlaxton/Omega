"""
omega.trace.store — SQLite-backed trace persistence and retrieval.

TraceStore is the single persistence interface for ExecutionTrace artifacts.
It handles:
- Persist: write a trace dict to SQLite (idempotent on trace_id)
- Query: retrieve traces by league, time range, outcome status
- Attach outcome: link an actual result to a persisted trace
- Graded traces: return traces with attached outcomes (for calibration)

Thread safety: uses one connection per TraceStore instance with WAL mode.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega.trace.bet_record import BetRecord
from omega.trace.market_snapshot import MarketMovement, MarketSnapshot
from omega.trace.schema import (
    CURRENT_VERSION,
    SCHEMA_V1,
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMA_V5,
    SCHEMA_V6,
    SCHEMA_V9,
    SCHEMA_V10,
    apply_v4_migration,
    apply_v7_migration,
    apply_v8_migration,
)

UTC = timezone.utc

logger = logging.getLogger("omega.trace.store")

_DEFAULT_DB_PATH = "omega_traces.db"


class TraceStore:
    """SQLite-backed trace persistence."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            # Default: repo root
            repo_root = Path(__file__).parent.parent.parent
            db_path = str(repo_root / _DEFAULT_DB_PATH)
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._journal_mode: str | None = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            try:
                row = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "SQLite WAL mode unsupported on this mount. Falling back to DELETE mode. "
                    "Concurrency degraded: ensure trace writes and calibration run sequentially. "
                    "(%s)",
                    exc,
                )
                row = self._conn.execute("PRAGMA journal_mode=DELETE").fetchone()
            self._journal_mode = str(row[0]).lower() if row else None
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist and record schema version stamps.

        Migrations are forward-additive: each version's DDL uses CREATE TABLE IF
        NOT EXISTS so a fresh DB and an old-version DB converge to CURRENT_VERSION
        without an explicit migration step.
        """
        # V1: traces, outcomes, schema_versions
        self.conn.executescript(SCHEMA_V1)
        self._record_version(1, "Initial schema: traces, outcomes, schema_versions")

        # V2: bet_records (user-confirmed wagers, for CLV resolution)
        self.conn.executescript(SCHEMA_V2)
        self._record_version(2, "Phase 6d: bet_records table for CLV tracking")

        # V3: closing_lines (market close snapshots for CLV)
        self.conn.executescript(SCHEMA_V3)
        self._record_version(3, "Phase 6e: closing_lines table for CLV computation")

        # V4: traces.session_id (groups traces by Claude Project chat session)
        apply_v4_migration(self.conn)
        self._record_version(4, "Phase 6f: traces.session_id column")

        # V5: market_snapshots (provider observations for line movement)
        self.conn.executescript(SCHEMA_V5)
        self._record_version(5, "Phase 6g: market_snapshots table for line movement")

        # V6: prop_outcomes (player-prop grading; separate from outcomes)
        self.conn.executescript(SCHEMA_V6)
        self._record_version(6, "Phase 6h: prop_outcomes table for player-prop grading")

        # V7: bet_records.session_id (mirrors traces.session_id) + BUG-3 cleanup
        apply_v7_migration(self.conn)
        self._record_version(
            7,
            "Phase 6i: bet_records.session_id column + BUG-3 prop-trace outcome cleanup",
        )

        # V8: one game outcome per trace
        apply_v8_migration(self.conn)
        self._record_version(
            8,
            "Phase 6j: enforce one game outcome row per trace",
        )

        # V9: evidence_signals + signal_performance (structured reasoning loop)
        self.conn.executescript(SCHEMA_V9)
        self._record_version(
            9,
            "Phase 6i: evidence_signals + signal_performance tables",
        )

        # V10: queryable simulation distribution summaries for continuous grading
        self.conn.executescript(SCHEMA_V10)
        self._record_version(
            10,
            "Phase 6k: simulation_distributions + dynamic outcome view",
        )

    def _record_version(self, version: int, description: str) -> None:
        """Idempotently stamp a schema version into schema_versions."""
        existing = self.conn.execute(
            "SELECT version FROM schema_versions WHERE version = ?",
            (version,),
        ).fetchone()
        if not existing:
            self.conn.execute(
                "INSERT INTO schema_versions (version, description) VALUES (?, ?)",
                (version, description),
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def persist(self, trace: dict[str, Any] | Any) -> str:
        """Write a trace to SQLite. Idempotent on trace_id.

        Args:
            trace: PersistableTrace or serialized trace dict containing trace_id, run_id, timestamp.

        Returns:
            trace_id of the persisted record.

        Raises:
            ValueError: if required fields are missing.
        """
        if hasattr(trace, "to_store_record"):
            trace = trace.to_store_record()
        elif hasattr(trace, "model_dump"):
            trace = trace.model_dump(mode="json")

        trace_id = str(trace.get("trace_id", ""))
        run_id = str(trace.get("run_id", ""))
        timestamp = str(trace.get("timestamp", ""))

        if not trace_id or not run_id or not timestamp:
            raise ValueError(
                f"Trace missing required fields: trace_id={trace_id!r}, "
                f"run_id={run_id!r}, timestamp={timestamp!r}"
            )

        full_trace = json.dumps(trace, default=str)

        session_id = trace.get("session_id")
        if session_id is not None:
            session_id = str(session_id) or None

        cur = self.conn.execute(
            """INSERT OR IGNORE INTO traces
               (trace_id, run_id, timestamp, prompt, league, matchup,
                execution_mode, simulation_seed, aggregate_quality,
                predictions, recommendations, odds_snapshot, downgrades,
                full_trace, schema_version, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                run_id,
                timestamp,
                trace.get("prompt", ""),
                trace.get("league"),
                trace.get("matchup"),
                trace.get("execution_mode"),
                trace.get("simulation_seed"),
                trace.get("aggregate_quality", 0.0),
                json.dumps(trace.get("predictions"), default=str)
                if trace.get("predictions")
                else None,
                json.dumps(trace.get("recommendations"), default=str)
                if trace.get("recommendations")
                else None,
                json.dumps(trace.get("odds_snapshot"), default=str)
                if trace.get("odds_snapshot")
                else None,
                json.dumps(trace.get("downgrades", []), default=str),
                full_trace,
                CURRENT_VERSION,
                session_id,
            ),
        )
        # Explode evidence into queryable rows only on a genuine first insert.
        # persist() is idempotent on trace_id; guarding on rowcount keeps the
        # evidence_signals table free of duplicates when a trace is re-persisted.
        if cur.rowcount and cur.rowcount > 0:
            self._write_evidence_signals(trace_id, trace)
            self._write_simulation_distributions(trace_id, trace)
        self.conn.commit()
        return trace_id

    def _write_simulation_distributions(self, trace_id: str, trace: dict[str, Any]) -> int:
        """Explode deterministic distribution summaries into V10 query rows."""
        rows_in = trace.get("simulation_distributions")
        if not isinstance(rows_in, list) or not rows_in:
            result = trace.get("result") or {}
            rows_in = result.get("simulation_distributions") or []
        if not isinstance(rows_in, list) or not rows_in:
            return 0

        rows: list[tuple[Any, ...]] = []
        for item in rows_in:
            if not isinstance(item, dict):
                continue
            dist_type = item.get("distribution_type")
            target = item.get("target")
            if not dist_type or not target:
                continue
            params = item.get("distribution_params") or {}
            rows.append(
                (
                    trace_id,
                    trace.get("kind"),
                    trace.get("league"),
                    target,
                    item.get("market"),
                    item.get("stat_key"),
                    dist_type,
                    json.dumps(params, default=str, sort_keys=True),
                    int(item.get("params_schema_version") or 1),
                    item.get("sample_mean"),
                    item.get("sample_std"),
                    item.get("p10"),
                    item.get("p50"),
                    item.get("p90"),
                    item.get("n_iterations"),
                    item.get("seed", trace.get("simulation_seed")),
                    item.get("context_hash"),
                    item.get("component_version") or trace.get("model_version"),
                )
            )
        if rows:
            self.conn.executemany(
                """INSERT INTO simulation_distributions
                   (trace_id, kind, league, target, market, stat_key,
                    distribution_type, distribution_params, params_schema_version,
                    sample_mean, sample_std, p10, p50, p90, n_iterations, seed,
                    context_hash, component_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def get_simulation_distributions(self, trace_id: str) -> list[dict[str, Any]]:
        """Return V10 simulation distribution rows attached to one trace."""
        rows = self.conn.execute(
            """SELECT distribution_id, trace_id, kind, league, target, market,
                      stat_key, distribution_type, distribution_params,
                      params_schema_version, sample_mean, sample_std, p10, p50,
                      p90, n_iterations, seed, context_hash, component_version,
                      created_at
               FROM simulation_distributions
               WHERE trace_id = ?
               ORDER BY distribution_id""",
            (trace_id,),
        ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            try:
                data["distribution_params"] = json.loads(data["distribution_params"])
            except (TypeError, json.JSONDecodeError):
                data["distribution_params"] = {}
            result.append(data)
        return result

    def _write_evidence_signals(self, trace_id: str, trace: dict[str, Any]) -> int:
        """Explode input_snapshot.evidence into queryable evidence_signals rows.

        Source of truth stays the full_trace JSON blob; this table only exists
        so retrospective scoring can JOIN signals to outcomes. Phase B writes a
        per-signal `evidence_application` list (aligned by index) describing
        whether/how the engine applied each signal; when absent (Phase-A traces
        or no engine apply) every signal is recorded as unapplied.

        Returns the number of evidence rows written.
        """
        input_snap = trace.get("input_snapshot") or {}
        evidence = input_snap.get("evidence") or []
        if not isinstance(evidence, list) or not evidence:
            return 0

        league = trace.get("league")
        application = trace.get("evidence_application")
        if not isinstance(application, list):
            application = []
        trace_evidence_mode = trace.get("evidence_mode")

        rows: list[tuple[Any, ...]] = []
        for idx, sig in enumerate(evidence):
            if not isinstance(sig, dict):
                continue
            app = (
                application[idx]
                if idx < len(application) and isinstance(application[idx], dict)
                else {}
            )
            rows.append(
                (
                    trace_id,
                    sig.get("signal_type"),
                    sig.get("category"),
                    sig.get("plane"),
                    sig.get("source"),
                    sig.get("confidence"),
                    sig.get("window"),
                    sig.get("direction"),
                    sig.get("stat_key"),
                    league,
                    json.dumps(sig.get("value"), default=str),
                    1 if app.get("applied") else 0,
                    app.get("factor"),
                    app.get("policy_version"),
                    app.get("evidence_mode") or trace_evidence_mode,
                )
            )
        if rows:
            self.conn.executemany(
                """INSERT INTO evidence_signals
                   (trace_id, signal_type, category, plane, source, confidence,
                    obs_window, direction, stat_key, league, value_json,
                    applied, applied_factor, policy_version, evidence_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def get_evidence_signals(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all evidence signal rows attached to a trace (may be empty)."""
        rows = self.conn.execute(
            """SELECT id, trace_id, signal_type, category, plane, source,
                      confidence, obs_window, direction, stat_key, league,
                      value_json, applied, applied_factor, policy_version,
                      evidence_mode, created_at
               FROM evidence_signals WHERE trace_id = ? ORDER BY id""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Signal performance (Phase 6i — retrospective evidence scoring)
    # ------------------------------------------------------------------

    def upsert_signal_performance(
        self, rows: list[Any], dataset_hash: str
    ) -> int:
        """Write retrospective signal-performance aggregates for one scoring run.

        ``rows`` are SignalPerformanceRow-shaped objects (see
        omega/strategy/signal_performance.py). ``dataset_hash`` identifies the
        scored dataset; all rows from one run share it and a single ``scored_at``
        timestamp so the report can read the latest run cleanly. Idempotent on
        (signal_type, source, obs_window, league, dataset_hash): re-running the
        same dataset replaces its rows rather than duplicating them.

        Returns the number of rows written.
        """
        scored_at = datetime.now(UTC).isoformat()
        payload = [
            (
                r.signal_type,
                r.source,
                r.obs_window,
                r.league,
                int(r.sample_size),
                int(r.direction_correct),
                float(r.direction_accuracy),
                float(r.mean_confidence),
                float(r.realized_hit_rate),
                float(r.calibration_gap),
                float(r.brier),
                dataset_hash,
                scored_at,
            )
            for r in rows
        ]
        if payload:
            self.conn.executemany(
                """INSERT OR REPLACE INTO signal_performance
                   (signal_type, source, obs_window, league, sample_size,
                    direction_correct, direction_accuracy, mean_confidence,
                    realized_hit_rate, calibration_gap, brier, dataset_hash,
                    scored_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                payload,
            )
            self.conn.commit()
        return len(payload)

    def get_signal_performance(
        self, league: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Return the most recent scoring run's signal-performance rows.

        "Most recent" is the latest ``scored_at`` (optionally scoped to a
        league). Older runs stay in the table for history but are not returned.
        """
        latest = self.conn.execute(
            "SELECT scored_at FROM signal_performance "
            + ("WHERE league = ? " if league else "")
            + "ORDER BY scored_at DESC LIMIT 1",
            (league,) if league else (),
        ).fetchone()
        if latest is None:
            return []

        clauses = ["scored_at = ?"]
        params: list[Any] = [latest["scored_at"]]
        if league:
            clauses.append("league = ?")
            params.append(league)
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT signal_type, source, obs_window, league, sample_size,
                       direction_correct, direction_accuracy, mean_confidence,
                       realized_hit_rate, calibration_gap, brier, dataset_hash,
                       scored_at
                FROM signal_performance
                WHERE {" AND ".join(clauses)}
                ORDER BY sample_size DESC, signal_type
                LIMIT ?""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Outcome attachment
    # ------------------------------------------------------------------

    def attach_outcome(
        self,
        trace_id: str,
        home_score: int,
        away_score: int,
        source: str = "manual",
    ) -> str:
        """Attach an actual outcome to a persisted trace.

        Args:
            trace_id: Must reference an existing trace.
            home_score: Final home team score.
            away_score: Final away team score.
            source: How the outcome was obtained ("manual", "api", "backtest").

        Returns:
            outcome_id of the created record.

        Raises:
            ValueError: if trace_id does not exist.
        """
        # Verify trace exists
        row = self.conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No trace found with trace_id={trace_id!r}")

        existing = self.conn.execute(
            "SELECT outcome_id FROM outcomes WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if existing:
            raise ValueError(
                f"Outcome already attached for trace_id={trace_id!r}; "
                "delete the existing outcome explicitly before re-grading"
            )

        # Determine result
        if home_score > away_score:
            result = "home_win"
        elif away_score > home_score:
            result = "away_win"
        else:
            result = "draw"

        outcome_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO outcomes (outcome_id, trace_id, home_score, away_score, result, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (outcome_id, trace_id, home_score, away_score, result, source),
        )
        self.conn.commit()
        return outcome_id

    # ------------------------------------------------------------------
    # Prop outcome attachment (Phase 6h — player-prop grading)
    # ------------------------------------------------------------------

    def attach_prop_outcome(
        self,
        trace_id: str,
        player_name: str,
        stat_type: str,
        stat_value: float,
        line: float,
        side: str,
        source: str = "manual",
    ) -> str:
        """Attach a graded player-prop outcome to a persisted trace.

        Idempotent on (trace_id, player_name, stat_type): re-attaching returns the
        existing row's id, mirroring closing_lines semantics.

        Args:
            trace_id: Must reference an existing trace.
            player_name: Canonical player name (matches input_snapshot.player_name).
            stat_type: Stat graded (e.g. "points", "rebounds", "hits", "strikeouts").
            stat_value: Actual stat the player produced.
            line: Prop line at decision time.
            side: "over" or "under" — the selection being graded.
            source: How the outcome was obtained (e.g. "manual",
                "api:espn_boxscore", "manual:espn_boxscore_YYYYMMDD").

        Returns:
            prop_outcome_id of the row (existing or newly inserted).

        Raises:
            ValueError: if trace_id does not exist or side is not over/under.
        """
        side_norm = side.lower().strip()
        if side_norm not in ("over", "under"):
            raise ValueError(f"side must be 'over' or 'under', got {side!r}")

        row = self.conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No trace found with trace_id={trace_id!r}")

        existing = self.conn.execute(
            """SELECT prop_outcome_id FROM prop_outcomes
               WHERE trace_id = ? AND player_name = ? AND stat_type = ?""",
            (trace_id, player_name, stat_type),
        ).fetchone()
        if existing:
            return existing["prop_outcome_id"]

        if stat_value == line:
            result = "push"
        elif (side_norm == "over" and stat_value > line) or (
            side_norm == "under" and stat_value < line
        ):
            result = "win"
        else:
            result = "loss"

        prop_outcome_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO prop_outcomes
               (prop_outcome_id, trace_id, player_name, stat_type,
                stat_value, line, side, result, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prop_outcome_id,
                trace_id,
                player_name,
                stat_type,
                float(stat_value),
                float(line),
                side_norm,
                result,
                source,
            ),
        )
        self.conn.commit()
        return prop_outcome_id

    def get_prop_outcomes(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all prop outcomes attached to a trace (may be empty)."""
        rows = self.conn.execute(
            """SELECT prop_outcome_id, trace_id, player_name, stat_type,
                      stat_value, line, side, result, source, attached_at
               FROM prop_outcomes WHERE trace_id = ? ORDER BY attached_at""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Bet records (Phase 6d — CLV substrate)
    # ------------------------------------------------------------------

    def record_bet(self, bet: BetRecord) -> str:
        """Persist a user-confirmed wager. Idempotent on (trace_id, market, selection_descriptor).

        Args:
            bet: A populated BetRecord. trace_id must reference an existing trace.

        Returns:
            bet_id of the persisted row.

        Raises:
            ValueError: if the referenced trace does not exist.
        """
        row = self.conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?", (bet.trace_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No trace found with trace_id={bet.trace_id!r}")

        # session_id is sourced from the linked trace so it stays consistent
        # with traces.session_id without requiring callers to plumb it through
        # the BetRecord model.
        self.conn.execute(
            """INSERT OR IGNORE INTO bet_records
               (bet_id, trace_id, book, market, selection, selection_descriptor,
                line_taken, odds_taken, stake_units, decision_timestamp, status,
                session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       (SELECT session_id FROM traces WHERE trace_id = ?))""",
            (
                bet.bet_id,
                bet.trace_id,
                bet.book,
                bet.market,
                bet.selection,
                bet.selection_descriptor,
                bet.line_taken,
                bet.odds_taken,
                bet.stake_units,
                bet.decision_timestamp,
                bet.status.value,
                bet.trace_id,
            ),
        )
        self.conn.commit()
        return bet.bet_id

    def get_bet_records(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all bet records attached to a trace (may be empty)."""
        rows = self.conn.execute(
            """SELECT bet_id, trace_id, book, market, selection, selection_descriptor,
                      line_taken, odds_taken, stake_units, decision_timestamp,
                      status, recorded_at, session_id
               FROM bet_records WHERE trace_id = ? ORDER BY recorded_at""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def query_ungraded_prop_bet_traces(
        self,
        league: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return traces linked to pending player-prop bet_records that have no
        prop_outcome yet.

        Defense-in-depth for BUG-2 (docs/session_bugs_20260519.md): when the
        agent minted a separate bet-confirmation trace, the bet's trace_id and
        the analysis trace_id end up disjoint, so prop_outcomes attached to the
        analysis trace never reach the bet. This method surfaces the bet's
        trace_id directly so the grading pipeline can attach under it.

        Filters mirror query_traces() for league/time so window semantics are
        consistent. Returns full trace dicts; callers are expected to call
        the same _prop_fields()/grading path as analysis-trace candidates.
        """
        clauses = [
            "b.status = 'pending'",
            "b.market LIKE 'player_prop:%'",
            "NOT EXISTS (SELECT 1 FROM prop_outcomes p WHERE p.trace_id = b.trace_id)",
        ]
        params: list[Any] = []
        if league:
            clauses.append("t.league = ?")
            params.append(league)
        if start:
            clauses.append("t.timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("t.timestamp <= ?")
            params.append(end)
        params.append(limit)

        sql = f"""
            SELECT DISTINCT t.trace_id, t.full_trace
            FROM bet_records b
            JOIN traces t ON t.trace_id = b.trace_id
            WHERE {" AND ".join(clauses)}
            ORDER BY t.timestamp DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, params).fetchall()
        return [json.loads(row["full_trace"]) for row in rows]

    def update_bet_status(self, bet_id: str, status: str) -> None:
        """Mark a bet won/lost/void/push after outcome resolves."""
        self.conn.execute(
            "UPDATE bet_records SET status = ? WHERE bet_id = ?",
            (status, bet_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Closing lines (Phase 6e — CLV resolution)
    # ------------------------------------------------------------------

    def attach_closing_line(
        self,
        trace_id: str,
        market: str,
        selection_descriptor: str,
        closing_odds: float,
        closing_line: float | None,
        closing_timestamp: str,
        source: str,
    ) -> str:
        """Attach a market-close snapshot to a trace + selection.

        Idempotent on (trace_id, market, selection_descriptor): re-running with a
        new source/timestamp leaves the first attached close in place. To force
        an overwrite, delete the row explicitly. This protects against a
        misconfigured cron clobbering a verified close.

        Args:
            trace_id: Must reference an existing trace.
            market: e.g. "moneyline", "spread", "total", "player_prop:pts".
            selection_descriptor: Canonical snake_case form (matches BetRecord).
            closing_odds: American odds at close.
            closing_line: Point/total at close; None for moneyline.
            closing_timestamp: ISO 8601 of the close snapshot.
            source: e.g. "the-odds-api:draftkings".

        Returns:
            closing_id of the row (existing or newly inserted).
        """
        row = self.conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No trace found with trace_id={trace_id!r}")

        existing = self.conn.execute(
            """SELECT closing_id FROM closing_lines
               WHERE trace_id = ? AND market = ? AND selection_descriptor = ?""",
            (trace_id, market, selection_descriptor),
        ).fetchone()
        if existing:
            return existing["closing_id"]

        closing_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            """INSERT INTO closing_lines
               (closing_id, trace_id, market, selection_descriptor,
                closing_line, closing_odds, closing_timestamp, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                closing_id,
                trace_id,
                market,
                selection_descriptor,
                closing_line,
                closing_odds,
                closing_timestamp,
                source,
            ),
        )
        self.conn.commit()
        return closing_id

    def get_closing_lines(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all closing-line snapshots attached to a trace."""
        rows = self.conn.execute(
            """SELECT closing_id, trace_id, market, selection_descriptor,
                      closing_line, closing_odds, closing_timestamp, source, captured_at
               FROM closing_lines WHERE trace_id = ? ORDER BY captured_at""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Market snapshots (Phase 6g - line movement substrate)
    # ------------------------------------------------------------------

    def record_market_snapshot(self, snapshot: MarketSnapshot) -> str:
        """Persist one provider market observation idempotently."""
        snapshot_id = snapshot.stable_id()
        self.conn.execute(
            """INSERT OR IGNORE INTO market_snapshots
               (snapshot_id, league, provider, provider_event_id, home_team,
                away_team, commence_time, bookmaker, market, selection, player,
                point, price, snapshot_timestamp, provider_last_update, source,
                schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot_id,
                snapshot.league.upper(),
                snapshot.provider,
                snapshot.provider_event_id,
                snapshot.home_team,
                snapshot.away_team,
                snapshot.commence_time,
                snapshot.bookmaker,
                snapshot.market,
                snapshot.selection,
                snapshot.player,
                snapshot.point,
                snapshot.price,
                snapshot.snapshot_timestamp,
                snapshot.provider_last_update,
                snapshot.source,
                snapshot.schema_version,
            ),
        )
        self.conn.commit()
        return snapshot_id

    def get_market_snapshots(
        self,
        provider_event_id: str,
        market: str | None = None,
        bookmaker: str | None = None,
        selection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return provider market observations for movement analysis."""
        clauses = ["provider_event_id = ?"]
        params: list[Any] = [provider_event_id]
        if market:
            clauses.append("market = ?")
            params.append(market)
        if bookmaker:
            clauses.append("bookmaker = ?")
            params.append(bookmaker)
        if selection:
            clauses.append("selection = ?")
            params.append(selection)
        rows = self.conn.execute(
            f"""SELECT snapshot_id, league, provider, provider_event_id, home_team,
                       away_team, commence_time, bookmaker, market, selection,
                       player, point, price, snapshot_timestamp,
                       provider_last_update, source, schema_version, captured_at
                FROM market_snapshots
                WHERE {" AND ".join(clauses)}
                ORDER BY snapshot_timestamp""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def compute_market_movement(
        self,
        provider_event_id: str,
        market: str,
        selection: str,
        bookmaker: str,
    ) -> dict[str, Any] | None:
        """Compute simple first-to-last line movement for an exact market row."""
        rows = self.get_market_snapshots(
            provider_event_id=provider_event_id,
            market=market,
            bookmaker=bookmaker,
            selection=selection,
        )
        if len(rows) < 2:
            return None
        first = rows[0]
        last = rows[-1]
        point_delta = None
        if first["point"] is not None and last["point"] is not None:
            point_delta = float(last["point"]) - float(first["point"])
        movement = MarketMovement(
            market=market,
            selection=selection,
            bookmaker=bookmaker,
            first_timestamp=first["snapshot_timestamp"],
            last_timestamp=last["snapshot_timestamp"],
            first_point=first["point"],
            last_point=last["point"],
            first_price=float(first["price"]),
            last_price=float(last["price"]),
            point_delta=point_delta,
            price_delta=float(last["price"]) - float(first["price"]),
        )
        return movement.model_dump()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Retrieve the full trace dict by ID."""
        row = self.conn.execute(
            "SELECT full_trace FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["full_trace"])

    def query_traces(
        self,
        league: str | None = None,
        start: str | None = None,
        end: str | None = None,
        has_outcome: bool | None = None,
        execution_mode: str | None = None,
        limit: int = 100,
        calibration_eligible_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Query traces with optional filters.

        Args:
            league: Filter by league (e.g. "NBA").
            start: ISO timestamp lower bound.
            end: ISO timestamp upper bound.
            has_outcome: True = only graded, False = only ungraded, None = all.
            execution_mode: Filter by mode (e.g. "native_sim").
            limit: Max results (default 100).

        Returns:
            List of trace dicts.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if league:
            clauses.append("t.league = ?")
            params.append(league)
        if start:
            clauses.append("t.timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("t.timestamp <= ?")
            params.append(end)
        if execution_mode:
            clauses.append("t.execution_mode = ?")
            params.append(execution_mode)

        # "Outcome" for query purposes means either a game outcome OR one or
        # more prop outcomes attached to the trace. EXISTS subqueries avoid
        # the row-duplication that would come from LEFT JOINing prop_outcomes
        # (which is 1:N per trace).
        any_outcome_sql = (
            "(EXISTS (SELECT 1 FROM outcomes WHERE outcomes.trace_id = t.trace_id) "
            "OR EXISTS (SELECT 1 FROM prop_outcomes WHERE prop_outcomes.trace_id = t.trace_id))"
        )
        if has_outcome is True:
            clauses.append(any_outcome_sql)
        elif has_outcome is False:
            clauses.append(f"NOT {any_outcome_sql}")

        if calibration_eligible_only:
            # Default-deny: legacy rows without explicit provenance are excluded.
            clauses.append("t.predictions IS NOT NULL")
            clauses.append("json_extract(t.full_trace, '$.result.status') = 'success'")
            clauses.append(
                "json_extract(t.full_trace, '$.trace_quality.calibration_eligible') = 1"
            )
            clauses.append(
                "json_extract(t.full_trace, '$.trace_quality.context_source') = 'provided'"
            )
            clauses.append(
                "json_extract(t.full_trace, '$.trace_quality.identity_status') = 'complete'"
            )

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        # LEFT JOIN outcomes stays for the convenient _outcome attach (1:1 in
        # practice). Prop outcomes are 1:N so we fetch them per-trace below.
        sql = f"""
            SELECT t.trace_id, t.full_trace,
                   o.outcome_id, o.home_score, o.away_score, o.result
            FROM traces t
            LEFT JOIN outcomes o ON t.trace_id = o.trace_id
            {where}
            ORDER BY t.timestamp DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            trace = json.loads(row["full_trace"])
            if row["outcome_id"]:
                trace["_outcome"] = {
                    "outcome_id": row["outcome_id"],
                    "home_score": row["home_score"],
                    "away_score": row["away_score"],
                    "result": row["result"],
                }
            prop_rows = self.get_prop_outcomes(row["trace_id"])
            if prop_rows:
                trace["_prop_outcomes"] = prop_rows
            distribution_rows = self.get_simulation_distributions(row["trace_id"])
            if distribution_rows:
                trace["_simulation_distributions"] = distribution_rows
            results.append(trace)
        return results

    def get_graded_traces(
        self, league: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Return calibration-eligible traces with attached outcomes.

        Only returns traces where the engine ran and produced model predictions
        (predictions IS NOT NULL). Excludes manual:no_engine_run, parlay, and
        pre-6h legacy traces that have outcomes attached but no probability to fit.
        Each returned dict has a '_outcome' key with the attached outcome.
        """
        return self.query_traces(
            league=league,
            has_outcome=True,
            calibration_eligible_only=True,
            limit=limit,
        )

    def get_session_summary(
        self, league: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Aggregate trace counts grouped by session_id. NULL session_ids excluded.

        Used by report_calibration.py to surface per-session metrics.
        """
        clauses = ["t.session_id IS NOT NULL"]
        params: list[Any] = []
        if league:
            clauses.append("t.league = ?")
            params.append(league)
        where = " WHERE " + " AND ".join(clauses)
        params.append(limit)

        rows = self.conn.execute(
            f"""
            SELECT t.session_id,
                   COUNT(*) AS trace_count,
                   SUM(
                       CASE WHEN
                           EXISTS (
                               SELECT 1 FROM outcomes o
                               WHERE o.trace_id = t.trace_id
                           )
                           OR EXISTS (
                               SELECT 1 FROM prop_outcomes p
                               WHERE p.trace_id = t.trace_id
                           )
                       THEN 1 ELSE 0 END
                   ) AS graded_count,
                   MIN(t.timestamp) AS first_ts,
                   MAX(t.timestamp) AS last_ts
            FROM traces t
            {where}
            GROUP BY t.session_id
            ORDER BY last_ts DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def schema_version(self) -> int:
        """Return the current schema version."""
        row = self.conn.execute("SELECT MAX(version) as v FROM schema_versions").fetchone()
        return row["v"] if row and row["v"] else 0

    def count(self) -> int:
        """Return total number of persisted traces."""
        row = self.conn.execute("SELECT COUNT(*) as n FROM traces").fetchone()
        return row["n"]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
