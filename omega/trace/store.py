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
from pathlib import Path
from typing import Any, Dict, List, Optional

from omega.trace.bet_record import BetRecord
from omega.trace.schema import CURRENT_VERSION, SCHEMA_V1, SCHEMA_V2, SCHEMA_V3

logger = logging.getLogger("omega.trace.store")

_DEFAULT_DB_PATH = "omega_traces.db"


class TraceStore:
    """SQLite-backed trace persistence."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            # Default: repo root
            repo_root = Path(__file__).parent.parent.parent
            db_path = str(repo_root / _DEFAULT_DB_PATH)
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
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

    def persist(self, trace: Dict[str, Any]) -> str:
        """Write a trace to SQLite. Idempotent on trace_id.

        Args:
            trace: Serialized ExecutionTrace dict (must contain trace_id, run_id, timestamp).

        Returns:
            trace_id of the persisted record.

        Raises:
            ValueError: if required fields are missing.
        """
        trace_id = str(trace.get("trace_id", ""))
        run_id = str(trace.get("run_id", ""))
        timestamp = str(trace.get("timestamp", ""))

        if not trace_id or not run_id or not timestamp:
            raise ValueError(
                f"Trace missing required fields: trace_id={trace_id!r}, "
                f"run_id={run_id!r}, timestamp={timestamp!r}"
            )

        full_trace = json.dumps(trace, default=str)

        self.conn.execute(
            """INSERT OR IGNORE INTO traces
               (trace_id, run_id, timestamp, prompt, league, matchup,
                execution_mode, simulation_seed, aggregate_quality,
                predictions, recommendations, odds_snapshot, downgrades,
                full_trace, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                if trace.get("predictions") else None,
                json.dumps(trace.get("recommendations"), default=str)
                if trace.get("recommendations") else None,
                json.dumps(trace.get("odds_snapshot"), default=str)
                if trace.get("odds_snapshot") else None,
                json.dumps(trace.get("downgrades", []), default=str),
                full_trace,
                CURRENT_VERSION,
            ),
        )
        self.conn.commit()
        return trace_id

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

        self.conn.execute(
            """INSERT OR IGNORE INTO bet_records
               (bet_id, trace_id, book, market, selection, selection_descriptor,
                line_taken, odds_taken, stake_units, decision_timestamp, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        self.conn.commit()
        return bet.bet_id

    def get_bet_records(self, trace_id: str) -> List[Dict[str, Any]]:
        """Return all bet records attached to a trace (may be empty)."""
        rows = self.conn.execute(
            """SELECT bet_id, trace_id, book, market, selection, selection_descriptor,
                      line_taken, odds_taken, stake_units, decision_timestamp,
                      status, recorded_at
               FROM bet_records WHERE trace_id = ? ORDER BY recorded_at""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

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
        closing_line: Optional[float],
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

    def get_closing_lines(self, trace_id: str) -> List[Dict[str, Any]]:
        """Return all closing-line snapshots attached to a trace."""
        rows = self.conn.execute(
            """SELECT closing_id, trace_id, market, selection_descriptor,
                      closing_line, closing_odds, closing_timestamp, source, captured_at
               FROM closing_lines WHERE trace_id = ? ORDER BY captured_at""",
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the full trace dict by ID."""
        row = self.conn.execute(
            "SELECT full_trace FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["full_trace"])

    def query_traces(
        self,
        league: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        has_outcome: Optional[bool] = None,
        execution_mode: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
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

        if has_outcome is True:
            clauses.append("o.outcome_id IS NOT NULL")
        elif has_outcome is False:
            clauses.append("o.outcome_id IS NULL")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        sql = f"""
            SELECT t.full_trace, o.outcome_id, o.home_score, o.away_score, o.result
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
            results.append(trace)
        return results

    def get_graded_traces(
        self, league: Optional[str] = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Return traces that have attached outcomes. Used by calibration fitter.

        Each returned dict has a '_outcome' key with the attached outcome.
        """
        return self.query_traces(league=league, has_outcome=True, limit=limit)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def schema_version(self) -> int:
        """Return the current schema version."""
        row = self.conn.execute(
            "SELECT MAX(version) as v FROM schema_versions"
        ).fetchone()
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
