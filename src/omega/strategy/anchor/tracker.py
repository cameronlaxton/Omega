"""
Anchor bet result tracker with CLV computation.

Persists anchor parlay results to SQLite for backtesting and calibration.
Follows the same TraceStore pattern: SQLite + WAL, idempotent writes,
outcome attachment after initial persistence.

Schema version: 1
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from omega.paths import default_trace_db_path

UTC = timezone.utc

logger = logging.getLogger("omega.strategy.anchor.tracker")

ANCHOR_SCHEMA_VERSION = 1

_ANCHOR_BETS_DDL = """
CREATE TABLE IF NOT EXISTS anchor_bets (
    bet_id            TEXT PRIMARY KEY,
    scan_date         TEXT NOT NULL,
    game              TEXT NOT NULL,
    league            TEXT NOT NULL DEFAULT 'NBA',
    sportsbook        TEXT NOT NULL DEFAULT 'BetMGM',
    legs_json         TEXT NOT NULL,
    num_legs          INTEGER NOT NULL,
    odds_taken        REAL NOT NULL,
    odds_close        REAL,
    modeled_true_p    REAL,
    result            TEXT NOT NULL DEFAULT 'PENDING',
    clv_pct           REAL,
    trace_id          TEXT,
    notes             TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    graded_at         TEXT,
    schema_version    INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_anchor_bets_date ON anchor_bets(scan_date);
CREATE INDEX IF NOT EXISTS idx_anchor_bets_result ON anchor_bets(result);
CREATE INDEX IF NOT EXISTS idx_anchor_bets_league ON anchor_bets(league);
"""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class AnchorBetLeg:
    """A single leg in a tracked anchor bet."""

    player: str
    team: str
    stat: str  # pts, reb, ast, 3pm, stl, blk
    threshold: float  # e.g. 20 for "20+ points"
    hit_rate: float  # empirical at scan time
    odds_over: float | None = None  # American odds
    result: str | None = None  # "HIT", "MISS", None


@dataclass
class AnchorBetRecord:
    """A tracked anchor bet for CLV and calibration."""

    bet_id: str
    scan_date: str  # ISO date: "2026-04-13"
    game: str  # "Thunder @ Clippers"
    legs: list[AnchorBetLeg]
    odds_taken: float  # decimal odds at bet time
    league: str = "NBA"
    sportsbook: str = "BetMGM"
    odds_close: float | None = None  # decimal odds at close
    modeled_true_p: float | None = None
    result: Literal["WIN", "LOSS", "PUSH", "PENDING"] = "PENDING"
    clv_pct: float | None = None
    trace_id: str | None = None
    notes: str | None = None

    @staticmethod
    def generate_id() -> str:
        return uuid.uuid4().hex[:12]

    def compute_clv(self) -> float | None:
        """Compute closing line value if close odds are available.

        CLV% = (implied_prob_close - implied_prob_taken) * 100
        Positive = you got a better price than the close.
        """
        if self.odds_close is None or self.odds_close <= 1.0:
            return None
        if self.odds_taken <= 1.0:
            return None
        implied_taken = 1.0 / self.odds_taken
        implied_close = 1.0 / self.odds_close
        clv = (implied_close - implied_taken) * 100.0
        return round(clv, 2)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AnchorBetTracker:
    """SQLite-backed anchor bet persistence and retrieval."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(default_trace_db_path())
        self._db_path = db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
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
        self.conn.executescript(_ANCHOR_BETS_DDL)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_bet(self, record: AnchorBetRecord) -> str:
        """Persist an anchor bet. Idempotent on bet_id.

        Returns bet_id.
        """
        legs_json = json.dumps([asdict(leg) for leg in record.legs])

        self.conn.execute(
            """INSERT OR REPLACE INTO anchor_bets
               (bet_id, scan_date, game, league, sportsbook, legs_json,
                num_legs, odds_taken, odds_close, modeled_true_p,
                result, clv_pct, trace_id, notes, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.bet_id,
                record.scan_date,
                record.game,
                record.league,
                record.sportsbook,
                legs_json,
                len(record.legs),
                record.odds_taken,
                record.odds_close,
                record.modeled_true_p,
                record.result,
                record.clv_pct,
                record.trace_id,
                record.notes,
                ANCHOR_SCHEMA_VERSION,
            ),
        )
        self.conn.commit()
        logger.info("Logged anchor bet %s: %s (%s)", record.bet_id, record.game, record.result)
        return record.bet_id

    def grade_bet(
        self,
        bet_id: str,
        result: Literal["WIN", "LOSS", "PUSH"],
        odds_close: float | None = None,
        notes: str | None = None,
    ) -> None:
        """Attach an outcome to a pending bet and compute CLV."""
        row = self.conn.execute(
            "SELECT bet_id, odds_taken FROM anchor_bets WHERE bet_id = ?", (bet_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No anchor bet found with bet_id={bet_id!r}")

        clv_pct = None
        if odds_close is not None and odds_close > 1.0:
            odds_taken = row["odds_taken"]
            if odds_taken > 1.0:
                clv_pct = round((1.0 / odds_close - 1.0 / odds_taken) * 100.0, 2)

        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """UPDATE anchor_bets
               SET result = ?, odds_close = ?, clv_pct = ?, graded_at = ?, notes = COALESCE(?, notes)
               WHERE bet_id = ?""",
            (result, odds_close, clv_pct, now, notes, bet_id),
        )
        self.conn.commit()
        logger.info("Graded anchor bet %s: %s (CLV: %s%%)", bet_id, result, clv_pct)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_bet(self, bet_id: str) -> AnchorBetRecord | None:
        """Retrieve a single bet by ID."""
        row = self.conn.execute("SELECT * FROM anchor_bets WHERE bet_id = ?", (bet_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def query_bets(
        self,
        league: str | None = None,
        result: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> list[AnchorBetRecord]:
        """Query bets with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if league:
            clauses.append("league = ?")
            params.append(league)
        if result:
            clauses.append("result = ?")
            params.append(result)
        if start_date:
            clauses.append("scan_date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("scan_date <= ?")
            params.append(end_date)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        rows = self.conn.execute(
            f"SELECT * FROM anchor_bets {where} ORDER BY scan_date DESC, created_at DESC LIMIT ?",
            params,
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary_stats(self, league: str | None = None) -> dict[str, Any]:
        """Compute summary statistics for graded bets."""
        bets = self.query_bets(league=league, limit=10000)
        graded = [b for b in bets if b.result in ("WIN", "LOSS", "PUSH")]

        if not graded:
            return {"total": len(bets), "graded": 0}

        wins = sum(1 for b in graded if b.result == "WIN")
        losses = sum(1 for b in graded if b.result == "LOSS")
        pushes = sum(1 for b in graded if b.result == "PUSH")
        win_rate = wins / len(graded) if graded else 0.0

        clvs = [b.clv_pct for b in graded if b.clv_pct is not None]
        avg_clv = sum(clvs) / len(clvs) if clvs else None

        odds_taken_list = [b.odds_taken for b in graded]
        avg_odds = sum(odds_taken_list) / len(odds_taken_list) if odds_taken_list else None

        # ROI: (total_returned - total_staked) / total_staked * 100
        total_staked = len(graded)  # 1 unit per bet
        total_returned = sum(b.odds_taken for b in graded if b.result == "WIN")
        roi_pct = (
            ((total_returned - total_staked) / total_staked * 100.0) if total_staked > 0 else 0.0
        )

        return {
            "total": len(bets),
            "graded": len(graded),
            "pending": len(bets) - len(graded),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(win_rate, 4),
            "avg_odds_taken": round(avg_odds, 3) if avg_odds else None,
            "avg_clv_pct": round(avg_clv, 2) if avg_clv else None,
            "clv_sample_size": len(clvs),
            "roi_pct": round(roi_pct, 2),
        }

    def export_csv(self) -> str:
        """Export all bets as CSV string."""
        bets = self.query_bets(limit=10000)
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(
            [
                "bet_id",
                "scan_date",
                "game",
                "league",
                "sportsbook",
                "num_legs",
                "legs",
                "odds_taken",
                "odds_close",
                "modeled_true_p",
                "result",
                "clv_pct",
                "notes",
            ]
        )

        for b in bets:
            legs_str = " | ".join(f"{leg.player} {leg.threshold:.0f}+ {leg.stat}" for leg in b.legs)
            writer.writerow(
                [
                    b.bet_id,
                    b.scan_date,
                    b.game,
                    b.league,
                    b.sportsbook,
                    len(b.legs),
                    legs_str,
                    b.odds_taken,
                    b.odds_close,
                    b.modeled_true_p,
                    b.result,
                    b.clv_pct,
                    b.notes,
                ]
            )

        return output.getvalue()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> AnchorBetRecord:
        legs_data = json.loads(row["legs_json"])
        legs = [AnchorBetLeg(**ld) for ld in legs_data]

        return AnchorBetRecord(
            bet_id=row["bet_id"],
            scan_date=row["scan_date"],
            game=row["game"],
            league=row["league"],
            sportsbook=row["sportsbook"],
            legs=legs,
            odds_taken=row["odds_taken"],
            odds_close=row["odds_close"],
            modeled_true_p=row["modeled_true_p"],
            result=row["result"],
            clv_pct=row["clv_pct"],
            trace_id=row["trace_id"],
            notes=row["notes"],
        )

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as n FROM anchor_bets").fetchone()
        return row["n"]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
