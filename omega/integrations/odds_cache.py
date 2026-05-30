"""SQLite Caching Layer for the pre-decision odds resolution module."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

class OddsCache:
    """Manages transactional caching of Odds API payloads with strict TTL eviction."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or self._resolve_db_path()
        self._init_db()

    def _resolve_db_path(self) -> Path:
        """Resolve database path with fallback to temp directory to avoid FUSE lock issues."""
        base_dir = Path.home() / ".omega" / "runtime"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            # Verify writability before committing to the path
            test_file = base_dir / ".write_test"
            test_file.touch(exist_ok=True)
            test_file.unlink(missing_ok=True)
            return base_dir / "omega_odds_cache.db"
        except Exception:
            # Filesystem Hardening: Fallback to OS temp directory
            temp_dir = Path(tempfile.gettempdir()) / "omega"
            temp_dir.mkdir(parents=True, exist_ok=True)
            return temp_dir / "omega_odds_cache.db"

    def _init_db(self) -> None:
        """Create the table schema if it does not already exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS odds_cache (
                    cache_key TEXT PRIMARY KEY,
                    league TEXT,
                    market_data TEXT,
                    inserted_at REAL
                )
            """)
            conn.commit()

    @staticmethod
    def compute_cache_key(
        league: str,
        market: str,
        home_team: str,
        away_team: str,
        game_date: str
    ) -> str:
        """Derive a deterministic SHA-256 cache key from query parameters."""
        norm_league = league.strip().upper()
        norm_market = market.strip().lower()
        norm_home = home_team.strip().lower()
        norm_away = away_team.strip().lower()
        norm_date = game_date.strip().lower()

        raw_str = f"{norm_league}{norm_market}{norm_home}{norm_away}{norm_date}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieve a cached record if it exists and has not expired (15 minutes)."""
        current_time = time.time()
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT market_data, inserted_at FROM odds_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            if row:
                market_data_str, inserted_at = row
                if current_time - inserted_at <= 900:  # 15 minutes TTL
                    try:
                        data = json.loads(market_data_str)
                        if "metadata" not in data:
                            data["metadata"] = []
                        if "source: local_cache" not in data["metadata"]:
                            data["metadata"].append("source: local_cache")
                        return data
                    except json.JSONDecodeError:
                        return None
        return None

    def set(self, cache_key: str, league: str, market_data: dict[str, Any]) -> None:
        """Store a fresh payload and run an append-hook to evict expired records."""
        current_time = time.time()
        market_data_str = json.dumps(market_data)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO odds_cache (cache_key, league, market_data, inserted_at)
                VALUES (?, ?, ?, ?)
            """, (cache_key, league.upper(), market_data_str, current_time))
            # Append-hook eviction
            conn.execute("DELETE FROM odds_cache WHERE ? - inserted_at > 900", (current_time,))
            conn.commit()

    def find_by_teams(self, league: str, market: str, home_team: str, away_team: str) -> dict[str, Any] | None:
        """Scan the cache for an unexpired record matching the league, teams, and market."""
        current_time = time.time()
        norm_league = league.strip().upper()
        norm_home = home_team.strip().lower()
        norm_away = away_team.strip().lower()

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT market_data, inserted_at FROM odds_cache WHERE league = ? AND ? - inserted_at <= 900",
                (norm_league, current_time)
            )
            rows = cursor.fetchall()
            for market_data_str, _ in rows:
                try:
                    data = json.loads(market_data_str)
                    cached_home = (data.get("home_team") or "").strip().lower()
                    cached_away = (data.get("away_team") or "").strip().lower()
                    if cached_home == norm_home and cached_away == norm_away:
                        if "metadata" not in data:
                            data["metadata"] = []
                        if "source: local_cache" not in data["metadata"]:
                            data["metadata"].append("source: local_cache")
                        return data
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    def find_by_event_id(self, league: str, event_id: str) -> dict[str, Any] | None:
        """Scan the cache for an unexpired record matching the league and event_id."""
        current_time = time.time()
        norm_league = league.strip().upper()
        norm_event_id = event_id.strip()

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT market_data, inserted_at FROM odds_cache WHERE league = ? AND ? - inserted_at <= 900",
                (norm_league, current_time)
            )
            rows = cursor.fetchall()
            for market_data_str, _ in rows:
                try:
                    data = json.loads(market_data_str)
                    if str(data.get("event_id", "")).strip() == norm_event_id:
                        if "metadata" not in data:
                            data["metadata"] = []
                        if "source: local_cache" not in data["metadata"]:
                            data["metadata"].append("source: local_cache")
                        return data
                except (json.JSONDecodeError, KeyError):
                    continue
        return None
