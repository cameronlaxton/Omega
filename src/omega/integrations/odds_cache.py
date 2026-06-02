"""SQLite Caching Layer for the pre-decision odds resolution module."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

# Per-entry-type TTLs (seconds). Success payloads keep the original 15-minute
# window; negative results (unavailable/empty markets) get a short 3-minute soft
# window so an automated loop stops re-hitting the Odds API on the same miss
# without masking a market that goes live shortly after. Event-list payloads
# use a hard 5-minute TTL for both populated and empty slate responses.
SUCCESS_TTL_SECONDS = 900
NEGATIVE_TTL_SECONDS = 180
EVENT_LIST_TTL_SECONDS = 300
_TTL_BY_ENTRY_TYPE = {
    "success": SUCCESS_TTL_SECONDS,
    "negative": NEGATIVE_TTL_SECONDS,
    "event_list": EVENT_LIST_TTL_SECONDS,
}


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
            test_file = base_dir / ".write_test"
            test_file.touch(exist_ok=True)
            test_file.unlink(missing_ok=True)
            return base_dir / "omega_odds_cache.db"
        except Exception:
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
                    market TEXT,
                    market_data TEXT,
                    inserted_at REAL,
                    entry_type TEXT NOT NULL DEFAULT 'success'
                )
            """)
            # Idempotent migration for pre-existing cache DBs missing new columns.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(odds_cache)")}
            if "entry_type" not in cols:
                conn.execute(
                    "ALTER TABLE odds_cache ADD COLUMN entry_type "
                    "TEXT NOT NULL DEFAULT 'success'"
                )
            if "market" not in cols:
                conn.execute("ALTER TABLE odds_cache ADD COLUMN market TEXT")
            conn.commit()

    @staticmethod
    def compute_cache_key(
        league: str,
        market: str,
        home_team: str,
        away_team: str,
        game_date: str,
        player_name: str | None = None,
        player_id: str | None = None,
    ) -> str:
        """Derive a deterministic SHA-256 cache key from query parameters."""
        norm_league = league.strip().upper()
        norm_market = market.strip().lower()
        norm_home = home_team.strip().lower()
        norm_away = away_team.strip().lower()
        norm_date = game_date.strip().lower()

        player_part = ""
        if player_id:
            player_part = f":id:{player_id.strip().lower()}"
        elif player_name:
            from omega.integrations.espn_boxscore import normalize_player_name
            player_part = f":name:{normalize_player_name(player_name)}"

        raw_str = f"{norm_league}{norm_market}{norm_home}{norm_away}{norm_date}{player_part}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_event_list_cache_key(
        league: str,
        commence_time_from: str | None = None,
        commence_time_to: str | None = None,
    ) -> str:
        """Deterministic cache key for active slate discovery."""
        norm_league = league.strip().upper()
        norm_from = (commence_time_from or "").strip()
        norm_to = (commence_time_to or "").strip()
        return f"events:{norm_league}:{norm_from}:{norm_to}"

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieve a cached record if it exists and has not expired.

        TTL is per-entry-type: 900s for ``success`` payloads, 180s for ``negative``
        (unavailable/empty) markers.
        """
        current_time = time.time()
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT market_data, inserted_at, entry_type "
                "FROM odds_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                market_data_str, inserted_at, entry_type = row
                ttl = _TTL_BY_ENTRY_TYPE.get(entry_type, SUCCESS_TTL_SECONDS)
                if current_time - inserted_at <= ttl:
                    try:
                        data = json.loads(market_data_str)
                        if "metadata" not in data:
                            data["metadata"] = []
                        if "source: local_cache" not in data["metadata"]:
                            data["metadata"].append("source: local_cache")
                        if entry_type == "negative" and "source: negative_cache" not in data["metadata"]:
                            data["metadata"].append("source: negative_cache")
                        if entry_type == "event_list" and "cache_kind: event_list" not in data["metadata"]:
                            data["metadata"].append("cache_kind: event_list")
                        return data
                    except json.JSONDecodeError:
                        return None
        return None

    def set(
        self,
        cache_key: str,
        league: str,
        market: str,
        market_data: dict[str, Any],
        *,
        entry_type: str = "success",
    ) -> None:
        """Store a payload and run an append-hook to evict expired records.

        ``market`` scopes the key so game and prop lookups never cross-pollute.
        ``entry_type`` selects the TTL applied on read and eviction: ``success``
        (900s), ``negative`` (180s), or ``event_list`` (300s).
        """
        if entry_type not in _TTL_BY_ENTRY_TYPE:
            raise ValueError(f"entry_type must be one of {sorted(_TTL_BY_ENTRY_TYPE)}")
        current_time = time.time()
        market_data_str = json.dumps(market_data)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO odds_cache
                    (cache_key, league, market, market_data, inserted_at, entry_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, league.upper(), market.strip().lower(), market_data_str, current_time, entry_type),
            )
            # Type-aware eviction: negatives expire at their own shorter TTL.
            conn.execute(
                "DELETE FROM odds_cache WHERE "
                "(entry_type = 'success'  AND ? - inserted_at > ?) OR "
                "(entry_type = 'negative' AND ? - inserted_at > ?) OR "
                "(entry_type = 'event_list' AND ? - inserted_at > ?)",
                (
                    current_time,
                    SUCCESS_TTL_SECONDS,
                    current_time,
                    NEGATIVE_TTL_SECONDS,
                    current_time,
                    EVENT_LIST_TTL_SECONDS,
                ),
            )
            conn.commit()

    def find_by_teams(
        self,
        league: str,
        market: str,
        home_team: str,
        away_team: str,
        player_name: str | None = None,
        player_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Scan the cache for an unexpired success record matching league, market, teams, and player."""
        current_time = time.time()
        norm_league = league.strip().upper()
        norm_market = market.strip().lower()
        norm_home = home_team.strip().lower()
        norm_away = away_team.strip().lower()

        from omega.integrations.espn_boxscore import normalize_player_name
        target_name_norm = normalize_player_name(player_name) if player_name else None
        target_id_norm = player_id.strip().lower() if player_id else None

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT market_data, inserted_at FROM odds_cache "
                "WHERE league = ? AND market = ? AND entry_type = 'success' "
                "AND ? - inserted_at <= ?",
                (norm_league, norm_market, current_time, SUCCESS_TTL_SECONDS),
            )
            rows = cursor.fetchall()
            for market_data_str, _ in rows:
                try:
                    data = json.loads(market_data_str)
                    cached_home = (data.get("home_team") or "").strip().lower()
                    cached_away = (data.get("away_team") or "").strip().lower()
                    if cached_home == norm_home and cached_away == norm_away:
                        if player_name or player_id:
                            quotes = data.get("quotes") or []
                            has_match = False
                            for q in quotes:
                                q_player_name = q.get("player")
                                q_player_id = q.get("player_id")
                                if target_id_norm and q_player_id and str(q_player_id).strip().lower() == target_id_norm:
                                    has_match = True
                                    break
                                if target_name_norm and q_player_name and normalize_player_name(q_player_name) == target_name_norm:
                                    has_match = True
                                    break
                            if not has_match:
                                continue

                        if "metadata" not in data:
                            data["metadata"] = []
                        if "source: local_cache" not in data["metadata"]:
                            data["metadata"].append("source: local_cache")
                        return data
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    def find_by_event_id(
        self,
        league: str,
        market: str,
        event_id: str,
        player_name: str | None = None,
        player_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Scan the cache for an unexpired success record matching league, market, event_id, and player."""
        current_time = time.time()
        norm_league = league.strip().upper()
        norm_market = market.strip().lower()
        norm_event_id = event_id.strip()

        from omega.integrations.espn_boxscore import normalize_player_name
        target_name_norm = normalize_player_name(player_name) if player_name else None
        target_id_norm = player_id.strip().lower() if player_id else None

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT market_data, inserted_at FROM odds_cache "
                "WHERE league = ? AND market = ? AND entry_type = 'success' "
                "AND ? - inserted_at <= ?",
                (norm_league, norm_market, current_time, SUCCESS_TTL_SECONDS),
            )
            rows = cursor.fetchall()
            for market_data_str, _ in rows:
                try:
                    data = json.loads(market_data_str)
                    if str(data.get("event_id", "")).strip() == norm_event_id:
                        if player_name or player_id:
                            quotes = data.get("quotes") or []
                            has_match = False
 