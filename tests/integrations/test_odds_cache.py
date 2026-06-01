"""Tests for the local SQLite pre-decision odds caching layer."""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omega.integrations.odds_cache import OddsCache
from omega.integrations.odds_resolver import resolve_odds


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_odds_cache.db"

def test_cache_schema_initialization(temp_db_path: Path):
    OddsCache(db_path=temp_db_path)
    assert temp_db_path.exists()

    with sqlite3.connect(str(temp_db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(odds_cache)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        assert "cache_key" in columns
        assert "league" in columns
        assert "market_data" in columns
        assert "inserted_at" in columns

def test_deterministic_key_computation():
    key1 = OddsCache.compute_cache_key("NBA", "game", "Los Angeles Lakers", "Boston Celtics", "2026-05-16")
    key2 = OddsCache.compute_cache_key("nba", "GAME", "los angeles lakers", "boston celtics", "2026-05-16 ")
    assert key1 == key2

    key3 = OddsCache.compute_cache_key("NBA", "game", "Los Angeles Lakers", "Boston Celtics", "2026-05-17")
    assert key1 != key3

def test_cache_hit_and_ttl_expiration(temp_db_path: Path):
    cache = OddsCache(db_path=temp_db_path)
    key = cache.compute_cache_key("NBA", "game", "lakers", "celtics", "2026-05-16")
    payload = {"status": "success", "event_id": "evt-1", "metadata": []}

    cache.set(key, "NBA", "game", payload)

    # Hit Verification
    hit = cache.get(key)
    assert hit is not None
    assert hit["event_id"] == "evt-1"
    assert "source: local_cache" in hit["metadata"]

    # Expiry Verification
    with sqlite3.connect(str(temp_db_path)) as conn:
        conn.execute("UPDATE odds_cache SET inserted_at = ? WHERE cache_key = ?", (time.time() - 901, key))
        conn.commit()

    expired = cache.get(key)
    assert expired is None

def test_cache_eviction_on_set(temp_db_path: Path):
    cache = OddsCache(db_path=temp_db_path)
    key_expired = cache.compute_cache_key("NBA", "game", "lakers", "celtics", "2026-05-16")
    key_fresh = cache.compute_cache_key("NBA", "game", "bulls", "knicks", "2026-05-16")

    cache.set(key_expired, "NBA", "game", {"status": "expired"})

    # Manually age the record
    with sqlite3.connect(str(temp_db_path)) as conn:
        conn.execute("UPDATE odds_cache SET inserted_at = ? WHERE cache_key = ?", (time.time() - 950, key_expired))
        conn.commit()

    cache.set(key_fresh, "NBA", "game", {"status": "fresh"})

    with sqlite3.connect(str(temp_db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT cache_key FROM odds_cache")
        keys = [row[0] for row in cursor.fetchall()]
        assert key_expired not in keys
        assert key_fresh in keys

def test_fallback_db_path_resolution(monkeypatch):
    # Simulate directory with no write permissions to test hardening
    original_mkdir = Path.mkdir
    def mock_mkdir(self, *args, **kwargs):
        if ".omega" in str(self):
            raise OSError("Permission denied")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    cache = OddsCache()
    assert "omega" in str(cache.db_path)
    assert tempfile.gettempdir() in str(cache.db_path)

def test_find_by_teams_and_event_id(temp_db_path: Path):
    cache = OddsCache(db_path=temp_db_path)
    payload = {"status": "success", "event_id": "evt-1234", "home_team": "Lakers", "away_team": "Celtics", "metadata": []}
    key = cache.compute_cache_key("NBA", "game", "lakers", "celtics", "2026-05-16")
    cache.set(key, "NBA", "game", payload)

    by_teams = cache.find_by_teams("NBA", "game", "Lakers", "Celtics")
    assert by_teams is not None
    assert by_teams["event_id"] == "evt-1234"

    by_event = cache.find_by_event_id("NBA", "game", "evt-1234")
    assert by_event is not None
    assert by_event["home_team"] == "Lakers"

def test_resolver_uses_cache_avoiding_external_calls(temp_db_path: Path):
    cache = OddsCache(db_path=temp_db_path)
    payload = {
        "status": "success",
        "kind": "game",
        "league": "NBA",
        "event_id": "evt-12345",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "commence_time": "2026-05-16T23:00:00Z",
        "default_bookmaker": "betmgm",
        "line_shopping": False,
        "all_books": False,
        "request_patch": {"odds": {"over_under": 220.5, "markets": []}},
        "quotes": [],
        "skipped_reasons": [],
        "quota": {},
        "metadata": ["source: local_cache"]
    }
    key = cache.compute_cache_key("NBA", "game", "Los Angeles Lakers", "Boston Celtics", "2026-05-16")
    cache.set(key, "NBA", "game", payload)

    mock_client = MagicMock()

    result = resolve_odds(
        kind="game",
        league="NBA",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        commence_time_from="2026-05-16T22:00:00Z",
        client=mock_client,
        cache=cache,
    )

    assert result["status"] == "success"
    assert "source: local_cache" in result["metadata"]
    mock_client.fetch_events.assert_not_called()
    mock_client.fetch_current_event_odds.assert_not_called()


# ---------------------------------------------------------------------------
# Negative-result cache (180s soft TTL)
# ---------------------------------------------------------------------------

def test_negative_entry_180s_ttl(temp_db_path: Path):
    cache = OddsCache(db_path=temp_db_path)
    key = cache.compute_cache_key("NBA", "game", "lakers", "celtics", "2026-05-16")
    payload = {"status": "unavailable", "skipped_reasons": ["no exact event match"]}

    cache.set(key, "NBA", "game", payload, entry_type="negative")

    hit = cache.get(key)
    assert hit is not None
    assert hit["status"] == "unavailable"
    assert "source: negative_cache" in hit["metadata"]

    # Just past the 180s negative TTL -> expired (but still inside the 900s
    # success window, proving the TTL is type-driven, not the old hardcoded 900s).
    with sqlite3.connect(str(temp_db_path)) as conn:
        conn.execute(
            "UPDATE odds_cache SET inserted_at = ? WHERE cache_key = ?",
            (time.time() - 181, key),
        )
        conn.commit()

    assert cache.get(key) is None


def test_success_vs_negative_ttl_differential(temp_db_path: Path):
    cache = OddsCache(db_path=temp_db_path)
    key_success = cache.compute_cache_key("NBA", "game", "lakers", "celtics", "2026-05-16")
    key_negative = cache.compute_cache_key("NBA", "game", "bulls", "knicks", "2026-05-16")

    cache.set(key_success, "NBA", "game", {"status": "success"}, entry_type="success")
    cache.set(key_negative, "NBA", "game", {"status": "unavailable"}, entry_type="negative")

    # Age both to 300s old: success (TTL 900) still live, negative (TTL 180) expired.
    with sqlite3.connect(str(temp_db_path)) as conn:
        conn.execute("UPDATE odds_cache SET inserted_at = ?", (time.time() - 300,))
        conn.commit()

    assert cache.get(key_success) is not None
    assert cache.get(key_negative) is None


def test_negative_eviction_on_set(temp_db_path: Path):
    cache = OddsCache(db_path=temp_db_path)
    key_stale = cache.compute_cache_key("NBA", "game", "lakers", "celtics", "2026-05-16")
    key_fresh = cache.compute_cache_key("NBA", "game", "bulls", "knicks", "2026-05-16")

    cache.set(key_stale, "NBA", "game", {"status": "unavailable"}, entry_type="negative")

    # Age the negative record past its 180s TTL (but well under 900s).
    with sqlite3.connect(str(temp_db_path)) as conn:
        conn.execute(
            "UPDATE odds_cache SET inserted_at = ? WHERE cache_key = ?",
            (time.time() - 200, key_stale),
        )
        conn.commit()

    # Any subsequent write runs the type-aware append-hook eviction.
    cache.set(key_fresh, "NBA", "game", {"status": "success"})

    with sqlite3.connect(str(temp_db_path)) as conn:
        keys = [row[0] for row in conn.execute("SELECT cache_key FROM odds_cache")]
    assert key_stale not in keys
    assert key_fresh in keys


def test_resolver_short_circuits_on_cached_negative(temp_db_path: Path, monkeypatch):
    monkeypatch.setattr(OddsCache, "_resolve_db_path", lambda self: temp_db_path)

    cache = OddsCache(db_path=temp_db_path)
    negative_payload = {
        "status": "unavailable",
        "kind": "game",
        "league": "NBA",
        "event_id": None,
        "home_team": None,
        "away_team": None,
        "default_bookmaker": "betmgm",
        "request_patch": None,
        "quotes": [],
        "skipped_reasons": ["no exact event match for Celtics @ Lakers"],
        "quota": {},
    }
    # Seed under the exact key the resolver will recompute for this request.
    key = cache.compute_cache_key("NBA", "game", "los angeles lakers", "boston celtics", "2026-05-16")
    cache.set(key, "NBA", "game", negative_payload, entry_type="negative")

    mock_client = MagicMock()
    result = resolve_odds(
        kind="game",
        league="NBA",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        commence_time_from="2026-05-16T22:00:00Z",
        client=mock_client,
    )

    assert result["status"] == "unavailable"
    assert "source: negative_cache" in result["metadata"]
    mock_client.fetch_events.assert_not_called()
    mock_client.fetch_current_event_odds.assert_not_called()


class TestPropCacheIdentity:
    def test_two_players_same_game_different_keys_no_cross_serve(self, temp_db_path: Path):
        cache = OddsCache(db_path=temp_db_path)

        key_lebron = cache.compute_cache_key("NBA", "pts", "lakers", "celtics", "2026-05-16", player_name="LeBron James")
        key_tatum = cache.compute_cache_key("NBA", "pts", "lakers", "celtics", "2026-05-16", player_name="Jayson Tatum")

        assert key_lebron != key_tatum

        payload_lebron = {"status": "success", "player": "LeBron James", "quotes": [{"player": "LeBron James", "price": -110}]}
        payload_tatum = {"status": "success", "player": "Jayson Tatum", "quotes": [{"player": "Jayson Tatum", "price": -115}]}

        cache.set(key_lebron, "NBA", "pts", payload_lebron)
        cache.set(key_tatum, "NBA", "pts", payload_tatum)

        res_lebron = cache.get(key_lebron)
        assert res_lebron is not None
        assert res_lebron["player"] == "LeBron James"

        res_tatum = cache.get(key_tatum)
        assert res_tatum is not None
        assert res_tatum["player"] == "Jayson Tatum"

    def test_one_player_with_stable_id(self, temp_db_path: Path):
        cache = OddsCache(db_path=temp_db_path)

        key_by_id = cache.compute_cache_key("NBA", "pts", "lakers", "celtics", "2026-05-16", player_id="pid-123")
        payload = {
            "status": "success",
            "player_id": "pid-123",
            "quotes": [{"player_id": "pid-123", "player": "LeBron James", "price": -110}]
        }

        cache.set(key_by_id, "NBA", "pts", payload)

        # Retrieval with ID succeeds
        hit_by_id = cache.get(key_by_id)
        assert hit_by_id is not None
        assert hit_by_id["player_id"] == "pid-123"

    def test_unmapped_player_normalized_name_fallback_works(self, temp_db_path: Path):
        cache = OddsCache(db_path=temp_db_path)

        # Retrieval using different name casing / spacing / accents Normalization
        key_original = cache.compute_cache_key("NBA", "pts", "lakers", "celtics", "2026-05-16", player_name="Luka Dončić")
        key_normalized = cache.compute_cache_key("NBA", "pts", "lakers", "celtics", "2026-05-16", player_name="luka doncic  ")

        assert key_original == key_normalized

    def test_legacy_playerless_prop_entry_misses_for_player_specific_lookup(self, temp_db_path: Path):
        cache = OddsCache(db_path=temp_db_path)

        # Legacy entry has no player_name/id in its cache key, and no quotes with the player inside
        key_legacy = cache.compute_cache_key("NBA", "pts", "lakers", "celtics", "2026-05-16")
        payload_legacy = {
            "status": "success",
            "quotes": []  # empty/no player info
        }
        cache.set(key_legacy, "NBA", "pts", payload_legacy)

        # Querying with a specific player should miss when using find_by_teams on the legacy record
        res = cache.find_by_teams("NBA", "pts", "lakers", "celtics", player_name="LeBron James")
        assert res is None
