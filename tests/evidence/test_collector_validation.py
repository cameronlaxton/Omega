"""Tests for validate_collector_numeric_fields in omega.evidence.collectors.base."""

import math
import pytest

from omega.evidence.collectors.base import validate_collector_numeric_fields


# ---------------------------------------------------------------------------
# Clean data passthrough
# ---------------------------------------------------------------------------

class TestCleanPassthrough:
    def test_team_stat_clean_data(self):
        data = {"off_rating": 112.5, "def_rating": 108.0, "pace": 100.0}
        result = validate_collector_numeric_fields(data, "team_stat")
        assert result == data

    def test_odds_clean_data(self):
        data = {
            "moneyline_home": -150,
            "moneyline_away": 130,
            "spread_home": -3.5,
            "over_under": 224.5,
        }
        result = validate_collector_numeric_fields(data, "odds")
        assert result == data

    def test_non_numeric_keys_preserved(self):
        data = {"off_rating": 112.5, "team_name": "Lakers", "_raw_text": "some text"}
        result = validate_collector_numeric_fields(data, "team_stat")
        assert result["team_name"] == "Lakers"
        assert result["_raw_text"] == "some text"
        assert result["off_rating"] == 112.5


# ---------------------------------------------------------------------------
# String coercion
# ---------------------------------------------------------------------------

class TestStringCoercion:
    def test_string_numeric_coerced_team_stat(self):
        data = {"off_rating": "112.5", "def_rating": "108.0"}
        result = validate_collector_numeric_fields(data, "team_stat")
        assert result["off_rating"] == 112.5
        assert result["def_rating"] == 108.0

    def test_string_numeric_coerced_odds(self):
        data = {"moneyline_home": "-150", "spread_home": "-3.5"}
        result = validate_collector_numeric_fields(data, "odds")
        assert result["moneyline_home"] == -150.0
        assert result["spread_home"] == -3.5


# ---------------------------------------------------------------------------
# Garbage rejection
# ---------------------------------------------------------------------------

class TestGarbageRejection:
    def test_non_numeric_string_dropped(self):
        data = {"off_rating": "not a number", "def_rating": 108.0}
        result = validate_collector_numeric_fields(data, "team_stat")
        assert "off_rating" not in result
        assert result["def_rating"] == 108.0

    def test_nan_dropped(self):
        data = {"off_rating": float("nan")}
        result = validate_collector_numeric_fields(data, "team_stat")
        assert "off_rating" not in result

    def test_inf_dropped(self):
        data = {"moneyline_home": float("inf")}
        result = validate_collector_numeric_fields(data, "odds")
        assert "moneyline_home" not in result

    def test_none_value_for_known_key_dropped(self):
        data = {"off_rating": None, "def_rating": 105.0}
        result = validate_collector_numeric_fields(data, "team_stat")
        assert "off_rating" not in result
        assert result["def_rating"] == 105.0

    def test_list_value_dropped(self):
        data = {"off_rating": [112.5]}
        result = validate_collector_numeric_fields(data, "team_stat")
        assert "off_rating" not in result


# ---------------------------------------------------------------------------
# Non-stat data types pass through unchanged
# ---------------------------------------------------------------------------

class TestNonStatPassthrough:
    def test_schedule_passthrough(self):
        data = {"game_time": "7:00 PM", "venue": "Madison Square Garden"}
        result = validate_collector_numeric_fields(data, "schedule")
        assert result is data  # exact same object, no processing

    def test_injury_passthrough(self):
        data = {"player": "LeBron James", "status": "questionable"}
        result = validate_collector_numeric_fields(data, "injury")
        assert result is data

    def test_news_signal_passthrough(self):
        data = {"headline": "Trade deadline approaching"}
        result = validate_collector_numeric_fields(data, "news_signal")
        assert result is data

    def test_player_stat_validated(self):
        """player_stat IS validated (it feeds sim)."""
        data = {"off_rating": "bad", "name": "Player One"}
        result = validate_collector_numeric_fields(data, "player_stat")
        assert "off_rating" not in result
        assert result["name"] == "Player One"
