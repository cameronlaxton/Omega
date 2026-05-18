"""
Tests for omega.integrations.espn_boxscore — player-stat extraction from
ESPN summary payloads.

Covers:
- normalize_player_name: accents, suffixes, punctuation
- parse_box_score for NBA (pts/reb/ast)
- parse_box_score for MLB batting (H, R, RBI, HR)
- parse_box_score for MLB pitching (K, IP→outs conversion)
- supported_prop_type recognizes the canonical set
"""
from __future__ import annotations

from omega.integrations.espn_boxscore import (
    MLB_BATTING_KEYS,
    MLB_PITCHING_KEYS,
    NBA_STAT_KEYS,
    normalize_player_name,
    parse_box_score,
    supported_prop_type,
)


class TestNormalizePlayerName:
    def test_strips_accents(self):
        assert normalize_player_name("Luka Dončić") == "luka doncic"

    def test_drops_jr_suffix(self):
        assert normalize_player_name("Marvin Bagley Jr.") == "marvin bagley"
        assert normalize_player_name("Marvin Bagley III") == "marvin bagley"

    def test_handles_apostrophes(self):
        assert normalize_player_name("De'Aaron Fox") == "deaaron fox"

    def test_collapses_whitespace(self):
        assert normalize_player_name("  Jayson    Tatum  ") == "jayson tatum"

    def test_handles_empty(self):
        assert normalize_player_name("") == ""
        assert normalize_player_name(None) == ""


def _nba_fixture() -> dict:
    """Minimal ESPN-shaped NBA summary payload."""
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": "Boston Celtics"},
                    "statistics": [
                        {
                            "name": "starters",
                            "keys": ["MIN", "FG", "PTS", "REB", "AST", "STL", "BLK"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Jayson Tatum"},
                                    "stats": ["38", "10-22", "28", "9", "6", "1", "2"],
                                },
                                {
                                    "athlete": {"displayName": "Jaylen Brown"},
                                    "stats": ["35", "8-15", "22", "5", "3", "0", "1"],
                                },
                            ],
                        }
                    ],
                },
                {
                    "team": {"displayName": "Miami Heat"},
                    "statistics": [
                        {
                            "name": "starters",
                            "keys": ["MIN", "FG", "PTS", "REB", "AST", "STL", "BLK"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Bam Adebayo"},
                                    "stats": ["34", "7-13", "18", "11", "4", "1", "0"],
                                },
                            ],
                        }
                    ],
                },
            ]
        }
    }


class TestParseBoxScoreNBA:
    def test_pts_reb_ast_extracted(self):
        stats = parse_box_score(_nba_fixture(), "NBA")
        assert stats["jayson tatum"]["pts"] == 28.0
        assert stats["jayson tatum"]["reb"] == 9.0
        assert stats["jayson tatum"]["ast"] == 6.0
        assert stats["bam adebayo"]["pts"] == 18.0

    def test_alias_keys_also_work(self):
        """Both 'pts' and 'points' map to the same ESPN PTS column."""
        stats = parse_box_score(_nba_fixture(), "NBA")
        assert stats["jayson tatum"]["pts"] == stats["jayson tatum"]["points"]

    def test_unknown_team_still_parsed(self):
        """Parser doesn't filter by team — caller filters via event_id selection."""
        stats = parse_box_score(_nba_fixture(), "NBA")
        assert "jaylen brown" in stats


def _mlb_fixture() -> dict:
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": "Boston Red Sox"},
                    "statistics": [
                        {
                            "name": "batting",
                            "keys": ["AB", "R", "H", "RBI", "HR", "BB", "SB"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Rafael Devers"},
                                    "stats": ["4", "2", "3", "4", "1", "0", "0"],
                                },
                            ],
                        },
                        {
                            "name": "pitching",
                            "keys": ["IP", "H", "R", "ER", "BB", "K"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Brayan Bello"},
                                    "stats": ["6.2", "5", "2", "2", "1", "8"],
                                },
                            ],
                        },
                    ],
                }
            ]
        }
    }


class TestParseBoxScoreMLB:
    def test_batting_hits_runs_rbi(self):
        stats = parse_box_score(_mlb_fixture(), "MLB")
        assert stats["rafael devers"]["hits"] == 3.0
        assert stats["rafael devers"]["runs"] == 2.0
        assert stats["rafael devers"]["rbi"] == 4.0
        assert stats["rafael devers"]["hr"] == 1.0

    def test_pitching_strikeouts(self):
        stats = parse_box_score(_mlb_fixture(), "MLB")
        assert stats["brayan bello"]["strikeouts"] == 8.0
        assert stats["brayan bello"]["k"] == 8.0
        assert stats["brayan bello"]["er"] == 2.0

    def test_pitching_ip_to_outs_conversion(self):
        """'6.2' IP = 6 innings + 2 outs = 20 outs total."""
        stats = parse_box_score(_mlb_fixture(), "MLB")
        assert stats["brayan bello"]["pitching_outs"] == 20.0
        assert stats["brayan bello"]["outs_recorded"] == 20.0


class TestSupportedPropType:
    def test_nba_pts_supported(self):
        assert supported_prop_type("NBA", "pts") is True
        assert supported_prop_type("NBA", "PTS") is True
        assert supported_prop_type("NBA", "points") is True

    def test_mlb_batting_and_pitching(self):
        assert supported_prop_type("MLB", "hits") is True
        assert supported_prop_type("MLB", "strikeouts") is True

    def test_unsupported_prop_type(self):
        assert supported_prop_type("NBA", "double_double") is False
        assert supported_prop_type("MLB", "no_hitter") is False

    def test_unsupported_league(self):
        assert supported_prop_type("NFL", "pass_yds") is False
