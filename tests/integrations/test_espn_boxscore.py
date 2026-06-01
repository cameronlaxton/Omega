"""
Tests for omega.integrations.espn_boxscore — player-stat extraction from
ESPN summary payloads.

Covers:
- normalize_player_name: accents, suffixes, punctuation
- parse_box_score for NBA (pts/reb/ast)
- parse_box_score for MLB batting (H, R, RBI, HR)
- parse_box_score for MLB pitching (K, IP→outs conversion)
- parse_box_score for soccer (goals/assists/shots/shots_on_target)
- supported_prop_type recognizes the canonical set for all leagues
"""

from __future__ import annotations

from omega.integrations.espn_boxscore import (
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


def _nba_current_fixture() -> dict:
    """Current ESPN summary shape: lowercase keys and blank category names."""
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": "Detroit Pistons"},
                    "statistics": [
                        {
                            "name": None,
                            "keys": [
                                "minutes",
                                "points",
                                "fieldGoalsMade-fieldGoalsAttempted",
                                "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                                "freeThrowsMade-freeThrowsAttempted",
                                "rebounds",
                                "assists",
                                "turnovers",
                                "steals",
                                "blocks",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Cade Cunningham"},
                                    "stats": [
                                        "38",
                                        "29",
                                        "10-22",
                                        "3-8",
                                        "6-7",
                                        "8",
                                        "9",
                                        "4",
                                        "1",
                                        "0",
                                    ],
                                }
                            ],
                        }
                    ],
                }
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

    def test_current_lowercase_payload_shape(self):
        stats = parse_box_score(_nba_current_fixture(), "NBA")
        assert stats["cade cunningham"]["pts"] == 29.0
        assert stats["cade cunningham"]["reb"] == 8.0
        assert stats["cade cunningham"]["ast"] == 9.0
        assert stats["cade cunningham"]["3pm"] == 3.0
        assert stats["cade cunningham"]["threes"] == 3.0
        assert stats["cade cunningham"]["pra"] == 46.0

    def test_pra_zero_derived_correctly(self):
        """PTS=0, REB=3, AST=2 -> pra == 5.0 present; PTS=0, REB=0, AST=0 -> pra == 0.0 present."""
        fixture = {
            "boxscore": {
                "players": [
                    {
                        "team": {"displayName": "Detroit Pistons"},
                        "statistics": [
                            {
                                "name": None,
                                "keys": ["points", "rebounds", "assists"],
                                "athletes": [
                                    {
                                        "athlete": {"displayName": "Zero Pts"},
                                        "stats": ["0", "3", "2"],
                                    },
                                    {
                                        "athlete": {"displayName": "Triple Zero"},
                                        "stats": ["0", "0", "0"],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        }
        stats = parse_box_score(fixture, "NBA")
        assert stats["zero pts"]["pra"] == 5.0
        assert stats["triple zero"]["pra"] == 0.0


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


def _mlb_current_fixture() -> dict:
    """Current ESPN summary shape with blank category names."""
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": "St. Louis Cardinals"},
                    "statistics": [
                        {
                            "name": None,
                            "keys": [
                                "hits-atBats",
                                "atBats",
                                "runs",
                                "hits",
                                "RBIs",
                                "homeRuns",
                                "walks",
                                "strikeouts",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Nolan Arenado"},
                                    "stats": ["2-4", "4", "1", "2", "3", "1", "0", "1"],
                                }
                            ],
                        },
                        {
                            "name": None,
                            "keys": [
                                "fullInnings.partInnings",
                                "hits",
                                "runs",
                                "earnedRuns",
                                "walks",
                                "strikeouts",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Andre Pallante"},
                                    "stats": ["5.1", "6", "2", "2", "1", "4"],
                                }
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

    def test_current_blank_category_payload_shape(self):
        stats = parse_box_score(_mlb_current_fixture(), "MLB")
        assert stats["nolan arenado"]["hits"] == 2.0
        assert stats["nolan arenado"]["rbi"] == 3.0
        assert stats["andre pallante"]["strikeouts"] == 4.0
        assert stats["andre pallante"]["pitching_outs"] == 16.0

    def test_total_bases_derived_from_plays(self):
        fixture = _mlb_fixture()
        fixture["boxscore"]["players"][0]["statistics"][0]["athletes"][0]["athlete"]["id"] = "12345"
        fixture["plays"] = [
            {
                "type": {"text": "Single"},
                "participants": [{"athlete": {"id": "12345"}, "type": "batter"}]
            },
            {
                "type": {"text": "Double"},
                "participants": [{"athlete": {"id": "12345"}, "type": "batter"}]
            },
            {
                "type": {"text": "Home Run"},
                "participants": [{"athlete": {"id": "12345"}, "type": "batter"}]
            }
        ]
        stats = parse_box_score(fixture, "MLB")
        # 1 Single (1) + 1 Double (2) + 1 HR (4) = 7.0
        assert stats["rafael devers"]["total_bases"] == 7.0


class TestSupportedPropType:
    def test_nba_pts_supported(self):
        assert supported_prop_type("NBA", "pts") is True
        assert supported_prop_type("NBA", "PTS") is True
        assert supported_prop_type("NBA", "points") is True
        assert supported_prop_type("NBA", "3pm") is True
        assert supported_prop_type("NBA", "pra") is True

    def test_mlb_batting_and_pitching(self):
        assert supported_prop_type("MLB", "hits") is True
        assert supported_prop_type("MLB", "strikeouts") is True
        assert supported_prop_type("MLB", "total_bases") is True

    def test_unsupported_prop_type(self):
        assert supported_prop_type("NBA", "double_double") is False
        assert supported_prop_type("MLB", "no_hitter") is False

    def test_wnba_supported(self):
        assert supported_prop_type("WNBA", "pts") is True
        assert supported_prop_type("WNBA", "reb") is True
        assert supported_prop_type("WNBA", "pra") is True
        assert supported_prop_type("WNBA", "3pm") is True
        assert supported_prop_type("WNBA", "double_double") is False

    def test_unsupported_league(self):
        assert supported_prop_type("NFL", "pass_yds") is False
        assert supported_prop_type("NHL", "goals") is False


def _wnba_fixture() -> dict:
    """WNBA box score — same ESPN basketball shape as NBA."""
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"displayName": "Las Vegas Aces"},
                    "statistics": [
                        {
                            "name": "starters",
                            "keys": ["MIN", "PTS", "REB", "AST"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "A'ja Wilson"},
                                    "stats": ["34", "28", "11", "4"],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }


class TestParseBoxScoreWNBA:
    def test_pts_reb_ast_extracted(self):
        stats = parse_box_score(_wnba_fixture(), "WNBA")
        player = stats["aja wilson"]
        assert player["pts"] == 28.0
        assert player["reb"] == 11.0
        assert player["ast"] == 4.0

    def test_pra_derived_for_wnba(self):
        stats = parse_box_score(_wnba_fixture(), "WNBA")
        # 28 + 11 + 4 = 43
        assert stats["aja wilson"]["pra"] == 43.0


def _soccer_fixture() -> dict:
    """ESPN soccer summary payload using the real ``rosters`` format (named-stat objects)."""

    def _stats(*pairs):
        return [{"name": k, "value": v} for k, v in pairs]

    return {
        "rosters": [
            {
                "team": {"displayName": "Real Madrid"},
                "roster": [
                    {
                        "athlete": {"displayName": "Kylian Mbappé"},
                        "stats": _stats(
                            ("totalGoals", 2.0),
                            ("goalAssists", 1.0),
                            ("totalShots", 5.0),
                            ("shotsOnTarget", 3.0),
                            ("yellowCards", 0.0),
                            ("redCards", 0.0),
                        ),
                    },
                    {
                        "athlete": {"displayName": "Vinícius Júnior"},
                        "stats": _stats(
                            ("totalGoals", 0.0),
                            ("goalAssists", 1.0),
                            ("totalShots", 3.0),
                            ("shotsOnTarget", 1.0),
                            ("yellowCards", 1.0),
                            ("redCards", 0.0),
                        ),
                    },
                ],
            },
            {
                "team": {"displayName": "Bayern Munich"},
                "roster": [
                    {
                        "athlete": {"displayName": "Harry Kane"},
                        "stats": _stats(
                            ("totalGoals", 1.0),
                            ("goalAssists", 0.0),
                            ("totalShots", 4.0),
                            ("shotsOnTarget", 2.0),
                            ("yellowCards", 0.0),
                            ("redCards", 0.0),
                        ),
                    },
                ],
            },
        ]
    }


class TestParseBoxScoreSoccer:
    def test_goals_assists_shots(self):
        stats = parse_box_score(_soccer_fixture(), "CHAMPIONS_LEAGUE")
        mbappe = stats["kylian mbappe"]
        assert mbappe["goals"] == 2.0
        assert mbappe["assists"] == 1.0
        assert mbappe["shots"] == 5.0
        assert mbappe["shots_on_target"] == 3.0
        assert mbappe["yellow_cards"] == 0.0

    def test_second_player_parsed(self):
        stats = parse_box_score(_soccer_fixture(), "CHAMPIONS_LEAGUE")
        vini = stats["vinicius junior"]
        assert vini["goals"] == 0.0
        assert vini["assists"] == 1.0
        assert vini["yellow_cards"] == 1.0

    def test_away_team_parsed(self):
        stats = parse_box_score(_soccer_fixture(), "CHAMPIONS_LEAGUE")
        kane = stats["harry kane"]
        assert kane["goals"] == 1.0
        assert kane["shots_on_target"] == 2.0

    def test_world_cup_league_code_also_works(self):
        stats = parse_box_score(_soccer_fixture(), "WORLD_CUP")
        assert stats["kylian mbappe"]["goals"] == 2.0

    def test_accented_name_normalized(self):
        stats = parse_box_score(_soccer_fixture(), "CHAMPIONS_LEAGUE")
        assert "kylian mbappe" in stats
        assert "vinicius junior" in stats

    def test_soccer_stats_derived_from_plays_fallback(self):
        fixture = {
            "rosters": [
                {
                    "team": {"displayName": "United States"},
                    "roster": [
                        {
                            "athlete": {"displayName": "Christian Pulisic"},
                            "plays": [
                                {"didScore": True},
                                {"didAssist": True},
                                {"yellowCard": True}
                            ]
                        }
                    ]
                }
            ]
        }
        stats = parse_box_score(fixture, "WORLD_CUP")
        pulisic = stats["christian pulisic"]
        assert pulisic["goals"] == 1.0
        assert pulisic["assists"] == 1.0
        assert pulisic["yellow_cards"] == 1.0
        assert pulisic["red_cards"] == 0.0


class TestSupportedPropTypeSoccer:
    def test_soccer_prop_types_supported(self):
        for league in ("CHAMPIONS_LEAGUE", "WORLD_CUP", "EPL", "MLS"):
            assert supported_prop_type(league, "goals") is True
            assert supported_prop_type(league, "assists") is True
            assert supported_prop_type(league, "shots") is True
            assert supported_prop_type(league, "shots_on_target") is True
            assert supported_prop_type(league, "yellow_cards") is True

    def test_soccer_unsupported_prop_type(self):
        assert supported_prop_type("CHAMPIONS_LEAGUE", "pass_completions") is False
        assert supported_prop_type("WORLD_CUP", "offsides") is False
