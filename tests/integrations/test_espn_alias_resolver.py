"""
Tests for omega.integrations.espn_nba — alias resolution and scoreboard parsing.

Covers:
- Every canonical name in NBA_TEAMS resolves to itself.
- Every alias resolves to its canonical name.
- Unknown strings return None.
- parse_scoreboard extracts a FinalGame from an ESPN fixture.
"""
from __future__ import annotations

from omega.integrations.espn_nba import NBA_TEAMS, canonical_team, parse_scoreboard


class TestCanonicalLookup:
    def test_canonical_names_resolve_to_themselves(self):
        for canonical in NBA_TEAMS:
            assert canonical_team(canonical) == canonical

    def test_aliases_resolve_to_canonical(self):
        for canonical, aliases in NBA_TEAMS.items():
            for alias in aliases:
                assert canonical_team(alias) == canonical, f"alias {alias!r} did not resolve to {canonical}"

    def test_case_insensitive(self):
        assert canonical_team("LAKERS") == "Los Angeles Lakers"
        assert canonical_team("Lakers") == "Los Angeles Lakers"
        assert canonical_team("lakers") == "Los Angeles Lakers"

    def test_unknown_returns_none(self):
        assert canonical_team("ZZZ Not A Team") is None
        assert canonical_team("") is None


class TestParseScoreboard:
    def test_extracts_final_game(self):
        fixture = {
            "events": [{
                "id": "401584321",
                "date": "2026-05-14T23:00Z",
                "competitions": [{
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "112",
                            "team": {"displayName": "Los Angeles Lakers", "abbreviation": "LAL"},
                        },
                        {
                            "homeAway": "away",
                            "score": "108",
                            "team": {"displayName": "Boston Celtics", "abbreviation": "BOS"},
                        },
                    ],
                }],
            }],
        }
        games = parse_scoreboard(fixture)
        assert len(games) == 1
        g = games[0]
        assert g.event_id == "401584321"
        assert g.date == "2026-05-14"
        assert g.home_team == "Los Angeles Lakers"
        assert g.away_team == "Boston Celtics"
        assert g.home_score == 112
        assert g.away_score == 108
        assert g.status == "final"

    def test_skips_events_without_both_competitors(self):
        fixture = {"events": [{"id": "x", "date": "2026-05-14T23:00Z", "competitions": [{}]}]}
        assert parse_scoreboard(fixture) == []

    def test_handles_in_progress_status(self):
        fixture = {
            "events": [{
                "id": "y",
                "date": "2026-05-14T23:00Z",
                "competitions": [{
                    "status": {"type": {"name": "STATUS_IN_PROGRESS"}},
                    "competitors": [
                        {"homeAway": "home", "score": "50", "team": {"displayName": "Boston Celtics"}},
                        {"homeAway": "away", "score": "48", "team": {"displayName": "Miami Heat"}},
                    ],
                }],
            }],
        }
        games = parse_scoreboard(fixture)
        assert len(games) == 1
        assert games[0].status == "in_progress"

    def test_falls_back_to_abbreviation(self):
        fixture = {
            "events": [{
                "id": "z",
                "date": "2026-05-14T23:00Z",
                "competitions": [{
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        # displayName is garbage but abbreviation works
                        {"homeAway": "home", "score": "100", "team": {"displayName": "weird", "abbreviation": "LAL"}},
                        {"homeAway": "away", "score": "95", "team": {"displayName": "also weird", "abbreviation": "BOS"}},
                    ],
                }],
            }],
        }
        games = parse_scoreboard(fixture)
        assert len(games) == 1
        assert games[0].home_team == "Los Angeles Lakers"
        assert games[0].away_team == "Boston Celtics"
