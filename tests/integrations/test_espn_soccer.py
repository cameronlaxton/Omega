"""
Unit tests for omega/integrations/espn_soccer.py.

Covers ESPN soccer scoreboard parsing (per-competition, 3-way results),
full-time → "final" status normalization, draw detection at the score level,
team-alias resolution, and the league-slug mapping. No network.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from omega.integrations import espn_soccer  # noqa: E402


def _event(home, away, hs, as_, *, completed=True, state="post", name="STATUS_FULL_TIME"):
    return {
        "id": "SOC-1",
        "date": "2026-05-17T19:00Z",
        "competitions": [
            {
                "status": {"type": {"completed": completed, "state": state, "name": name}},
                "competitors": [
                    {"homeAway": "home", "score": hs, "team": {"displayName": home}},
                    {"homeAway": "away", "score": as_, "team": {"displayName": away}},
                ],
            }
        ],
    }


class TestSlugMapping:
    def test_known_leagues_have_slugs(self):
        assert espn_soccer.espn_slug("EPL") == "eng.1"
        assert espn_soccer.espn_slug("la_liga") == "esp.1"
        assert espn_soccer.espn_slug("CHAMPIONS_LEAGUE") == "uefa.champions"

    def test_unknown_league_is_none(self):
        assert espn_soccer.espn_slug("NFL") is None

    def test_fetch_rejects_unknown_league(self):
        with pytest.raises(ValueError):
            espn_soccer.fetch_scoreboard("2026-05-17", "NFL", url_opener=lambda *a, **k: None)


class TestCanonicalTeam:
    def test_alias_resolves(self):
        assert espn_soccer.canonical_team("Man City") == "Manchester City"
        assert espn_soccer.canonical_team("barca") == "Barcelona"
        assert espn_soccer.canonical_team("PSG") == "Paris Saint-Germain"

    def test_unknown_falls_back_to_stripped_input(self):
        # Unknown clubs still match when both sides use the same display name.
        assert espn_soccer.canonical_team("  Brentford  ") == "Brentford"

    def test_empty_is_none(self):
        assert espn_soccer.canonical_team("") is None
        assert espn_soccer.canonical_team("   ") is None


class TestParseScoreboard:
    def test_home_win_parsed(self):
        games = espn_soccer.parse_scoreboard({"events": [_event("Arsenal", "Chelsea", 2, 1)]}, league="EPL")
        assert len(games) == 1
        g = games[0]
        assert g.home_team == "Arsenal"
        assert g.away_team == "Chelsea"
        assert (g.home_score, g.away_score) == (2, 1)
        assert g.status == "final"
        assert g.league == "EPL"

    def test_draw_is_equal_scores(self):
        games = espn_soccer.parse_scoreboard({"events": [_event("Inter", "Milan", 1, 1)]}, league="SERIE_A")
        g = games[0]
        assert g.home_score == g.away_score == 1
        assert g.status == "final"

    def test_in_progress_not_marked_final(self):
        ev = _event("Arsenal", "Chelsea", 1, 0, completed=False, state="in", name="STATUS_FIRST_HALF")
        games = espn_soccer.parse_scoreboard({"events": [ev]}, league="EPL")
        assert games[0].status != "final"

    def test_missing_competitors_skipped(self):
        ev = {"id": "X", "date": "2026-05-17T19:00Z", "competitions": [{"status": {"type": {"completed": True}}, "competitors": []}]}
        assert espn_soccer.parse_scoreboard({"events": [ev]}, league="EPL") == []

    def test_alias_applied_during_parse(self):
        games = espn_soccer.parse_scoreboard({"events": [_event("Man City", "Spurs", 3, 0)]}, league="EPL")
        assert games[0].home_team == "Manchester City"
        assert games[0].away_team == "Tottenham Hotspur"
