"""Tests for the ESPN WNBA scoreboard integration.

Covers scoreboard parsing against a fixture payload, team-alias resolution, and
the OMEGA_REPLAY_MODE live-fetch guard (no network in unit tests).

References:
  omega/integrations/espn_wnba.py
  omega/integrations/_guards.py
"""

from __future__ import annotations

import pytest

from omega.integrations import espn_wnba
from omega.integrations._guards import OmegaReplayModeError


def _never_called(*_args, **_kwargs):
    raise AssertionError("url_opener should not be called — guard should fire first")


_FIXTURE = {
    "events": [
        {
            "id": "401620001",
            "date": "2026-05-20T23:00Z",
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "88",
                            "team": {"displayName": "Las Vegas Aces", "abbreviation": "LV"},
                        },
                        {
                            "homeAway": "away",
                            "score": "82",
                            "team": {"displayName": "New York Liberty", "abbreviation": "NY"},
                        },
                    ],
                }
            ],
        }
    ]
}


def test_parse_scoreboard_extracts_final_game():
    games = espn_wnba.parse_scoreboard(_FIXTURE)
    assert len(games) == 1
    game = games[0]
    assert game.home_team == "Las Vegas Aces"
    assert game.away_team == "New York Liberty"
    assert game.home_score == 88
    assert game.away_score == 82
    assert game.status == "final"


def test_canonical_team_resolves_aliases():
    assert espn_wnba.canonical_team("aces") == "Las Vegas Aces"
    assert espn_wnba.canonical_team("Liberty") == "New York Liberty"
    assert espn_wnba.canonical_team("GSV") == "Golden State Valkyries"
    assert espn_wnba.canonical_team("not a team xyz") is None


def test_fetch_scoreboard_blocked_in_replay_mode(monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        espn_wnba.fetch_scoreboard("2026-05-20", url_opener=_never_called)


def test_fetch_scoreboard_allowed_without_replay_mode(monkeypatch):
    monkeypatch.delenv("OMEGA_REPLAY_MODE", raising=False)
    called = []

    class _FakeResp:
        def __enter__(self):
            called.append(True)
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return b'{"events": []}'

    espn_wnba.fetch_scoreboard("2026-05-20", url_opener=lambda *_a, **_kw: _FakeResp())
    assert called, "url_opener was not called — guard may be blocking incorrectly"
