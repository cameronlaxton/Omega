from __future__ import annotations

from omega.integrations import espn_nfl, espn_nhl


def _event(status: str = "STATUS_FINAL") -> dict:
    return {
        "id": "evt-1",
        "date": "2026-01-01T00:00Z",
        "competitions": [
            {
                "status": {"type": {"name": status}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": "3",
                        "team": {"displayName": "Boston Bruins", "abbreviation": "BOS"},
                    },
                    {
                        "homeAway": "away",
                        "score": "2",
                        "team": {"displayName": "New York Rangers", "abbreviation": "NYR"},
                    },
                ],
            }
        ],
    }


def test_nhl_scoreboard_skips_non_final_events():
    final_games = espn_nhl.parse_scoreboard({"events": [_event("STATUS_FINAL")]})
    live_games = espn_nhl.parse_scoreboard({"events": [_event("STATUS_IN_PROGRESS")]})

    assert len(final_games) == 1
    assert live_games == []


def test_nfl_scoreboard_skips_non_final_events():
    def nfl_event(status: str) -> dict:
        event = _event(status)
        competitors = event["competitions"][0]["competitors"]
        competitors[0]["team"] = {"displayName": "Dallas Cowboys", "abbreviation": "DAL"}
        competitors[1]["team"] = {"displayName": "New York Giants", "abbreviation": "NYG"}
        return event

    final_games = espn_nfl.parse_scoreboard({"events": [nfl_event("STATUS_FINAL")]})
    live_games = espn_nfl.parse_scoreboard({"events": [nfl_event("STATUS_IN_PROGRESS")]})

    assert len(final_games) == 1
    assert live_games == []


def test_nfl_and_nhl_context_fetches_use_https():
    urls: list[str] = []

    def opener(url: str, **_kwargs):
        urls.append(url)
        raise RuntimeError("stop after url capture")

    espn_nfl.fetch_team_context("Dallas Cowboys", url_opener=opener)
    espn_nfl.fetch_player_context("123", url_opener=opener)
    espn_nhl.fetch_team_context("Boston Bruins", url_opener=opener)
    espn_nhl.fetch_player_context("456", url_opener=opener)

    assert urls
    assert all(url.startswith("https://site.api.espn.com/") for url in urls)
