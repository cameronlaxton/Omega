"""Unit tests for the game_context resolver and Odds API scores parsing."""

from __future__ import annotations

from types import SimpleNamespace

from omega.integrations.game_context import resolve_game_context
from omega.integrations.odds_api import ScoreEvent, parse_scores


def _final(date_str, home, away):
    return SimpleNamespace(date=date_str, home_team=home, away_team=away, status="final")


def _scoreboard(games_by_date):
    def _fn(league, date_str):
        return games_by_date.get(date_str, [])

    return _fn


def test_resolve_rest_days_from_scoreboard():
    sb = _scoreboard(
        {
            # away (Pacers) played the night before -> 0 days rest (back-to-back)
            "2026-05-09": [_final("2026-05-09", "Indiana Pacers", "Miami Heat")],
            # home (Celtics) played two nights before -> 1 day of rest
            "2026-05-08": [_final("2026-05-08", "Boston Celtics", "New York Knicks")],
        }
    )
    out = resolve_game_context(
        "NBA", "Boston Celtics", "Indiana Pacers", "2026-05-10", scoreboard_fn=sb
    )
    gc = out["game_context"]
    assert gc["rest_days"] == 1  # home reference
    assert gc["home_rest_days"] == 1
    assert gc["away_rest_days"] == 0
    assert gc["is_b2b_home"] is False
    assert gc["is_b2b_away"] is True
    assert gc["is_playoff"] is True  # within NBA heuristic window
    assert out["provenance"]["rest_days_source"] == "scoreboard"
    assert out["provenance"]["is_playoff_source"] == "date_heuristic"


def test_b2b_flag_when_played_previous_night():
    sb = _scoreboard(
        {"2026-05-09": [_final("2026-05-09", "Boston Celtics", "Indiana Pacers")]}
    )
    out = resolve_game_context(
        "NBA", "Boston Celtics", "Indiana Pacers", "2026-05-10", scoreboard_fn=sb
    )
    assert out["game_context"]["is_b2b_home"] is True
    assert out["game_context"]["rest_days"] == 0


def test_missing_schedule_lands_in_needs_manual():
    out = resolve_game_context(
        "NBA",
        "Boston Celtics",
        "Indiana Pacers",
        "2026-05-10",
        scoreboard_fn=_scoreboard({}),
        odds_client=SimpleNamespace(fetch_scores=lambda *a, **k: []),
    )
    assert "home_rest_days" in out["needs_manual"]
    assert "away_rest_days" in out["needs_manual"]
    assert out["provenance"]["rest_days_source"] is None


def test_applicable_evidence_includes_markov_flag_and_suggestions():
    out = resolve_game_context(
        "NBA", "Boston Celtics", "Indiana Pacers", "2026-05-10",
        scoreboard_fn=_scoreboard({}),
        odds_client=SimpleNamespace(fetch_scores=lambda *a, **k: []),
    )
    sigs = {e["signal_type"] for e in out["applicable_evidence"]}
    assert "pace_up" in sigs  # basketball + markov-eligible
    markov = {e["signal_type"] for e in out["applicable_evidence"] if e["markov_eligible"]}
    assert "pace_up" in markov and "recent_form" not in markov
    # unwired semantic dims surfaced as suggested evidence (reuse motivation_edge)
    reasons = {s["reason"].split(":")[0] for s in out["suggested_evidence"]}
    assert "vs_former_team" in reasons
    assert any(s["signal_type"] == "motivation_edge" for s in out["suggested_evidence"])


def test_rivalry_suggests_motivation_edge():
    out = resolve_game_context(
        "NBA", "Boston Celtics", "Los Angeles Lakers", "2026-05-10",
        scoreboard_fn=_scoreboard({}),
        odds_client=SimpleNamespace(fetch_scores=lambda *a, **k: []),
    )
    rivalry = [s for s in out["suggested_evidence"] if s["reason"].startswith("rivalry")]
    assert rivalry and rivalry[0]["signal_type"] == "motivation_edge"


def test_mlb_park_factor_resolved():
    out = resolve_game_context(
        "MLB", "Colorado Rockies", "San Diego Padres", "2026-07-04",
        scoreboard_fn=_scoreboard({}),
        odds_client=SimpleNamespace(fetch_scores=lambda *a, **k: []),
    )
    assert out["game_context"]["park_factor"] == 1.15
    assert out["provenance"]["park_factor_source"] == "static_approximate"


def test_odds_scores_fallback_for_league_without_scoreboard():
    fake_client = SimpleNamespace(
        fetch_scores=lambda league, days_from=3: [
            ScoreEvent(
                event_id="e1",
                sport_key="icehockey_nhl",
                commence_time="2026-05-08T23:00:00Z",
                completed=True,
                home_team="Boston Bruins",
                away_team="Toronto Maple Leafs",
            )
        ]
    )
    out = resolve_game_context(
        "NHL", "Boston Bruins", "Toronto Maple Leafs", "2026-05-10",
        odds_client=fake_client,
    )
    assert out["game_context"]["home_rest_days"] == 1
    assert out["provenance"]["rest_days_source"] == "odds_api_scores"


def test_parse_scores_reads_completed_flag_and_teams():
    payload = [
        {
            "id": "abc",
            "sport_key": "basketball_nba",
            "commence_time": "2026-05-09T23:00:00Z",
            "completed": True,
            "home_team": "Boston Celtics",
            "away_team": "Indiana Pacers",
            "scores": [{"name": "Boston Celtics", "score": "110"}],
        }
    ]
    events = parse_scores(payload)
    assert len(events) == 1
    assert events[0].completed is True
    assert events[0].home_team == "Boston Celtics"
    assert parse_scores("not-a-list") == []
