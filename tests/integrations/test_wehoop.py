"""Tests for the wehoop WNBA historical adapter.

Covers artifact construction, ETL fail-loud on schema drift, alias-based team
exclusion, cache-served fetch (no network), replay-mode guarding, and that a
built artifact replays deterministically through the WNBA backend.

References:
  omega/integrations/wehoop.py
  omega/integrations/_etl.py
  omega/strategy/artifacts.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from omega.core.contracts.schemas import GameAnalysisRequest
from omega.core.contracts.service import analyze_game
from omega.integrations import wehoop
from omega.integrations._etl import SourceSchemaDriftError
from omega.integrations._guards import OmegaReplayModeError


def _game_rows(game_id=1):
    return [
        {
            "game_id": game_id,
            "season": 2025,
            "game_date": "2025-05-20",
            "team_display_name": "Las Vegas Aces",
            "opponent_team_display_name": "New York Liberty",
            "home_away": "home",
            "team_score": 88,
            "opponent_team_score": 82,
            "field_goals_attempted": 70,
            "free_throws_attempted": 20,
            "offensive_rebounds": 10,
            "turnovers": 12,
        },
        {
            "game_id": game_id,
            "season": 2025,
            "game_date": "2025-05-20",
            "team_display_name": "New York Liberty",
            "opponent_team_display_name": "Las Vegas Aces",
            "home_away": "away",
            "team_score": 82,
            "opponent_team_score": 88,
            "field_goals_attempted": 72,
            "free_throws_attempted": 18,
            "offensive_rebounds": 9,
            "turnovers": 14,
        },
    ]


def test_build_artifacts_from_box_rows():
    artifacts, skipped = wehoop.build_wnba_artifacts(_game_rows())
    assert skipped == []
    assert len(artifacts) == 1
    a = artifacts[0]
    assert a.league == "WNBA"
    assert a.home_team == "Las Vegas Aces"
    assert a.away_team == "New York Liberty"
    assert a.outcome == {"home_score": 88, "away_score": 82}
    # Winner has the higher offensive rating; required basketball keys present.
    assert a.home_context["off_rating"] > a.away_context["off_rating"]
    assert set(a.home_context) == {"off_rating", "def_rating", "pace"}


def test_artifact_id_is_deterministic():
    a1 = wehoop.build_wnba_artifacts(_game_rows())[0][0]
    a2 = wehoop.build_wnba_artifacts(_game_rows())[0][0]
    assert a1.artifact_id == a2.artifact_id


def test_schema_drift_fails_loud():
    rows = _game_rows()
    del rows[0]["team_score"]  # upstream renamed/dropped a required column
    with pytest.raises(SourceSchemaDriftError) as exc:
        wehoop.build_wnba_artifacts(rows)
    assert exc.value.source == "wehoop"


def test_unresolved_team_excluded_when_alias_table_present():
    rows = _game_rows()
    rows[0]["team_display_name"] = "Mystery Team"  # not in alias table
    alias_table = {"canonical": ["Las Vegas Aces", "New York Liberty"], "aliases": {}}
    artifacts, skipped = wehoop.build_wnba_artifacts(rows, alias_table=alias_table)
    assert artifacts == []
    assert len(skipped) == 1


def test_incomplete_pair_skipped():
    rows = _game_rows()
    rows = rows[:1]  # only the home row
    artifacts, skipped = wehoop.build_wnba_artifacts(rows)
    assert artifacts == []
    assert "incomplete" in skipped[0]


def test_fetch_served_from_cache_without_network(tmp_path):
    # Pre-seed the cache so fetch_team_box must not hit the network.
    cache_dir = tmp_path / "wehoop"
    cache_dir.mkdir(parents=True)
    df = pd.DataFrame(_game_rows())
    df.to_parquet(cache_dir / "wnba_team_box_2025.parquet", index=False)

    def _never(*_a, **_kw):
        raise AssertionError("network fetch should not happen on a cache hit")

    out = wehoop.fetch_team_box(2025, cache_root=str(tmp_path), url_opener=_never)
    assert len(out) == 2


def test_release_url_uses_current_sportsdataverse_asset():
    assert "sportsdataverse-data/releases/download/espn_wnba_team_boxscores" in (
        wehoop.WEHOOP_TEAM_BOX_URL_TEMPLATE
    )


def test_cold_fetch_blocked_in_replay_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")

    def _never(*_a, **_kw):
        raise AssertionError("guard should fire before any network call")

    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        wehoop.fetch_team_box(2025, cache_root=str(tmp_path), url_opener=_never)


def test_artifact_replays_deterministically_through_wnba_backend():
    artifact = wehoop.build_wnba_artifacts(_game_rows())[0][0]
    req = GameAnalysisRequest(
        home_team=artifact.home_team,
        away_team=artifact.away_team,
        league="WNBA",
        n_iterations=3000,
        seed=artifact.simulation_seed,
        home_context=artifact.home_context,
        away_context=artifact.away_context,
        game_context={"is_playoff": False, "rest_days": 2},
    )
    r1 = analyze_game(req)
    r2 = analyze_game(req)
    assert r1.status == "success"
    assert r1.simulation is not None
    assert r2.simulation is not None
    assert r1.simulation.simulation_backend == "markov_state_wnba"
    assert r1.simulation.predicted_total == r2.simulation.predicted_total
    assert r1.simulation.home_win_prob == r2.simulation.home_win_prob
