"""Tests for the StatsBomb Open Data adapter (Phase 7 M2).

Covers profile-group competition selection, the Dixon-Coles fit dataset
(score pairs), cache-served fetches with zero network calls, ETL fail-loud on
schema drift, xG aggregation, alias-based team resolution/exclusion, and
replay-mode guarding.
"""

from __future__ import annotations

import json

import pytest

from omega.integrations import statsbomb
from omega.integrations._etl import SourceSchemaDriftError
from omega.integrations._guards import OmegaReplayModeError


_COMPETITIONS = [
    {
        "competition_id": 43,
        "season_id": 106,
        "competition_name": "FIFA World Cup",
        "season_name": "2022",
        "country_name": "International",
    },
    {
        "competition_id": 55,
        "season_id": 282,
        "competition_name": "UEFA Euro",
        "season_name": "2024",
        "country_name": "Europe",
    },
    {
        "competition_id": 2,
        "season_id": 27,
        "competition_name": "Premier League",
        "season_name": "2015/2016",
        "country_name": "England",
    },
]


def _match(match_id, home, away, hs, as_):
    return {
        "match_id": match_id,
        "match_date": "2022-12-18",
        "home_team": {"home_team_id": 1, "home_team_name": home},
        "away_team": {"away_team_id": 2, "away_team_name": away},
        "home_score": hs,
        "away_score": as_,
        "competition": {"competition_id": 43},
    }


_WC_MATCHES = [
    _match(1, "Argentina", "France", 3, 3),
    _match(2, "Croatia", "Morocco", 2, 1),
    _match(3, "France", "Morocco", 2, 0),
]

_EURO_MATCHES = [
    _match(10, "Spain", "England", 2, 1),
]


def _shot(team, xg):
    return {"type": {"name": "Shot"}, "team": {"name": team}, "shot": {"statsbomb_xg": xg}}


_EVENTS_MATCH_1 = [
    {"type": {"name": "Pass"}, "team": {"name": "Argentina"}},
    _shot("Argentina", 0.4),
    _shot("Argentina", 0.2),
    _shot("France", 0.5),
]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _opener(routes, counter):
    def open_url(url, timeout=None):
        counter["n"] += 1
        for fragment, payload in routes.items():
            if fragment in url:
                return _FakeResponse(payload)
        raise AssertionError(f"unexpected url: {url}")

    return open_url


_ROUTES = {
    "competitions.json": _COMPETITIONS,
    "matches/43/106.json": _WC_MATCHES,
    "matches/55/282.json": _EURO_MATCHES,
    "events/1.json": _EVENTS_MATCH_1,
}


def test_profile_group_strips_version():
    assert statsbomb.profile_group("fifa_intl_v1") == "fifa_intl"
    assert statsbomb.profile_group("epl_v12") == "epl"
    assert statsbomb.profile_group("fifa_intl") == "fifa_intl"


def test_select_profile_competitions_filters_to_group():
    selected = statsbomb.select_profile_competitions(_COMPETITIONS, "fifa_intl_v1")
    names = {c.competition_name for c in selected}
    assert names == {"FIFA World Cup", "UEFA Euro"}


def test_unknown_profile_group_raises():
    with pytest.raises(ValueError, match="unknown Dixon-Coles profile group"):
        statsbomb.select_profile_competitions(_COMPETITIONS, "mars_league_v1")


def test_load_profile_matches_returns_score_pairs(tmp_path):
    counter = {"n": 0}
    pairs = statsbomb.load_profile_matches(
        "fifa_intl_v1",
        cache_root=str(tmp_path),
        url_opener=_opener(_ROUTES, counter),
    )
    assert sorted(pairs) == sorted([(3, 3), (2, 1), (2, 0), (2, 1)])
    assert counter["n"] == 3  # competitions + two match files


def test_second_pull_is_served_from_cache(tmp_path):
    counter = {"n": 0}
    opener = _opener(_ROUTES, counter)
    statsbomb.load_profile_matches("fifa_intl_v1", cache_root=str(tmp_path), url_opener=opener)
    first = counter["n"]
    statsbomb.load_profile_matches("fifa_intl_v1", cache_root=str(tmp_path), url_opener=opener)
    assert counter["n"] == first  # zero additional network calls


def test_schema_drift_fails_loud(tmp_path):
    broken = [dict(_WC_MATCHES[0])]
    del broken[0]["home_score"]
    routes = dict(_ROUTES)
    routes["matches/43/106.json"] = broken
    with pytest.raises(SourceSchemaDriftError) as exc:
        statsbomb.load_profile_matches(
            "fifa_intl_v1", cache_root=str(tmp_path), url_opener=_opener(routes, {"n": 0})
        )
    assert exc.value.source == "statsbomb"


def test_cold_fetch_blocked_in_replay_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")

    def _never(*_a, **_kw):
        raise AssertionError("guard should fire before any network call")

    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        statsbomb.fetch_competitions(cache_root=str(tmp_path), url_opener=_never)


def test_xg_aggregation_per_team():
    matches = [statsbomb.SBMatch.model_validate(_WC_MATCHES[0])]
    agg = statsbomb.compute_team_xg_aggregates(matches, {1: _EVENTS_MATCH_1})
    assert agg["Argentina"]["xg_for"] == pytest.approx(0.6)
    assert agg["Argentina"]["xg_against"] == pytest.approx(0.5)
    assert agg["France"]["xg_for"] == pytest.approx(0.5)
    assert agg["France"]["xg_against"] == pytest.approx(0.6)
    assert agg["Argentina"]["matches"] == 1


def test_build_xg_priors_resolves_aliases_and_excludes_unresolved():
    aggregates = {
        "Korea Republic": {"xg_for": 2.0, "xg_against": 3.0, "matches": 2},
        "Atlantis": {"xg_for": 1.0, "xg_against": 1.0, "matches": 1},
    }
    alias_table = {
        "canonical": ["South Korea"],
        "aliases": {"Korea Republic": "South Korea"},
    }
    priors, unresolved = statsbomb.build_xg_priors(
        aggregates,
        competition="FIFA World Cup",
        season="2022",
        as_of_date="2026-06-10",
        alias_table=alias_table,
    )
    assert unresolved == ["Atlantis"]
    assert len(priors) == 1
    prior = priors[0]
    assert prior.team == "South Korea"
    assert prior.xg_for == pytest.approx(1.0)  # per-game average
    assert prior.xg_against == pytest.approx(1.5)
    assert prior.source == "statsbomb"


def test_build_xg_priors_without_alias_table_keeps_names():
    aggregates = {"France": {"xg_for": 4.2, "xg_against": 1.4, "matches": 3}}
    priors, unresolved = statsbomb.build_xg_priors(
        aggregates,
        competition="FIFA World Cup",
        season="2022",
        as_of_date="2026-06-10",
    )
    assert unresolved == []
    assert priors[0].team == "France"
    assert priors[0].xg_for == pytest.approx(1.4)


def test_repo_alias_table_resolves_world_cup_variants():
    from omega.integrations._etl import load_alias_table, resolve_entity

    table = load_alias_table("SOCCER")
    assert resolve_entity("Korea Republic", table) == "South Korea"
    assert resolve_entity("USA", table) == "United States"
    assert resolve_entity("IR Iran", table) == "Iran"
    assert resolve_entity("France", table) == "France"
