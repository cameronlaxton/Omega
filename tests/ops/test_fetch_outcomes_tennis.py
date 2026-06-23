"""Tests for omega-fetch-outcomes-tennis (Phase 7 M3 PR-T6)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from omega.integrations.odds_api import (
    parse_scores,
    resolve_tennis_sport_keys,
)
from omega.ops.fetch_outcomes_tennis import (
    collect_odds_api_results,
    collect_sackmann_results,
    parse_sets_from_score,
)

# ---------------------------------------------------------------------------
# Score-string parsing (sackmann backfill)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        ("7-6(5) 6-4", (2, 0)),
        ("4-6 6-3 7-5", (2, 1)),
        ("6-4 3-6 7-6(10) 6-2", (3, 1)),
        ("6-0 6-0", (2, 0)),
        ("6-3 4-1 RET", (1, 0)),  # retirement with winner ahead in sets
        ("W/O", None),
        ("", None),
        (None, None),
        ("2-6 1-2 RET", None),  # winner not ahead in completed sets — never grade
    ],
)
def test_parse_sets_from_score(score, expected):
    assert parse_sets_from_score(score) == expected


# ---------------------------------------------------------------------------
# Sackmann source collection
# ---------------------------------------------------------------------------

_HEADER = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
    "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
    "winner_ht,winner_ioc,winner_age,loser_id,loser_seed,loser_entry,"
    "loser_name,loser_hand,loser_ht,loser_ioc,loser_age,score,best_of,round,"
    "minutes,w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,"
    "w_bpFaced,l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,"
    "l_bpFaced,winner_rank,winner_rank_points,loser_rank,loser_rank_points"
)


def _row(date_int, winner, loser, score):
    return (
        f"2026-540,Wimbledon,Grass,128,G,{date_int},701,1,,,{winner},R,191,ITA,24.0,"
        f"2,,,{loser},R,188,SRB,39.0,{score},5,F,200,12,2,90,60,48,15,14,4,5,"
        f"8,3,100,62,46,17,14,7,10,1,11000,2,9000"
    )


def test_collect_sackmann_results_matches_window(tmp_path):
    csv_text = "\n".join(
        [
            _HEADER,
            _row(20260629, "Jannik Sinner", "Novak Djokovic", "7-6(5) 6-4 6-4"),
            _row(20250101, "Old Winner", "Old Loser", "6-4 6-4"),  # outside window
        ]
    )
    (tmp_path / "atp_matches_2026.csv").write_text(csv_text, encoding="utf-8")

    results = collect_sackmann_results(
        ["atp"],
        date(2026, 7, 1),
        date(2026, 7, 14),
        {"canonical": [], "aliases": {}},
        local_root=str(tmp_path),
        cache_root=str(tmp_path / "cache"),
    )
    assert len(results) == 1
    (((pair, played), sets_map),) = results.items()
    assert played == date(2026, 6, 29)  # keyed by tourney start date
    assert sorted(sets_map.values()) == [0, 3]


def test_lookup_match_falls_back_within_tournament_window():
    """Sackmann keys carry the tournament START date; a mid-event trace date
    (e.g. a Wimbledon QF on July 8) must still resolve via the window
    fallback, while exact-date entries win and out-of-window entries never
    match."""
    from omega.ops.fetch_outcomes_tennis import lookup_match

    pair = frozenset({"a", "b"})
    tournament = {(pair, date(2026, 6, 29)): {"a": 3, "b": 1}}
    assert lookup_match(tournament, pair, date(2026, 7, 8)) == {"a": 3, "b": 1}
    # Same pair, two tournaments in range: the most recent start date wins.
    two = {
        (pair, date(2026, 6, 1)): {"a": 2, "b": 0},
        (pair, date(2026, 6, 29)): {"a": 3, "b": 1},
    }
    assert lookup_match(two, pair, date(2026, 7, 8)) == {"a": 3, "b": 1}
    # Exact-date hit short-circuits.
    exact = {(pair, date(2026, 7, 8)): {"a": 0, "b": 2}, **two}
    assert lookup_match(exact, pair, date(2026, 7, 8)) == {"a": 0, "b": 2}
    # Outside the lookback window -> no match.
    assert lookup_match(tournament, pair, date(2026, 8, 15)) is None
    assert lookup_match(tournament, pair, date(2026, 6, 20)) is None


# ---------------------------------------------------------------------------
# Odds API source
# ---------------------------------------------------------------------------


def test_parse_scores_carries_provider_scores_array():
    payload = [
        {
            "id": "evt1",
            "sport_key": "tennis_atp_wimbledon",
            "commence_time": "2026-06-29T12:00:00Z",
            "completed": True,
            "home_team": "Jannik Sinner",
            "away_team": "Novak Djokovic",
            "scores": [
                {"name": "Jannik Sinner", "score": "3"},
                {"name": "Novak Djokovic", "score": "1"},
            ],
        }
    ]
    events = parse_scores(payload)
    assert events[0].scores == (("Jannik Sinner", "3"), ("Novak Djokovic", "1"))
    # Older payload shape without scores stays valid.
    assert parse_scores([{"id": "x", "completed": False}])[0].scores == ()


class _FakeSport:
    def __init__(self, key, active=True):
        self.key = key
        self.active = active


class _FakeClient:
    def __init__(self, sports=None, scores_by_key=None, fail_sports=False):
        self._sports = sports or []
        self._scores_by_key = scores_by_key or {}
        self._fail_sports = fail_sports
        self.sports_calls = 0

    def fetch_sports(self, all_sports=True):
        self.sports_calls += 1
        if self._fail_sports:
            raise RuntimeError("provider down")
        return self._sports

    def fetch_scores(self, league, days_from=3, *, sport_key=None):
        return self._scores_by_key.get(sport_key, [])


def _sports_index():
    return [
        _FakeSport("tennis_atp_wimbledon"),
        _FakeSport("tennis_atp_us_open", active=False),
        _FakeSport("tennis_wta_wimbledon"),
        _FakeSport("soccer_epl"),
    ]


def test_resolve_tennis_sport_keys_filters_active_tour(tmp_path):
    client = _FakeClient(sports=_sports_index())
    keys = resolve_tennis_sport_keys(client, "ATP", cache_dir=tmp_path)
    assert keys == ["tennis_atp_wimbledon"]
    # Second call within TTL serves from disk — no provider call.
    resolve_tennis_sport_keys(client, "ATP", cache_dir=tmp_path)
    assert client.sports_calls == 1


def test_resolve_tennis_sport_keys_stale_fallback(tmp_path):
    (tmp_path / "tennis_keys_atp.json").write_text(
        json.dumps(["tennis_atp_old_open"]), encoding="utf-8"
    )
    failing = _FakeClient(fail_sports=True)
    keys = resolve_tennis_sport_keys(
        failing,
        "ATP",
        cache_dir=tmp_path,
        ttl_seconds=0,  # force refresh attempt
    )
    assert keys == ["tennis_atp_old_open"]  # last-good served

    with pytest.raises(RuntimeError):
        resolve_tennis_sport_keys(failing, "WTA", cache_dir=tmp_path)


def test_resolve_tennis_sport_keys_rejects_unknown_tour(tmp_path):
    with pytest.raises(ValueError, match="tour must be one of"):
        resolve_tennis_sport_keys(_FakeClient(), "ITF", cache_dir=tmp_path)


def test_collect_odds_api_results():
    events = parse_scores(
        [
            {
                "id": "evt1",
                "sport_key": "tennis_atp_wimbledon",
                "commence_time": "2026-06-29T12:00:00Z",
                "completed": True,
                "home_team": "Jannik Sinner",
                "away_team": "Novak Djokovic",
                "scores": [
                    {"name": "Jannik Sinner", "score": "3"},
                    {"name": "Novak Djokovic", "score": "1"},
                ],
            },
            {
                "id": "evt2",
                "sport_key": "tennis_atp_wimbledon",
                "commence_time": "2026-06-29T15:00:00Z",
                "completed": False,
                "home_team": "A",
                "away_team": "B",
            },
        ]
    )

    class _Client(_FakeClient):
        pass

    client = _Client(
        sports=_sports_index(),
        scores_by_key={"tennis_atp_wimbledon": events},
    )
    import tempfile as _tf

    with _tf.TemporaryDirectory() as td:
        from omega.integrations import odds_api as oa

        old_dir = oa._TENNIS_KEYS_CACHE_DIR
        oa._TENNIS_KEYS_CACHE_DIR = __import__("pathlib").Path(td)
        try:
            results = collect_odds_api_results(
                ["ATP"], {"canonical": [], "aliases": {}}, client=client
            )
        finally:
            oa._TENNIS_KEYS_CACHE_DIR = old_dir
    assert len(results) == 1
    sets_map = next(iter(results.values()))
    assert sorted(sets_map.values()) == [1, 3]
