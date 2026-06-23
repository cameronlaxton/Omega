"""Tests for tennis odds resolution: dynamic sport keys, list_events, ATP/WTA."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest

from omega.integrations.odds_api import (
    TENNIS_TOUR_KEY_PREFIXES,
    HistoricalEvent,
    OddsApiKeyMissing,
    SportInfo,
    resolve_tennis_sport_keys,
)
from omega.integrations.odds_cache import OddsCache
from omega.integrations.odds_resolver import list_events

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_sport(key: str, active: bool = True) -> SportInfo:
    return SportInfo(
        key=key,
        group="Tennis",
        title=key,
        description="",
        active=active,
        has_outrights=False,
    )


def _make_event(event_id: str, home: str, away: str, sport_key: str) -> HistoricalEvent:
    return HistoricalEvent(
        event_id=event_id,
        sport_key=sport_key,
        commence_time="2026-06-19T13:00:00Z",
        home_team=home,
        away_team=away,
    )


def _make_client(sports: list[SportInfo], events_per_key: dict[str, list]) -> MagicMock:
    client = MagicMock()
    client.fetch_sports.return_value = sports
    client.last_quota_headers = {}

    def _fetch_events(
        league, commence_time_from=None, commence_time_to=None, sport_key=None, request_cost=0
    ):
        return events_per_key.get(sport_key, []) if sport_key is not None else []

    client.fetch_events.side_effect = _fetch_events
    return client


# ---------------------------------------------------------------------------
# TENNIS_TOUR_KEY_PREFIXES contract
# ---------------------------------------------------------------------------


def test_atp_prefix_maps_correctly():
    assert TENNIS_TOUR_KEY_PREFIXES["ATP"] == "tennis_atp"


def test_wta_prefix_maps_correctly():
    assert TENNIS_TOUR_KEY_PREFIXES["WTA"] == "tennis_wta"


def test_invalid_tour_raises():
    client = MagicMock()
    client.fetch_sports.return_value = []
    with pytest.raises(ValueError, match="tour must be one of"):
        resolve_tennis_sport_keys(client, "NBA")


# ---------------------------------------------------------------------------
# resolve_tennis_sport_keys
# ---------------------------------------------------------------------------


def test_atp_keys_filtered_from_sports_index(tmp_path):
    sports = [
        _make_sport("tennis_atp_wimbledon"),
        _make_sport("tennis_atp_us_open"),
        _make_sport("tennis_wta_wimbledon"),  # should be excluded
        _make_sport("basketball_nba"),  # should be excluded
        _make_sport("tennis_atp_inactive", active=False),  # inactive, excluded
    ]
    client = MagicMock()
    client.fetch_sports.return_value = sports

    keys = resolve_tennis_sport_keys(client, "ATP", cache_dir=str(tmp_path))
    assert sorted(keys) == ["tennis_atp_us_open", "tennis_atp_wimbledon"]


def test_wta_keys_filtered_from_sports_index(tmp_path):
    sports = [
        _make_sport("tennis_wta_french_open"),
        _make_sport("tennis_atp_french_open"),  # ATP excluded
        _make_sport("tennis_wta_wimbledon"),
    ]
    client = MagicMock()
    client.fetch_sports.return_value = sports

    keys = resolve_tennis_sport_keys(client, "WTA", cache_dir=str(tmp_path))
    assert sorted(keys) == ["tennis_wta_french_open", "tennis_wta_wimbledon"]


def test_no_active_keys_returns_empty(tmp_path):
    client = MagicMock()
    client.fetch_sports.return_value = [_make_sport("tennis_atp_wimbledon", active=False)]
    keys = resolve_tennis_sport_keys(client, "ATP", cache_dir=str(tmp_path))
    assert keys == []


def test_stale_cache_served_on_fetch_failure(tmp_path):
    cache_file = tmp_path / "tennis_keys_atp.json"
    cache_file.write_text('["tennis_atp_wimbledon"]', encoding="utf-8")

    # Make mtime old enough that it's past the TTL, but override ttl to 0 for the test
    client = MagicMock()
    client.fetch_sports.side_effect = RuntimeError("network down")

    # ttl_seconds=0 means the fresh-fetch path is taken; but it fails → stale cache served
    keys = resolve_tennis_sport_keys(client, "ATP", cache_dir=str(tmp_path), ttl_seconds=0)
    assert keys == ["tennis_atp_wimbledon"]


def test_cold_failure_without_cache_raises(tmp_path):
    client = MagicMock()
    client.fetch_sports.side_effect = RuntimeError("network down")
    with pytest.raises(RuntimeError, match="network down"):
        resolve_tennis_sport_keys(client, "ATP", cache_dir=str(tmp_path), ttl_seconds=0)


# ---------------------------------------------------------------------------
# list_events with tennis leagues
# ---------------------------------------------------------------------------


class _NoCache(OddsCache):
    """Stub cache that always misses so we hit the fetch path."""

    def __init__(self) -> None:
        pass

    def get(self, cache_key: str):
        return None

    def set(self, cache_key: str, league: str, market: str, market_data, **kwargs):
        pass

    @staticmethod
    def compute_event_list_cache_key(
        league: str,
        commence_time_from: str | None = None,
        commence_time_to: str | None = None,
    ) -> str:
        return f"events:{league}"


_TENNIS_RESOLVER = "omega.integrations.odds_resolver.resolve_tennis_sport_keys"


def test_atp_list_events_aggregates_from_multiple_keys():
    atp_keys = ["tennis_atp_wimbledon", "tennis_atp_us_open"]
    events_per_key = {
        "tennis_atp_wimbledon": [_make_event("e1", "Djokovic", "Alcaraz", "tennis_atp_wimbledon")],
        "tennis_atp_us_open": [_make_event("e2", "Sinner", "Medvedev", "tennis_atp_us_open")],
    }
    client = MagicMock()
    client.last_quota_headers = {}

    def _fetch_events(
        league, commence_time_from=None, commence_time_to=None, sport_key=None, request_cost=0
    ):
        return events_per_key.get(sport_key, []) if sport_key is not None else []

    client.fetch_events.side_effect = _fetch_events

    with patch(_TENNIS_RESOLVER, return_value=atp_keys):
        result = list_events(league="ATP", client=client, cache=_NoCache())

    assert result["status"] == "success"
    assert result["league"] == "ATP"
    assert len(result["events"]) == 2
    event_ids = {e["event_id"] for e in result["events"]}
    assert event_ids == {"e1", "e2"}
    assert "sport_keys" in result


def test_wta_list_events_returns_wta_events():
    wta_keys = ["tennis_wta_wimbledon"]
    events_per_key = {
        "tennis_wta_wimbledon": [_make_event("w1", "Swiatek", "Gauff", "tennis_wta_wimbledon")],
    }
    client = MagicMock()
    client.last_quota_headers = {}
    client.fetch_events.side_effect = lambda _, **kw: events_per_key.get(kw.get("sport_key"), [])

    with patch(_TENNIS_RESOLVER, return_value=wta_keys):
        result = list_events(league="WTA", client=client, cache=_NoCache())

    assert result["status"] == "success"
    assert result["events"][0]["event_id"] == "w1"


def test_no_active_atp_tournament_returns_empty():
    client = MagicMock()
    client.last_quota_headers = {}

    with patch(_TENNIS_RESOLVER, return_value=[]):
        result = list_events(league="ATP", client=client, cache=_NoCache())

    assert result["status"] == "empty"
    assert result["events"] == []
    assert any("no active" in r.lower() for r in result["skipped_reasons"])


def test_api_key_missing_propagates_gracefully():
    client = MagicMock()
    client.last_quota_headers = {}

    with patch(_TENNIS_RESOLVER, side_effect=OddsApiKeyMissing("OMEGA_ODDS_API_KEY not set")):
        result = list_events(league="ATP", client=client, cache=_NoCache())

    assert result["status"] == "unavailable"
    assert any(
        "OMEGA_ODDS_API_KEY" in r or "api_key" in r.lower() for r in result["skipped_reasons"]
    )


def test_non_tennis_list_events_not_affected():
    """MLB list_events should use the static SPORT_KEY_MAP path, not the tennis helper."""
    client = MagicMock()
    client.last_quota_headers = {}
    client.fetch_events.return_value = [
        _make_event("mlb1", "Yankees", "Red Sox", "baseball_mlb"),
    ]

    result = list_events(league="MLB", client=client, cache=_NoCache())
    assert result["league"] == "MLB"
    # Should NOT have called fetch_sports (tennis dynamic key path)
    client.fetch_sports.assert_not_called()


def test_grand_slam_dispatches_to_tennis_path():
    """GRAND_SLAM is in TENNIS_LEAGUES so list_events must not call _require_sport_key."""
    client = MagicMock()
    client.last_quota_headers = {}

    with patch(_TENNIS_RESOLVER, return_value=[]):
        # Should not raise ValueError from _require_sport_key
        result = list_events(league="GRAND_SLAM", client=client, cache=_NoCache())

    assert result["status"] in ("empty", "unavailable", "success")
