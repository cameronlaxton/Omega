"""Tests for MLB prop resolution: structured skip reason codes and failure gating.

Key invariant: a failed prop lookup must never silently become an actionable
bet.  The ``skip_code`` field on ``status=unavailable`` results must be
machine-readable and stable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from omega.integrations.odds_api import OddsApiKeyMissing
from omega.integrations.odds_resolver import (
    SKIP_MARKET_UNAVAILABLE,
    SKIP_MISSING_PARAMETERS,
    SKIP_NO_API_KEY,
    SKIP_NO_EVENT_MATCH,
    SKIP_NO_PROVIDER_MAPPING,
    SKIP_NO_QUOTES_MATCH,
    _classify_skip_code,
    resolve_odds,
)

# ---------------------------------------------------------------------------
# _classify_skip_code unit tests
# ---------------------------------------------------------------------------


def test_classify_no_api_key():
    assert _classify_skip_code(["OMEGA_ODDS_API_KEY not set"]) == SKIP_NO_API_KEY


def test_classify_no_event_match():
    assert (
        _classify_skip_code(["no exact event match for Yankees @ Red Sox"]) == SKIP_NO_EVENT_MATCH
    )


def test_classify_event_id_required():
    assert (
        _classify_skip_code(["event_id or exact home_team+away_team is required"])
        == SKIP_NO_EVENT_MATCH
    )


def test_classify_no_provider_mapping():
    assert (
        _classify_skip_code(["no provider market mapping for MLB prop_type='era'"])
        == SKIP_NO_PROVIDER_MAPPING
    )


def test_classify_market_unavailable():
    assert (
        _classify_skip_code(["betmgm does not list market 'pitcher_strikeouts' for this event"])
        == SKIP_MARKET_UNAVAILABLE
    )


def test_classify_no_quotes_match():
    assert _classify_skip_code(["no exact BetMGM market match"]) == SKIP_NO_QUOTES_MATCH


def test_classify_missing_parameters():
    assert (
        _classify_skip_code(["player_name and prop_type are required for prop odds"])
        == SKIP_MISSING_PARAMETERS
    )


def test_classify_empty_returns_unknown():
    from omega.integrations.odds_resolver import SKIP_UNKNOWN

    assert _classify_skip_code([]) == SKIP_UNKNOWN


def test_classify_unknown_string():
    from omega.integrations.odds_resolver import SKIP_UNKNOWN

    assert _classify_skip_code(["some unexpected error message"]) == SKIP_UNKNOWN


# ---------------------------------------------------------------------------
# skip_code present in unavailable results
# ---------------------------------------------------------------------------


class _NoCache:
    def get(self, key):
        return None

    def find_by_teams(self, *a, **kw):
        return None

    def find_by_event_id(self, *a, **kw):
        return None

    def set(self, *a, **kw):
        pass

    def compute_cache_key(self, *a, **kw):
        return "test-key"


def _mock_client_no_key():
    client = MagicMock()
    client.last_quota_headers = {}
    client.fetch_events.side_effect = OddsApiKeyMissing("OMEGA_ODDS_API_KEY not set")
    return client


def test_no_api_key_produces_skip_code():
    result = resolve_odds(
        kind="prop",
        league="MLB",
        home_team="Yankees",
        away_team="Red Sox",
        player_name="Gerrit Cole",
        prop_type="strikeouts_pitched",
        line=7.5,
        client=_mock_client_no_key(),
        cache=_NoCache(),
    )
    assert result["status"] == "unavailable"
    assert "skip_code" in result
    assert result["skip_code"] == SKIP_NO_API_KEY


def test_no_provider_mapping_produces_skip_code():
    """Requesting an unmapped MLB prop type must give skip_code=no_provider_mapping."""
    client = MagicMock()
    client.last_quota_headers = {}
    # Even if we somehow had an event_id, the market mapping step fires first
    client.fetch_events.return_value = [
        MagicMock(
            event_id="evt-abc",
            home_team="Yankees",
            away_team="Red Sox",
            commence_time="2026-06-20T19:00:00Z",
            sport_key="baseball_mlb",
        )
    ]

    result = resolve_odds(
        kind="prop",
        league="MLB",
        home_team="Yankees",
        away_team="Red Sox",
        player_name="Gerrit Cole",
        prop_type="era",  # NOT in PROP_MARKET_MAP["MLB"]
        line=3.00,
        client=client,
        cache=_NoCache(),
    )
    assert result["status"] == "unavailable"
    assert result["skip_code"] == SKIP_NO_PROVIDER_MAPPING
    assert any("no provider market mapping" in r for r in result["skipped_reasons"])


def test_no_event_match_produces_skip_code():
    """When no matching event is found, skip_code must be no_event_match."""
    client = MagicMock()
    client.last_quota_headers = {}
    client.fetch_events.return_value = []  # empty — no matching events

    result = resolve_odds(
        kind="game",
        league="MLB",
        home_team="Yankees",
        away_team="Red Sox",
        client=client,
        cache=_NoCache(),
    )
    assert result["status"] == "unavailable"
    assert result["skip_code"] == SKIP_NO_EVENT_MATCH


def test_missing_player_name_produces_skip_code():
    """Calling prop resolution without player_name must produce skip_code=missing_parameters."""
    client = MagicMock()
    client.last_quota_headers = {}
    client.fetch_events.return_value = [
        MagicMock(
            event_id="e1",
            home_team="Yankees",
            away_team="Red Sox",
            commence_time="2026-06-20T19:00:00Z",
            sport_key="baseball_mlb",
        ),
    ]

    result = resolve_odds(
        kind="prop",
        league="MLB",
        home_team="Yankees",
        away_team="Red Sox",
        player_name=None,  # missing
        prop_type="strikeouts_pitched",
        line=7.5,
        event_id="e1",
        client=client,
        cache=_NoCache(),
    )
    assert result["status"] == "unavailable"
    assert result["skip_code"] == SKIP_MISSING_PARAMETERS


# ---------------------------------------------------------------------------
# Invariant: failed prop lookup cannot become actionable
# ---------------------------------------------------------------------------


def test_unavailable_result_has_null_request_patch():
    """Any unavailable result must have request_patch=None so the engine cannot be called."""
    client = _mock_client_no_key()
    result = resolve_odds(
        kind="prop",
        league="MLB",
        home_team="Yankees",
        away_team="Red Sox",
        player_name="Gerrit Cole",
        prop_type="strikeouts_pitched",
        line=7.5,
        client=client,
        cache=_NoCache(),
    )
    assert result["status"] == "unavailable"
    assert result["request_patch"] is None, (
        "Unavailable props must have request_patch=None — "
        "a non-None request_patch can be fed to the engine and would produce "
        "an actionable trace without valid odds."
    )


def test_unavailable_result_carries_skip_code_for_programmatic_parsing():
    """Ensure skip_code is always present so agents can parse without string matching."""
    client = _mock_client_no_key()
    result = resolve_odds(
        kind="game",
        league="MLB",
        home_team="Yankees",
        away_team="Red Sox",
        client=client,
        cache=_NoCache(),
    )
    assert "skip_code" in result
    assert isinstance(result["skip_code"], str)
    assert len(result["skip_code"]) > 0


def test_skip_code_constant_values_are_stable():
    """These values are documented; changing them is a breaking change."""
    assert SKIP_NO_API_KEY == "no_api_key"
    assert SKIP_NO_EVENT_MATCH == "no_event_match"
    assert SKIP_MARKET_UNAVAILABLE == "market_unavailable"
    assert SKIP_NO_PROVIDER_MAPPING == "no_provider_mapping"
    assert SKIP_NO_QUOTES_MATCH == "no_quotes_match"
    assert SKIP_MISSING_PARAMETERS == "missing_parameters"
