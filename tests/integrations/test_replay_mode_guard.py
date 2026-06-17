"""Tests for the OMEGA_REPLAY_MODE live-fetch guard.

Verifies that setting OMEGA_REPLAY_MODE=1 raises OmegaReplayModeError at the
integration layer before any network call is attempted (espn_nba, espn_mlb,
espn_boxscore, odds_api). Guard is env-var-driven; tests that supply a mock
url_opener do NOT trip it — the guard is orthogonal to unit-test injection.

References:
  omega/integrations/_guards.py
  omega/integrations/espn_nba.py::fetch_scoreboard
  omega/integrations/espn_mlb.py::fetch_scoreboard
  omega/integrations/espn_boxscore.py::fetch_box_score
  omega/integrations/odds_api.py::OddsApiClient._get_json
"""

from __future__ import annotations

import pytest

from omega.integrations._guards import OmegaReplayModeError

# ---------------------------------------------------------------------------
# Helper: make a mock url_opener that should never be called
# ---------------------------------------------------------------------------


def _never_called(*_args, **_kwargs):
    raise AssertionError("url_opener should not have been called — guard should have fired first")


# ---------------------------------------------------------------------------
# espn_nba
# ---------------------------------------------------------------------------


def test_espn_nba_fetch_scoreboard_blocked_in_replay_mode(monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    from omega.integrations import espn_nba

    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        espn_nba.fetch_scoreboard("2026-05-01", url_opener=_never_called)


def test_espn_nba_fetch_scoreboard_allowed_without_replay_mode(monkeypatch):
    monkeypatch.delenv("OMEGA_REPLAY_MODE", raising=False)
    from omega.integrations import espn_nba

    called = []

    class _FakeResp:
        def __enter__(self):
            called.append(True)
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return b'{"events": []}'

    espn_nba.fetch_scoreboard("2026-05-01", url_opener=lambda *_a, **_kw: _FakeResp())
    assert called, "url_opener was not called — guard may be blocking incorrectly"


# ---------------------------------------------------------------------------
# espn_mlb
# ---------------------------------------------------------------------------


def test_espn_mlb_fetch_scoreboard_blocked_in_replay_mode(monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    from omega.integrations import espn_mlb

    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        espn_mlb.fetch_scoreboard("2026-05-01", url_opener=_never_called)


# ---------------------------------------------------------------------------
# espn_boxscore
# ---------------------------------------------------------------------------


def test_espn_boxscore_fetch_blocked_in_replay_mode(monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    from omega.integrations import espn_boxscore

    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        espn_boxscore.fetch_box_score("NBA", "event123", url_opener=_never_called)


# ---------------------------------------------------------------------------
# odds_api
# ---------------------------------------------------------------------------


def test_odds_api_get_json_blocked_in_replay_mode(monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    from omega.integrations.odds_api import OddsApiClient

    client = OddsApiClient(api_key="fake-key-for-test")
    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        client._get_json("/sports", {})


# ---------------------------------------------------------------------------
# Static discipline: every integration module must reference the guard
# ---------------------------------------------------------------------------
#
# Phase 7 standard: it is easy to forget assert_not_replay_mode() in a new
# adapter — the failure is silent until a backtest accidentally hits the
# network. This static scan fails CI the moment a new module under
# omega/integrations/ omits the guard symbol. __init__.py is exempt (no fetch).
#
# Pure-local helpers that perform no live fetch (e.g. the SQLite odds cache) are
# allowlisted: the guard belongs at the fetch seam, not in a passive cache. The
# odds resolver already calls assert_not_replay_mode() before it ever consults
# the cache or the Odds API client, so the cache-miss/refresh path is protected
# upstream. A guard inside a fetch-less module would be vacuous and misleading.
#
# To keep the allowlist honest, test_pure_local_modules_make_no_network_calls
# asserts that nothing on the allowlist references a network-fetch primitive —
# so a real adapter can never be whitelisted to silence the guard requirement.

PURE_LOCAL_MODULES = frozenset({"odds_cache.py", "_soccer_xg.py"})

# Indicators that a module performs (or directly triggers) a live network fetch.
_FETCH_INDICATORS = ("urllib", "requests", "http.client", "url_opener", "OddsApiClient")


def test_pure_local_modules_make_no_network_calls():
    """Allowlisted pure-local modules must not contain network-fetch primitives."""
    import pathlib

    import omega.integrations as integrations_pkg

    pkg_dir = pathlib.Path(integrations_pkg.__file__).parent
    for name in sorted(PURE_LOCAL_MODULES):
        source = (pkg_dir / name).read_text(encoding="utf-8")
        leaked = [ind for ind in _FETCH_INDICATORS if ind in source]
        assert not leaked, (
            f"{name} is allowlisted as pure-local but references network "
            f"primitives {leaked}. It must add an assert_not_replay_mode() guard "
            "and be removed from PURE_LOCAL_MODULES."
        )


def test_every_integration_module_references_guard():
    import pathlib

    import omega.integrations as integrations_pkg

    pkg_dir = pathlib.Path(integrations_pkg.__file__).parent
    offenders = []
    for path in sorted(pkg_dir.glob("*.py")):
        if path.name == "__init__.py" or path.name in PURE_LOCAL_MODULES:
            continue
        source = path.read_text(encoding="utf-8")
        if "assert_not_replay_mode" not in source:
            offenders.append(path.name)
    assert not offenders, (
        "integration modules missing assert_not_replay_mode reference: "
        f"{offenders}. Every adapter that can fetch live data must guard "
        "against OMEGA_REPLAY_MODE (or be allowlisted in PURE_LOCAL_MODULES if "
        "it performs no live fetch)."
    )
