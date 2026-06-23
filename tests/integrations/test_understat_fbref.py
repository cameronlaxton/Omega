"""Tests for the Understat + FBref xG redundancy adapters (Phase 7 M2 PR-S6).

Fixture-driven (no network): page parsing, fail-loud on structure drift,
alias resolution/exclusion, cache-served fetches, replay guarding, and the
cross-source disagreement check.
"""

from __future__ import annotations

import json

import pytest

from omega.integrations import fbref, understat
from omega.integrations._etl import SourceSchemaDriftError
from omega.integrations._guards import OmegaReplayModeError
from omega.trace.priors import XgPrior

# ---------------------------------------------------------------------------
# Understat
# ---------------------------------------------------------------------------


def _understat_html(teams: dict) -> str:
    blob = json.dumps(teams).replace("\\", "\\\\").replace("'", "\\'")
    # Understat hex-escapes the blob; unicode_escape decoding must handle both
    # plain and escaped forms, so escape a marker character to prove it.
    blob = blob.replace("{", "\\x7B", 1)
    return f"<html><script>var teamsData = JSON.parse('{blob}');</script></html>"


_UNDERSTAT_TEAMS = {
    "88": {
        "id": "88",
        "title": "Manchester City",
        "history": [
            {"xG": 2.2, "xGA": 0.7},
            {"xG": 1.8, "xGA": 1.1},
        ],
    },
    "82": {
        "id": "82",
        "title": "Tottenham",
        "history": [{"xG": 1.4, "xGA": 1.6}],
    },
}


def test_understat_parse_and_priors():
    rows = understat.parse_teams_data(_understat_html(_UNDERSTAT_TEAMS))
    priors, unresolved = understat.build_xg_priors(
        rows, league="EPL", season="2025", as_of_date="2026-06-10"
    )
    assert unresolved == []
    by_team = {p.team: p for p in priors}
    assert by_team["Manchester City"].xg_for == pytest.approx(2.0)
    assert by_team["Manchester City"].xg_against == pytest.approx(0.9)
    assert by_team["Manchester City"].matches == 2
    assert by_team["Tottenham"].source == "understat"


def test_understat_structure_drift_fails_loud():
    with pytest.raises(ValueError, match="structure drift"):
        understat.parse_teams_data("<html><body>cloudflare says hi</body></html>")


def test_understat_schema_drift_fails_loud():
    rows = [{"team": "Arsenal", "matches": 2}]  # missing xg totals
    with pytest.raises(SourceSchemaDriftError):
        understat.build_xg_priors(rows, league="EPL", season="2025", as_of_date="2026-06-10")


def test_understat_unknown_league_raises():
    with pytest.raises(ValueError, match="no Understat slug"):
        understat.fetch_league_html("MLS", "2025")


def test_understat_cache_hit_makes_no_network_call(tmp_path):
    cache_dir = tmp_path / "understat"
    cache_dir.mkdir(parents=True)
    (cache_dir / "EPL_2025.html").write_text(_understat_html(_UNDERSTAT_TEAMS), encoding="utf-8")

    def _never(*_a, **_kw):
        raise AssertionError("network fetch should not happen on a cache hit")

    html = understat.fetch_league_html("EPL", "2025", cache_root=str(tmp_path), url_opener=_never)
    assert "teamsData" in html


def test_understat_cold_fetch_blocked_in_replay_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    with pytest.raises(OmegaReplayModeError):
        understat.fetch_league_html("EPL", "2025", cache_root=str(tmp_path))


# ---------------------------------------------------------------------------
# FBref
# ---------------------------------------------------------------------------


def _fbref_row(team: str, games: int, xg_for: float, xg_against: float) -> str:
    return (
        f'<tr><th data-stat="team" scope="row"><a href="/x">{team}</a></th>'
        f'<td data-stat="games">{games}</td>'
        f'<td data-stat="xg_for">{xg_for}</td>'
        f'<td data-stat="xg_against">{xg_against}</td></tr>'
    )


_FBREF_HTML = (
    "<html><table>"
    + _fbref_row("Manchester City", 38, 78.6, 32.1)
    + _fbref_row("Tottenham", 38, 60.8, 58.3)
    + "</table></html>"
)


def test_fbref_parse_and_priors():
    rows = fbref.parse_standings(_FBREF_HTML)
    priors, unresolved = fbref.build_xg_priors(
        rows, league="EPL", season="2025", as_of_date="2026-06-10"
    )
    assert unresolved == []
    by_team = {p.team: p for p in priors}
    assert by_team["Manchester City"].xg_for == pytest.approx(78.6 / 38, abs=1e-3)
    assert by_team["Manchester City"].matches == 38
    assert by_team["Tottenham"].source == "fbref"


def test_fbref_structure_drift_fails_loud():
    with pytest.raises(ValueError, match="structure drift"):
        fbref.parse_standings("<html><body>checking your browser...</body></html>")


def test_fbref_alias_exclusion():
    rows = fbref.parse_standings(_FBREF_HTML)
    alias_table = {"canonical": ["Manchester City"], "aliases": {}}
    priors, unresolved = fbref.build_xg_priors(
        rows,
        league="EPL",
        season="2025",
        as_of_date="2026-06-10",
        alias_table=alias_table,
    )
    assert [p.team for p in priors] == ["Manchester City"]
    assert unresolved == ["Tottenham"]


def test_fbref_cold_fetch_blocked_in_replay_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    with pytest.raises(OmegaReplayModeError):
        fbref.fetch_comp_html("EPL", cache_root=str(tmp_path))


# ---------------------------------------------------------------------------
# Cross-source disagreement
# ---------------------------------------------------------------------------


def _prior(team: str, xg_for: float, source: str) -> XgPrior:
    return XgPrior(
        team=team,
        competition="EPL",
        season="2025",
        xg_for=xg_for,
        xg_against=1.0,
        matches=38,
        source=source,
        as_of_date="2026-06-10",
    )


def test_cross_check_flags_disagreement_above_threshold():
    events = understat.cross_check_xg(
        {
            "understat": [_prior("Manchester City", 2.0, "understat")],
            "fbref": [_prior("Manchester City", 1.5, "fbref")],
        }
    )
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "data_provenance"
    assert event["status"] == "warn"
    assert event["outputs"]["relative_disagreement"] == pytest.approx(0.25)


def test_cross_check_silent_within_threshold():
    events = understat.cross_check_xg(
        {
            "understat": [_prior("Manchester City", 2.0, "understat")],
            "fbref": [_prior("Manchester City", 1.9, "fbref")],
        }
    )
    assert events == []


def test_cross_check_ignores_single_source_teams():
    events = understat.cross_check_xg(
        {
            "understat": [_prior("Arsenal", 1.8, "understat")],
            "fbref": [_prior("Tottenham", 1.6, "fbref")],
        }
    )
    assert events == []
