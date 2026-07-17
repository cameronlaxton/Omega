"""Phase 0: Recommendation Lab gating + primary matchup API safety.

Verification-plan coverage (design §17, API/presentation tests):
- lab flag gates both HTML and JSON routes (404 when off, 200 when on);
- primary routes stay reachable either way;
- the primary matchup DTO carries no denied keys or blocked phrases;
- no-recommendation traces appear in the primary surface;
- legacy traces render through the labeled compatibility path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from tests.ui.conftest import make_trace, write_valid_sidecar

LAB_JSON_ROUTES = (
    "/api/scanner",
    "/api/traces/sandbox-aaa",
    "/api/traces/sandbox-aaa/similar",
    "/api/bets",
    "/api/bets/led-aaa-1",
    "/api/clv",
    "/api/clv-scatter",
)
LAB_HTML_ROUTES = (
    "/scanner",
    "/traces/sandbox-aaa",
    "/traces/sandbox-aaa/similar",
    "/bets",
    "/bets/led-aaa-1",
    "/clv",
)
PRIMARY_ROUTES = (
    "/",
    "/traces",
    "/sessions",
    "/diagnostics",
    "/api/healthz",
    "/api/diagnostics",
    "/api/matchups",
)

EVENT_IDENTITY = {
    "schema_version": 1,
    "provider": "the-odds-api",
    "provider_event_id": "ev-777",
    "event_key": "MLB::the-odds-api::ev-777",
    "league": "MLB",
    "home_team": "Yankees",
    "away_team": "Red Sox",
    "game_date": "2026-07-16",
}


@pytest.fixture
def lab_off(monkeypatch):
    monkeypatch.delenv("OMEGA_ENABLE_RECOMMENDATION_LAB", raising=False)


def _seed_matchup_db(db_path: str) -> None:
    store = TraceStore(db_path=db_path)
    with store.autolog_suppressed():
        store.persist(
            make_trace(
                "sandbox-game-1",
                league="MLB",
                kind="game",
                matchup="Red Sox @ Yankees",
                event_identity=dict(EVENT_IDENTITY),
                input_snapshot={
                    "league": "MLB",
                    "home_team": "Yankees",
                    "away_team": "Red Sox",
                },
                result={
                    "status": "success",
                    "context_source": "provided",
                    "simulation": {
                        "home_win_prob": 58.0,
                        "away_win_prob": 42.0,
                        "draw_prob": None,
                    },
                    "edges": [
                        {
                            "side": "home",
                            "team": "Yankees",
                            "market": "moneyline",
                            "calibrated_prob": 0.58,
                            "market_implied": 0.55,
                            "edge_pct": 3.0,
                            "ev_pct": 5.0,
                            "confidence_tier": "B",
                            "recommended_units": 1.0,
                            "market_odds": -120,
                        },
                        {
                            "side": "away",
                            "team": "Red Sox",
                            "market": "moneyline",
                            "calibrated_prob": 0.42,
                            "market_implied": 0.45,
                            "edge_pct": -3.0,
                            "ev_pct": -5.0,
                            "confidence_tier": "Pass",
                            "recommended_units": 0.0,
                            "market_odds": 100,
                        },
                    ],
                },
                calibration_audit=[
                    {
                        "profile_id": "iso_mlb_game_v8",
                        "profile_maturity": "production",
                        "sample_size": 400,
                        "ece": 0.03,
                    }
                ],
            )
        )
        # A no-recommendation legacy trace (no event identity, no edges).
        store.persist(
            make_trace(
                "sandbox-norec-1",
                league="NBA",
                kind="game",
                matchup="Celtics @ Lakers",
                recommendations=[],
                result={"status": "success", "context_source": "provided", "edges": []},
            )
        )
    store.close()


@pytest.fixture
def matchup_client(tmp_path, sessions_dir) -> TestClient:
    db_path = str(tmp_path / "matchups.db")
    _seed_matchup_db(db_path)
    write_valid_sidecar(sessions_dir, "sess-test-1")
    app = build_console_app(db_path=db_path, sessions_dir=str(sessions_dir))
    return TestClient(app)


class TestLabGate:
    @pytest.mark.parametrize("path", LAB_JSON_ROUTES + LAB_HTML_ROUTES)
    def test_lab_routes_404_when_flag_off(self, client, lab_off, path):
        assert client.get(path).status_code == 404, path

    @pytest.mark.parametrize("path", LAB_JSON_ROUTES + LAB_HTML_ROUTES)
    def test_lab_routes_open_when_flag_on(self, client, path):
        # conftest autouse fixture sets OMEGA_ENABLE_RECOMMENDATION_LAB=1
        assert client.get(path).status_code == 200, path

    @pytest.mark.parametrize("path", PRIMARY_ROUTES)
    def test_primary_routes_stay_reachable_without_lab(self, client, lab_off, path):
        assert client.get(path).status_code == 200, path

    def test_nav_hides_lab_links_when_off(self, client, lab_off):
        html = client.get("/").text
        for href in ('href="/scanner"', 'href="/bets"', 'href="/clv"'):
            assert href not in html

    def test_nav_shows_lab_links_when_on(self, client):
        html = client.get("/").text
        for href in ('href="/scanner"', 'href="/bets"', 'href="/clv"'):
            assert href in html


class TestMatchupApi:
    def test_list_groups_by_event_and_includes_no_recommendation_traces(
        self, matchup_client, lab_off
    ):
        briefs = matchup_client.get("/api/matchups").json()
        keys = {b["group_key"] for b in briefs}
        assert EVENT_IDENTITY["event_key"] in keys
        assert "trace:sandbox-norec-1" in keys
        legacy = next(b for b in briefs if b["group_key"] == "trace:sandbox-norec-1")
        assert legacy["identity_warning"] is True

    def test_brief_detail_shows_symmetric_probabilities(self, matchup_client, lab_off):
        brief = matchup_client.get(f"/api/matchups/{EVENT_IDENTITY['event_key']}").json()
        assert brief["event_key"] == EVENT_IDENTITY["event_key"]
        market = brief["markets"][0]
        probs = market["probabilities"]
        assert probs["disclosure"] == "shown"
        assert [o["outcome_key"] for o in probs["outcomes"]] == ["home", "away"]
        assert "not a recommendation" in probs["estimate_label"]

    def test_brief_detail_legacy_trace_key(self, matchup_client, lab_off):
        brief = matchup_client.get("/api/matchups/trace:sandbox-norec-1").json()
        assert brief["identity_warning"] is True
        assert brief["markets"][0]["probabilities"]["disclosure"] == "withheld"

    def test_unknown_matchup_404(self, matchup_client, lab_off):
        assert matchup_client.get("/api/matchups/trace:nope").status_code == 404
        assert matchup_client.get("/api/matchups/MLB::x::y").status_code == 404

    def test_primary_dto_contains_no_denied_keys_or_phrases(self, matchup_client, lab_off):
        from omega.core.contracts.language import blocked_language
        from omega.trace.decision_support import DENYLIST_KEYS

        for path in ("/api/matchups", f"/api/matchups/{EVENT_IDENTITY['event_key']}"):
            dumped = json.dumps(matchup_client.get(path).json())
            for key in sorted(DENYLIST_KEYS):
                assert f'"{key}"' not in dumped, f"{path} leaked {key}"
            assert blocked_language(dumped) == [], path

    def test_matchups_work_with_lab_on_too(self, matchup_client):
        assert matchup_client.get("/api/matchups").status_code == 200


def _get(obj: dict[str, Any], *path: str) -> Any:
    for key in path:
        obj = obj[key]
    return obj


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
