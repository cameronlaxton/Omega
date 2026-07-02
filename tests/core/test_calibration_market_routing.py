"""_calibrate_audited routes the calibration market from the PLANE.

A player prop must look up a prop profile (market="prop"); a game side looks up
the game profile; a 3-way draw keeps its own market. Before this fix every
non-draw side collapsed onto market="game", letting a game profile calibrate
props (and vice-versa).
"""

from __future__ import annotations

import pytest

from omega.core.contracts import service


@pytest.fixture
def captured_market(monkeypatch):
    captured: dict[str, object] = {}

    def fake_apply(
        raw, league=None, context_hints=None, market="game", market_prob=None, substrate_ref=None
    ):
        captured["market"] = market
        captured["market_prob"] = market_prob
        captured["substrate_ref"] = substrate_ref
        return raw, {
            "raw_prob": raw,
            "calibrated_prob": raw,
            "method_resolved": None,
            "profile_id": None,
            "context_slice": None,
            "resolved_slice": None,
            "path": "static_identity",
        }

    monkeypatch.setattr(service, "apply_calibration_audited", fake_apply)
    return captured


def test_prop_plane_routes_to_prop_market(captured_market):
    service._calibrate_audited(0.6, league="NBA", plane="prop", market="over")
    assert captured_market["market"] == "prop"
    service._calibrate_audited(0.4, league="NBA", plane="prop", market="under")
    assert captured_market["market"] == "prop"


def test_game_plane_routes_to_game_market(captured_market):
    service._calibrate_audited(0.6, league="NBA", plane="game", market="home")
    assert captured_market["market"] == "game"
    service._calibrate_audited(0.6, league="NBA", plane="game", market="spread")
    assert captured_market["market"] == "game"


def test_draw_routes_to_draw_market(captured_market):
    service._calibrate_audited(0.3, league="EPL", plane="game", market="draw")
    assert captured_market["market"] == "draw"


def test_market_probability_threads_to_shared_calibration(captured_market):
    service._calibrate_audited(
        0.6, league="NBA", plane="prop", market="over", market_prob=0.52
    )
    assert captured_market["market"] == "prop"
    assert captured_market["market_prob"] == 0.52
