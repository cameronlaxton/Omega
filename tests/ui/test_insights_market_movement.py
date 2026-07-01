"""A2 — Market Movement Explainer (omega.ui.insights.build_market_movement)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from omega.ui.insights import build_market_movement, clv_interpretation
from omega.ui.normalizers import build_trace_recommendation_view
from tests.ui.conftest import make_trace


def _rec(*, odds, side="home", market="moneyline", true_prob=None, calibrated=None, line=None):
    rec: dict = {"side": side, "market": market, "odds": odds}
    if line is not None:
        rec["line"] = line
    if true_prob is not None:
        rec["true_prob"] = true_prob
    if calibrated is not None:
        rec["calibrated_prob"] = calibrated
    trace = {"trace_id": "mm", "kind": "game", "recommendations": [rec],
             "predictions": {}, "result": {}}
    view = build_trace_recommendation_view(trace)
    return view.recommendations[0]


def _close(odds, line=None, market="moneyline", side="home"):
    descriptor = "home_moneyline" if market == "moneyline" else f"{market}_{side}"
    return [{"market": market, "selection_descriptor": descriptor,
             "closing_odds": odds, "closing_line": line, "source": "test"}]


def test_clv_interpretation_token():
    assert clv_interpretation(0.05) == "market_confirms"
    assert clv_interpretation(-0.05) == "market_disagrees"
    assert clv_interpretation(0.0) == "no_confirmation"
    assert clv_interpretation(None) == "no_confirmation"


def test_market_moved_toward_with_large_edge_is_early_value():
    # taken +150 (implied .40) vs close -110 (implied .524): clv +0.124; model .60.
    mm = build_market_movement(rec=_rec(odds=150, calibrated=0.60), closing_lines=_close(-110))
    assert mm.available is True
    assert mm.direction == "toward"
    assert mm.clv_points > 0
    assert mm.interpretation == "early_value"
    assert mm.residual_edge is None
    assert mm.model_probability is None


def test_market_moved_toward_but_value_absorbed():
    mm = build_market_movement(rec=_rec(odds=150, calibrated=0.525), closing_lines=_close(-110))
    assert mm.direction == "toward"
    assert mm.interpretation == "value_absorbed"


def test_market_moved_against():
    # taken -150 (.60) vs close +120 (.4545): clv negative.
    mm = build_market_movement(rec=_rec(odds=-150, calibrated=0.62), closing_lines=_close(120))
    assert mm.direction == "against"
    assert mm.interpretation == "market_disagrees"
    assert mm.clv_points < 0


def test_flat_market_is_no_confirmation():
    mm = build_market_movement(rec=_rec(odds=-110, calibrated=0.55), closing_lines=_close(-110))
    assert mm.direction == "flat"
    assert mm.interpretation == "no_confirmation"


def test_no_closing_line_is_insufficient_data():
    mm = build_market_movement(rec=_rec(odds=-110, calibrated=0.55), closing_lines=[])
    assert mm.available is False
    assert mm.interpretation == "insufficient_data"
    assert any(w.code == "no_closing_line" for w in mm.warnings)


def test_no_recommendation_is_insufficient_data():
    mm = build_market_movement(rec=None, closing_lines=_close(-110))
    assert mm.available is False
    assert mm.interpretation == "insufficient_data"
    assert any(w.code == "no_recommendation" for w in mm.warnings)


def test_point_delta_when_lines_present():
    mm = build_market_movement(
        rec=_rec(odds=-110, side="over", market="total", true_prob=0.55),
        closing_lines=_close(-110, line=224.5, market="total", side="over"),
    )
    # taken line comes from the rec; here only the close has a line, so point_delta
    # needs both — assert it is None when the taken line is absent (no guessing).
    assert mm.point_delta is None
    assert mm.closing_line == 224.5


def test_total_point_move_same_price_counts_as_market_movement():
    mm = build_market_movement(
        rec=_rec(odds=-110, side="over", market="total", true_prob=0.55, line=224.5),
        closing_lines=_close(-110, line=226.5, market="total", side="over"),
    )
    assert mm.available is True
    assert mm.clv_points == 0
    assert mm.point_delta == 2.0
    assert mm.direction == "toward"
    assert mm.interpretation in {"early_value", "market_confirms", "value_absorbed"}


def _client(tmp_path: Path, setup: Callable[[TraceStore], None]) -> TestClient:
    db = str(tmp_path / "a2.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        setup(store)
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def test_trace_detail_api_surfaces_market_movement(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(make_trace("mm-1")))
    body = client.get("/api/traces/mm-1").json()
    mm = body["market_movement"]
    assert mm is not None
    assert mm["interpretation"] in {
        "early_value", "market_confirms", "value_absorbed",
        "market_disagrees", "no_confirmation", "insufficient_data",
    }
    # The seeded trace has no closing line captured -> insufficient data, honestly.
    assert mm["available"] is False
    assert "Market Movement" in client.get("/traces/mm-1").text
