"""A3 — Signal Conflict Detector (omega.ui.insights.build_signal_conflict)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from omega.ui.insights import build_market_movement, build_signal_conflict
from omega.ui.normalizers import build_trace_recommendation_view
from tests.ui.conftest import make_trace


def _rec(*, side="home", market="moneyline", edge_pct=4.2, true_prob=0.6, calibrated=None, odds=-150):
    rec: dict = {"side": side, "market": market, "edge_pct": edge_pct, "odds": odds}
    if true_prob is not None:
        rec["true_prob"] = true_prob
    if calibrated is not None:
        rec["calibrated_prob"] = calibrated
    trace = {"trace_id": "sc", "kind": "game", "recommendations": [rec],
             "predictions": {}, "result": {}}
    return build_trace_recommendation_view(trace).recommendations[0]


def _sig(stype, direction, conf, applied=True):
    return {"signal_type": stype, "direction": direction, "confidence": conf, "applied": applied}


def _close(odds, market="moneyline"):
    return [{"market": market, "selection_descriptor": "home_moneyline",
             "closing_odds": odds, "closing_line": None, "source": "test"}]


def _conflict(rec, signals, *, application=None, mm=None):
    return build_signal_conflict(
        rec=rec, evidence_signals=signals, evidence_application=application, market_movement=mm
    )


def test_aligned_signals_are_low_conflict():
    sc = _conflict(_rec(), [_sig("usage", "home", 0.7), _sig("pace", "home", 0.65)])
    assert sc.conflict_level == "low"
    assert sc.dominant_conflict is None
    assert sc.supporting_count == 2 and sc.opposing_count == 0


def test_opposing_weight_triggers_signal_disagreement():
    sc = _conflict(_rec(), [_sig("usage", "home", 0.6), _sig("rest", "away", 0.8)])
    assert "signal_disagreement" in sc.conflicts
    assert sc.dominant_conflict == "signal_disagreement"
    assert sc.conflict_level == "high"  # opposition outweighs support
    assert any(w.code == "signal_disagreement" and w.severity == "fail" for w in sc.warnings)


def test_market_moved_against_positive_edge_is_market_conflict():
    rec = _rec(odds=-150, calibrated=0.62)
    mm = build_market_movement(rec=rec, closing_lines=_close(120))
    assert mm.direction == "against"
    sc = _conflict(rec, [_sig("usage", "home", 0.7), _sig("pace", "home", 0.65)], mm=mm)
    assert sc.conflicts == ["market_conflict"]
    assert sc.dominant_conflict == "market_conflict"
    assert sc.conflict_level == "high"


def test_thin_evidence_with_edge_is_model_edge_conflict():
    sc = _conflict(_rec(), [_sig("usage", "home", 0.7)])  # only one applied signal
    assert "model_edge_conflict" in sc.conflicts
    assert sc.dominant_conflict == "model_edge_conflict"
    assert sc.conflict_level == "medium"


def test_damping_marks_correlation_conflict():
    sc = _conflict(
        _rec(),
        [_sig("usage", "home", 0.7), _sig("pace", "home", 0.6)],
        application=[{"signal_type": "pace", "family_role": "damped"}],
    )
    assert "correlation_conflict" in sc.conflicts
    role_rows = {r.signal_type: r.family_role for r in sc.rows}
    assert role_rows.get("pace") == "damped"


def test_single_supporting_signal_is_dominant_single():
    # Two applied signals (avoids model-edge conflict) but only one takes a side.
    sc = _conflict(_rec(), [_sig("usage", "home", 0.7), _sig("pace", None, 0.5)])
    assert sc.dominant_conflict == "dominant_single_signal"
    assert sc.supporting_count == 1 and sc.neutral_count == 1


def _client(tmp_path: Path, setup: Callable[[TraceStore], None]) -> TestClient:
    db = str(tmp_path / "a3.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        setup(store)
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def test_trace_detail_api_surfaces_signal_conflict(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(make_trace("sc-1")))
    body = client.get("/api/traces/sc-1").json()
    sc = body["signal_conflict"]
    assert sc is not None
    assert sc["conflict_level"] in {"low", "medium", "high"}
    assert "Signal Conflict" in client.get("/traces/sc-1").text
