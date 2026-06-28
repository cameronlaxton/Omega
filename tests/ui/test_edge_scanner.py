"""Honest Edge Scanner — recent DB-backed recommendations, no fabricated score.

Pins the doctrine: market-aware Model Output labels, a defined confidence source
hierarchy, ranking by real engine edge, and the absence of any "Value Score",
"Best Price", or BET affordance.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from omega.ui.service import ConsoleService, _normalize_edge_pct
from tests.ui.conftest import make_trace


def _svc(seeded) -> ConsoleService:
    return ConsoleService(
        TraceStore(db_path=seeded["db_path"], read_only=True),
        sessions_dir=str(seeded["sessions_dir"]),
    )


def test_scanner_page_renders(client):
    html = client.get("/scanner").text
    assert "Edge Scanner" in html
    assert "Recorded Price" in html
    assert "Model Output" in html


def test_scanner_is_honest_no_bet_no_value_score(client):
    html = client.get("/scanner").text
    assert "Value Score" not in html
    assert "Best Price" not in html
    assert ">BET<" not in html and ">Bet<" not in html


def test_scanner_market_aware_labels(seeded):
    svc = _svc(seeded)
    try:
        view = svc.edge_scanner()
    finally:
        svc.close()
    by_trace = {r.trace_id: r for r in view.rows}
    # Moneyline -> model probability (percentage); player prop -> recorded market line.
    assert by_trace["sandbox-aaa"].model_output_label == "Model Probability"
    assert by_trace["sandbox-aaa"].model_output_is_pct is True
    assert by_trace["sandbox-bbb"].model_output_label == "Recorded Line"
    assert by_trace["sandbox-bbb"].model_output_is_pct is False


def test_scanner_confidence_prefers_engine_tier(seeded):
    svc = _svc(seeded)
    try:
        view = svc.edge_scanner()
    finally:
        svc.close()
    aaa = next(r for r in view.rows if r.trace_id == "sandbox-aaa")
    assert aaa.confidence == "A"
    assert aaa.confidence_source == "model_confidence_tier"
    assert aaa.confidence_computed is False


def test_scanner_confidence_unavailable_without_engine_tier(tmp_path: Path):
    db = str(tmp_path / "s.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        # No confidence_tier on the rec -> the console must not synthesize one.
        store.persist(
            make_trace(
                "no-tier",
                recommendations=[{"side": "home", "market": "moneyline", "edge_pct": 3.0}],
            )
        )
    store.close()
    svc = ConsoleService(TraceStore(db_path=db, read_only=True), sessions_dir=str(sessions))
    try:
        view = svc.edge_scanner()
    finally:
        svc.close()
    r = next(x for x in view.rows if x.trace_id == "no-tier")
    assert r.confidence is None
    assert r.confidence_source == "unavailable"
    assert r.confidence_computed is False


def test_scanner_edge_display_normalizes_units(tmp_path: Path):
    # edge_pct=4.0 (percent form) must render "+4.00%", NOT "400%"; edge=0.03
    # (fraction form) must render "+3.00%".
    db = str(tmp_path / "s.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        store.persist(make_trace("pct", recommendations=[{"side": "home", "market": "moneyline", "edge_pct": 4.0}]))
        store.persist(make_trace("frac", recommendations=[{"side": "home", "market": "moneyline", "edge": 0.03}]))
    store.close()
    svc = ConsoleService(TraceStore(db_path=db, read_only=True), sessions_dir=str(sessions))
    try:
        view = svc.edge_scanner()
    finally:
        svc.close()
    by_trace = {r.trace_id: r for r in view.rows}
    assert by_trace["pct"].edge_display == "+4.00%"
    assert by_trace["frac"].edge_display == "+3.00%"
    assert by_trace["pct"].edge_positive is True


def test_scanner_ranked_by_edge(seeded):
    svc = _svc(seeded)
    try:
        view = svc.edge_scanner()
    finally:
        svc.close()
    edges = [
        _normalize_edge_pct(r.edge.value, r.edge.source_path)
        for r in view.rows
        if r.edge.value is not None
    ]
    assert edges == sorted(edges, reverse=True)


def test_scanner_api_endpoint_is_read_only(client):
    resp = client.get("/api/scanner")
    assert resp.status_code == 200
    body = resp.json()
    assert "rows" in body and len(body["rows"]) >= 1
    assert client.post("/api/scanner", json={}).status_code == 405


def test_scanner_empty_state(tmp_path: Path):
    db = str(tmp_path / "empty.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    TraceStore(db_path=db).close()
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    html = client.get("/scanner").text
    assert "No recent DB-backed recommendations found." in html
