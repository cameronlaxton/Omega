"""Calibration reliability diagram — model probability bucket vs realized hit rate.

Pins the doctrine: server-computed geometry with a y=x diagonal, buckets below
``min_n`` are SUPPRESSED (never drawn), and too-few graded outcomes render an
honest empty state rather than a fabricated point.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore
from omega.ui.service import ConsoleService, _scatter_geometry
from tests.ui.conftest import make_trace


def test_scatter_geometry_maps_corners_and_zero_guides():
    pts, x0, y0 = _scatter_geometry(
        [(0.0, 0.0), (1.0, 1.0)], x_lo=0.0, x_hi=1.0, y_lo=0.0, y_hi=1.0, view_w=360, view_h=360
    )
    (bx, by), (tx, ty) = pts
    assert bx < tx  # x increases left→right
    assert by > ty  # y is inverted (0 at the bottom)
    assert x0 is not None and y0 is not None  # 0 is inside [0,1]
    # When 0 is outside the domain, no guide pixel is returned.
    _, x0b, y0b = _scatter_geometry(
        [(5.0, 5.0)], x_lo=1.0, x_hi=10.0, y_lo=1.0, y_hi=10.0, view_w=100, view_h=100
    )
    assert x0b is None and y0b is None


def _seed(db: str, *, won: int, lost: int) -> None:
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        i = 0
        for status, count in ((LedgerStatus.WON, won), (LedgerStatus.LOST, lost)):
            for _ in range(count):
                tid = f"t-{i}"
                i += 1
                store.persist(make_trace(tid, matchup="A @ B"))
                store.record_ledger_bet(
                    LedgerBet(
                        ledger_id="l-" + tid,
                        trace_id=tid,
                        bet_date="2026-03-21",
                        league="NBA",
                        sport="basketball",
                        matchup="A @ B",
                        market="moneyline",
                        bookmaker="dk",
                        selection="home",
                        selection_descriptor="home_moneyline",
                        odds=-150.0,
                        stake_amount=25.0,
                        net_pnl=(10.0 if status == LedgerStatus.WON else -25.0),
                        status=status,
                        provenance=BetProvenance.USER_CONFIRMED,
                        decision_timestamp="2026-03-21T12:00:00Z",
                    )
                )
    store.close()


def _svc(db: str, sessions: Path) -> ConsoleService:
    return ConsoleService(TraceStore(db_path=db, read_only=True), sessions_dir=str(sessions))


def test_reliability_bins_realized_hit_rate(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db, won=4, lost=2)  # all share model prob ~0.58 -> one bucket, n=6
    svc = _svc(db, sessions)
    try:
        diag = svc.reliability_diagram(min_n=1)
    finally:
        svc.close()
    assert diag.n_pairs == 6
    assert diag.n_plotted == 1
    b = diag.bins[0]
    assert b.n == 6
    assert abs(b.hit_rate - 4 / 6) < 1e-3  # hit_rate is rounded to 4dp in the model
    # diagonal spans the plot (and y is inverted).
    assert diag.diag_x1 < diag.diag_x2 and diag.diag_y1 > diag.diag_y2


def test_reliability_suppresses_low_n_buckets(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db, won=2, lost=1)  # n=3 < min_n
    svc = _svc(db, sessions)
    try:
        diag = svc.reliability_diagram(min_n=5)
    finally:
        svc.close()
    assert diag.n_pairs == 3
    assert diag.n_plotted == 0 and diag.bins == []
    assert any(w.code == "insufficient_graded_outcomes" for w in diag.warnings)


def test_reliability_empty_db_is_honest(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    TraceStore(db_path=db).close()
    svc = _svc(db, sessions)
    try:
        diag = svc.reliability_diagram()
    finally:
        svc.close()
    assert diag.bins == [] and diag.n_pairs == 0
    assert any(w.code == "insufficient_graded_outcomes" for w in diag.warnings)


def test_calibration_page_renders_reliability(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db, won=4, lost=2)
    html = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions))).get("/calibration").text
    assert "Reliability" in html
    assert "reliability-diag" in html and "reliability-dot" in html


def test_reliability_api_read_only(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db, won=4, lost=2)
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    assert client.get("/api/reliability").status_code == 200
    assert client.post("/api/reliability", json={}).status_code == 405
