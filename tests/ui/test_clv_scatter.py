"""CLV scatter — closing-line value (x) vs net result (y), process vs luck.

Pins the doctrine: only bets with BOTH a closing line and a graded net result are
plotted (the rest are reported as excluded), quadrant guides mark CLV=0 / net=0,
and an empty source renders an honest empty state.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore
from omega.ui.service import ConsoleService
from tests.ui.conftest import make_trace


def _seed(db: str, *, with_close: bool = True) -> None:
    store = TraceStore(db_path=db)
    rows = [
        ("t-win", -150.0, -200.0, LedgerStatus.WON, 16.0),
        ("t-loss", -120.0, -110.0, LedgerStatus.LOST, -25.0),
    ]
    with store.autolog_suppressed():
        for tid, odds, close, status, pnl in rows:
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
                    odds=odds,
                    stake_amount=25.0,
                    net_pnl=pnl,
                    status=status,
                    provenance=BetProvenance.USER_CONFIRMED,
                    decision_timestamp="2026-03-21T12:00:00Z",
                )
            )
            if with_close:
                store.attach_closing_line(
                    tid, "moneyline", "home_moneyline", close, None, "2026-03-21T19:00:00Z", "dk"
                )
    store.close()


def _svc(db: str, sessions: Path) -> ConsoleService:
    return ConsoleService(TraceStore(db_path=db, read_only=True), sessions_dir=str(sessions))


def test_scatter_plots_graded_bets_with_quadrants(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db)
    svc = _svc(db, sessions)
    try:
        sc = svc.clv_scatter()
    finally:
        svc.close()
    assert sc.n_plotted == 2 and sc.n_excluded == 0
    assert sc.x0 is not None and sc.y0 is not None  # both quadrant guides in range
    assert sorted(p.tone for p in sc.points) == ["neg", "pos"]
    # every point carries pre-formatted, signed displays (no client math).
    assert all(p.clv_display[0] in "+-" and p.pnl_display[0] in "+-" for p in sc.points)


def test_scatter_excludes_bets_without_close(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db, with_close=False)
    svc = _svc(db, sessions)
    try:
        sc = svc.clv_scatter()
    finally:
        svc.close()
    # No closing lines -> clv_report yields no rows -> nothing to plot, honest empty.
    assert sc.n_plotted == 0
    assert sc.n_excluded == 2
    assert any(w.code == "no_graded_clv" for w in sc.warnings)


def test_clv_page_renders_scatter(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db)
    html = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions))).get("/clv").text
    assert "Process vs Luck" in html
    assert "scatter-dot" in html and "scatter-guide" in html


def test_scatter_api_read_only(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db)
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    assert client.get("/api/clv-scatter").status_code == 200
    assert client.post("/api/clv-scatter", json={}).status_code == 405
