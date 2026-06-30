"""A4 — Similar Historical Spots (ConsoleService.similar_spots)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore
from tests.ui.conftest import make_trace


def _bet(idx: int, trace_id: str, status: LedgerStatus, *, league="NBA", market="moneyline") -> LedgerBet:
    return LedgerBet(
        ledger_id=f"led-{idx}",
        trace_id=trace_id,
        bet_date="2026-03-21",
        league=league,
        sport="basketball",
        matchup="A @ B",
        market=market,
        bookmaker="dk",
        selection="home",
        selection_descriptor="home_moneyline",
        odds=-150.0,
        stake_amount=10.0,
        status=status,
        provenance=BetProvenance.USER_CONFIRMED,
        decision_timestamp="2026-03-21T12:00:00Z",
    )


def _seed_cohort(store: TraceStore, *, n_win: int, n_loss: int, edge: float = 4.2) -> None:
    """Seed N NBA moneyline traces (edge bucket fixed by ``edge``) + settled bets."""
    idx = 0
    for status, count in ((LedgerStatus.WON, n_win), (LedgerStatus.LOST, n_loss)):
        for _ in range(count):
            tid = f"hist-{idx}"
            store.persist(
                make_trace(
                    tid,
                    league="NBA",
                    kind="game",
                    recommendations=[
                        {"side": "home", "market": "moneyline", "confidence_tier": "B", "edge_pct": edge}
                    ],
                )
            )
            store.record_ledger_bet(_bet(idx, tid, status))
            idx += 1
    # The target spot (no bet of its own): same structure as the cohort.
    store.persist(
        make_trace(
            "target",
            league="NBA",
            kind="game",
            recommendations=[
                {"side": "home", "market": "moneyline", "confidence_tier": "B", "edge_pct": edge}
            ],
        )
    )


def _client(tmp_path: Path, *, n_win: int, n_loss: int) -> TestClient:
    db = str(tmp_path / "a4.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        _seed_cohort(store, n_win=n_win, n_loss=n_loss)
    store.close()
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def _structural(view: dict) -> dict:
    return next(c for c in view["cohorts"] if c["code"] == "structural")


def test_strong_cohort_yields_strong_support(tmp_path):
    client = _client(tmp_path, n_win=15, n_loss=10)  # 25 decided, 60% hit
    view = client.get("/api/traces/target/similar").json()
    assert view["available"] is True
    assert view["league"] == "NBA" and view["market_family"] == "moneyline"
    assert view["edge_bucket"] == "3-6%"
    s = _structural(view)
    assert s["wins"] == 15 and s["losses"] == 10
    assert abs(s["hit_rate"] - 0.6) < 1e-9
    assert s["thin_sample"] is False
    assert view["historical_support"] == "strong"


def test_losing_cohort_yields_weak_support(tmp_path):
    client = _client(tmp_path, n_win=8, n_loss=17)  # 25 decided, 32% hit
    view = client.get("/api/traces/target/similar").json()
    assert view["historical_support"] == "weak"


def test_thin_history_is_insufficient_and_flagged(tmp_path):
    client = _client(tmp_path, n_win=3, n_loss=2)  # only 5 decided
    view = client.get("/api/traces/target/similar").json()
    assert view["historical_support"] == "insufficient"
    assert _structural(view)["thin_sample"] is True
    assert any(w["code"] == "thin_history" for w in view["warnings"])


def test_target_excluded_from_its_own_cohort(tmp_path):
    # 15 winners + a target that, if counted, would change totals. Target has no
    # bet, so the structural decided count must equal the cohort size exactly.
    client = _client(tmp_path, n_win=15, n_loss=10)
    view = client.get("/api/traces/target/similar").json()
    s = _structural(view)
    assert s["wins"] + s["losses"] == 25


def test_similar_page_renders_and_404(tmp_path):
    client = _client(tmp_path, n_win=15, n_loss=10)
    html = client.get("/traces/target/similar").text
    assert "Historical support" in html
    assert "strong" in html
    assert client.get("/api/traces/nope/similar").status_code == 404
    assert client.get("/traces/nope/similar").status_code == 404


def test_seeded_single_bet_is_insufficient(tmp_path):
    # The default seeded fixture has a single settled bet -> never a verdict.
    db = str(tmp_path / "a4b.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        store.persist(make_trace("solo"))
    store.record_ledger_bet(_bet(99, "solo", LedgerStatus.WON))
    store.close()
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    view = client.get("/api/traces/solo/similar").json()
    # The only settled bet is the target's own -> excluded -> no comparable history.
    assert view["historical_support"] == "insufficient"
