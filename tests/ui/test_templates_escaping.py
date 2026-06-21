"""Hand-rolled templates must autoescape user/LLM-influenced strings (XSS guard)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from tests.ui.conftest import make_trace, write_valid_sidecar

SCRIPT = "<script>alert('xss')</script>"
IMG = "<img src=x onerror=alert('y')>"


def _client(tmp_path: Path) -> TestClient:
    db = str(tmp_path / "x.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        store.persist(
            make_trace(
                "sandbox-xss",
                session_id="sess-xss",
                matchup=SCRIPT,
                recommendations=[{"market": SCRIPT, "confidence_tier": "A"}],
            )
        )
        from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
        store.record_ledger_bet(
            LedgerBet(
                ledger_id="led-xss",
                trace_id="sandbox-xss",
                bet_date="2026-03-21",
                league="NBA",
                sport="basketball",
                matchup=IMG,
                market=IMG,
                bookmaker=SCRIPT,
                selection=IMG,
                selection_descriptor="moneyline",
                odds=-110.0,
                stake_amount=10.0,
                status=LedgerStatus.PENDING,
                provenance=BetProvenance.USER_CONFIRMED,
                decision_timestamp="2026-03-21T12:00:00Z",
            )
        )
    store.close()
    write_valid_sidecar(sessions, "sess-xss", agent_notes=IMG)
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def test_trace_pages_escape_injected_html(tmp_path: Path):
    client = _client(tmp_path)
    for path in ("/traces", "/traces/sandbox-xss"):
        text = client.get(path).text
        assert SCRIPT not in text, f"unescaped <script> in {path}"
        assert "&lt;script&gt;" in text, f"expected escaped script in {path}"


def test_session_page_escapes_injected_html(tmp_path: Path):
    text = _client(tmp_path).get("/sessions/sess-xss").text
    assert IMG not in text
    assert "&lt;img" in text


def test_bet_pages_escape_injected_html(tmp_path: Path):
    client = _client(tmp_path)
    for path in ("/bets", "/bets/led-xss"):
        text = client.get(path).text
        assert SCRIPT not in text, f"unescaped <script> in {path}"
        assert IMG not in text, f"unescaped <img> in {path}"
        assert "&lt;script&gt;" in text, f"expected escaped script in {path}"
        assert "&lt;img" in text, f"expected escaped img in {path}"
