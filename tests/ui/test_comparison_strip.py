"""Per-bet comparison strip — dumbbell (Omega vs market) and CLV ribbon.

Pins the doctrine: geometry is server-computed (fixed [0,1] domain so rows are
comparable), a genuinely-missing side renders one dot plus an honest note (never
a fabricated marker), and the two strip modes never mix units.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore
from omega.ui.schemas import ExtractedFieldModel, NormalizedRecommendationModel
from omega.ui.service import (
    ConsoleService,
    _model_vs_market_strip,
    _prob01,
    _prob_strip,
    _strip_x,
)
from tests.ui.conftest import make_trace


def _ef(value: object = None) -> ExtractedFieldModel:
    return ExtractedFieldModel(value=value, source="db_trace_payload")


def _rec(*, calibrated=None, raw=None, implied=None) -> NormalizedRecommendationModel:
    return NormalizedRecommendationModel(
        is_primary=True,
        rank=1,
        market=_ef("moneyline"),
        selection=_ef("home"),
        line=_ef(),
        odds=_ef(-150),
        raw_probability=_ef(raw),
        calibrated_probability=_ef(calibrated),
        implied_probability=_ef(implied),
        engine_edge=_ef(),
        computed_edge=_ef(),
        kelly_fraction=_ef(),
        recommended_units=_ef(),
        raw_confidence_tier=_ef(),
        display_confidence_band=_ef(),
    )


# --- pure geometry / builders ----------------------------------------------


def test_strip_x_is_clamped_and_linear():
    assert _strip_x(0.0, 0.0, 1.0, 200) == 8.0  # left pad
    assert _strip_x(1.0, 0.0, 1.0, 200) == 192.0  # right pad
    assert 99.0 <= _strip_x(0.5, 0.0, 1.0, 200) <= 101.0
    # out-of-range values clamp to the axis ends (never overflow the strip).
    assert _strip_x(-1.0, 0.0, 1.0, 200) == 8.0
    assert _strip_x(2.0, 0.0, 1.0, 200) == 192.0


def test_prob01_coerces_percent_scale():
    assert _prob01(0.58) == 0.58
    assert _prob01(58.0) == 0.58  # percent-scaled source must not clamp to 1.0
    assert _prob01(None) is None
    assert _prob01(-3) is None


def test_prob_strip_two_dots_with_signed_gap():
    s = _prob_strip(
        mode="model_vs_market",
        unit="win probability",
        primary=(0.6, "model", "Omega", "model"),
        secondary=(0.5, "market", "Market", "market"),
        gap=0.1,
    )
    assert s is not None
    assert [d.key for d in s.dots] == ["model", "market"]
    assert s.dots[0].display == "60.0%" and s.dots[1].display == "50.0%"
    assert s.seg_x1 is not None and s.seg_x2 is not None
    assert s.gap_display == "+10.00%" and s.gap_positive is True and s.seg_tone == "pos"


def test_prob_strip_single_side_keeps_note_and_no_fabrication():
    s = _prob_strip(
        mode="model_vs_market",
        unit="win probability",
        primary=(0.6, "model", "Omega", "model"),
        secondary=(None, "market", "Market", "market"),
        gap=None,
        missing_note="market price only",
    )
    assert s is not None and len(s.dots) == 1
    assert s.seg_x1 is None and s.gap_display is None
    assert s.note == "market price only"


def test_prob_strip_none_when_both_missing():
    assert (
        _prob_strip(
            mode="model_vs_market",
            unit="win probability",
            primary=(None, "model", "Omega", "model"),
            secondary=(None, "market", "Market", "market"),
            gap=None,
        )
        is None
    )


def test_model_vs_market_strip_prefers_calibrated_then_raw():
    assert _model_vs_market_strip(_rec(calibrated=0.62, raw=0.55, implied=0.5)).dots[0].value == 0.62
    assert _model_vs_market_strip(_rec(calibrated=None, raw=0.55, implied=0.5)).dots[0].value == 0.55


# --- integration -----------------------------------------------------------


def _seed(db: str) -> None:
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        store.persist(make_trace("t-a", kind="game", matchup="A @ B"))
        store.record_ledger_bet(
            LedgerBet(
                ledger_id="led-a",
                trace_id="t-a",
                bet_date="2026-03-21",
                league="NBA",
                sport="basketball",
                matchup="A @ B",
                market="moneyline",
                bookmaker="dk",
                selection="A ML",
                selection_descriptor="home_moneyline",
                odds=-150.0,
                stake_amount=25.0,
                net_pnl=16.0,
                status=LedgerStatus.WON,
                provenance=BetProvenance.USER_CONFIRMED,
                decision_timestamp="2026-03-21T12:00:00Z",
            )
        )
        store.attach_closing_line(
            "t-a", "moneyline", "home_moneyline", -200.0, None, "2026-03-21T19:00:00Z", "dk"
        )
    store.close()


def test_scanner_row_has_strip_and_page_renders_svg(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db)
    svc = ConsoleService(TraceStore(db_path=db, read_only=True), sessions_dir=str(sessions))
    try:
        view = svc.edge_scanner()
    finally:
        svc.close()
    assert view.rows and view.rows[0].strip is not None
    assert view.rows[0].strip.mode == "model_vs_market"
    html = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions))).get("/scanner").text
    assert "Omega vs Market" in html
    assert 'class="comp-strip"' in html and "strip-dot" in html


def test_clv_row_has_market_movement_ribbon(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db)
    svc = ConsoleService(TraceStore(db_path=db, read_only=True), sessions_dir=str(sessions))
    try:
        clv = svc.clv_report()
    finally:
        svc.close()
    r = clv.rows[0]
    assert r.strip is not None and r.strip.mode == "market_movement"
    assert [d.key for d in r.strip.dots] == ["taken", "closing"]
    assert r.strip.outcome == "won"
    # The ribbon axis is implied probability (incl. vig) — never mixed with the
    # dumbbell's "win probability" unit.
    assert "implied" in r.strip.unit


def test_trace_and_bet_detail_expose_strip(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    _seed(db)
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))
    assert 'class="comp-strip"' in client.get("/traces/t-a").text
    assert 'class="comp-strip"' in client.get("/bets/led-a").text


def test_scanner_escapes_malicious_matchup(tmp_path: Path):
    db = str(tmp_path / "c.db")
    sessions = tmp_path / "s"
    sessions.mkdir()
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        store.persist(make_trace("t-x", matchup="<script>alert(1)</script>"))
    store.close()
    html = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions))).get("/scanner").text
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
