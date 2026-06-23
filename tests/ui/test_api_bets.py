"""Bet list/detail API: DB-backed rows, trace linkage, settlement/staking fields."""

from __future__ import annotations


def test_bet_list_returns_db_rows(client):
    body = client.get("/api/bets").json()
    assert body["pagination"]["total"] == 1
    row = body["rows"][0]
    assert row["ledger_id"] == "led-aaa-1"
    assert row["status"] == "won"
    assert row["net_pnl"] == 16.67
    assert row["bookmaker"] == "draftkings"
    assert row["provenance"] == "user_confirmed"
    assert row["field_sources"]["net_pnl"] == "bet_ledger"


def test_bet_filters(client):
    assert client.get("/api/bets?status=won").json()["pagination"]["total"] == 1
    assert client.get("/api/bets?status=lost").json()["pagination"]["total"] == 0
    assert client.get("/api/bets?bookmaker=draftkings").json()["pagination"]["total"] == 1
    assert client.get("/api/bets?bookmaker=fanduel").json()["pagination"]["total"] == 0
    assert client.get("/api/bets?league=NBA").json()["pagination"]["total"] == 1


def test_bet_detail_links_to_trace(client):
    body = client.get("/api/bets/led-aaa-1").json()
    assert body["ledger_id"] == "led-aaa-1"
    assert body["trace_id"] == "sandbox-aaa"
    # Recommendation values come from the linked trace's DB payload.
    assert body["linked_trace_id"] == "sandbox-aaa"
    assert body["linked_trace_recommendations"] is not None
    recs = body["linked_trace_recommendations"]
    assert any(r.get("edge_pct") == 4.2 for r in recs)
    assert body["field_sources"]["linked_trace_recommendations"] == "db_trace_payload"


def test_bet_detail_preserves_settlement_and_staking(client):
    body = client.get("/api/bets/led-aaa-1").json()
    ledger = body["ledger"]
    # Settlement / PnL.
    assert ledger["status"] == "won"
    assert ledger["payout_amount"] == 41.67
    assert ledger["net_pnl"] == 16.67
    # Staking policy metadata.
    assert body["staking"]["staking_policy_id"] == "sp-flat-1"
    assert body["staking"]["staking_policy_version"] == 1
    assert body["staking"]["exposure_limits_version"] == 2
    assert body["staking"]["sizing_reasons"] == ["max_exposure_cap"]
    # Correlation group.
    assert body["correlation_group"] == "nba-2026-03-21"


def test_bet_detail_does_not_assume_edge_in_ledger(client):
    """edge%/Kelly/units must not be read from bet_ledger — they live only in the
    linked trace payload."""
    body = client.get("/api/bets/led-aaa-1").json()
    assert "edge_pct" not in body["ledger"]
    assert "kelly_fraction" not in body["ledger"]
    assert "recommended_units" not in body["ledger"]


def test_bet_detail_404(client):
    assert client.get("/api/bets/nope").status_code == 404


def test_bet_numbers_unaffected_by_sidecar(seeded):
    """The bet endpoints never read sidecars; a misleading sidecar number must
    not change a bet_ledger value."""
    from fastapi.testclient import TestClient

    from omega.ops.console_server import build_console_app
    from tests.ui.conftest import write_valid_sidecar

    write_valid_sidecar(seeded["sessions_dir"], "sess-test-1", exec_stats={"net_pnl": 9999.0})
    client = TestClient(
        build_console_app(db_path=seeded["db_path"], sessions_dir=str(seeded["sessions_dir"]))
    )
    body = client.get("/api/bets/led-aaa-1").json()
    assert body["ledger"]["net_pnl"] == 16.67  # bet_ledger value, not 9999.0
