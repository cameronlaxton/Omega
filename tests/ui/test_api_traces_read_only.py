"""Trace list/detail API: DB-backed rows, pagination, filters, DB authority."""

from __future__ import annotations

from tests.ui.conftest import write_valid_sidecar


def test_trace_list_returns_db_rows(client):
    body = client.get("/api/traces").json()
    ids = {r["trace_id"] for r in body["rows"]}
    assert {"sandbox-aaa", "sandbox-bbb", "sandbox-ccc"} <= ids
    assert body["pagination"]["total"] == 3
    # DB-backed denormalized fields surface on the row.
    aaa = next(r for r in body["rows"] if r["trace_id"] == "sandbox-aaa")
    assert aaa["confidence_tiers"] == ["A"]
    assert "moneyline" in aaa["markets"]
    assert aaa["has_outcome"] is True  # outcome attached in fixture
    assert aaa["field_sources"]["confidence_tiers"] == "db_trace_payload"


def test_trace_detail_returns_db_payload_and_links(client):
    body = client.get("/api/traces/sandbox-aaa").json()
    assert body["trace_id"] == "sandbox-aaa"
    assert body["aggregate_quality"] == 0.85
    assert body["payload"]["trace_id"] == "sandbox-aaa"  # full JSON blob
    assert body["outcome"] is not None and body["outcome"]["home_score"] == 110
    # The user-confirmed bet is linked from bet_ledger.
    assert any(b["ledger_id"] == "led-aaa-1" for b in body["bets"])
    assert body["field_sources"]["outcome"] == "outcomes"
    assert body["field_sources"]["bets"] == "bet_ledger"


def test_trace_detail_404(client):
    assert client.get("/api/traces/nope").status_code == 404


def test_pagination(client):
    p1 = client.get("/api/traces?page=1&page_size=2").json()
    assert len(p1["rows"]) == 2
    assert p1["pagination"]["has_next"] is True
    assert p1["pagination"]["has_prev"] is False
    p2 = client.get("/api/traces?page=2&page_size=2").json()
    assert len(p2["rows"]) == 1
    assert p2["pagination"]["has_next"] is False
    # Disjoint pages.
    assert not ({r["trace_id"] for r in p1["rows"]} & {r["trace_id"] for r in p2["rows"]})


def test_filter_by_league(client):
    body = client.get("/api/traces?league=EPL").json()
    assert {r["trace_id"] for r in body["rows"]} == {"sandbox-ccc"}


def test_filter_by_kind(client):
    body = client.get("/api/traces?kind=prop").json()
    assert {r["trace_id"] for r in body["rows"]} == {"sandbox-bbb"}


def test_filter_by_confidence(client):
    # aaa and ccc carry the default tier-A recommendation; bbb is tier B.
    a = client.get("/api/traces?confidence=A").json()
    assert {r["trace_id"] for r in a["rows"]} == {"sandbox-aaa", "sandbox-ccc"}
    b = client.get("/api/traces?confidence=B").json()
    assert {r["trace_id"] for r in b["rows"]} == {"sandbox-bbb"}


def test_filter_by_session(client):
    body = client.get("/api/traces?session_id=sess-test-1").json()
    assert {r["trace_id"] for r in body["rows"]} == {"sandbox-aaa", "sandbox-bbb"}


def test_filter_by_date_range(client):
    only_22 = client.get("/api/traces?date_from=2026-03-22").json()
    assert {r["trace_id"] for r in only_22["rows"]} == {"sandbox-ccc"}
    upto_21 = client.get("/api/traces?date_to=2026-03-21").json()
    assert {r["trace_id"] for r in upto_21["rows"]} == {"sandbox-aaa", "sandbox-bbb"}


def test_sidecar_cannot_override_db_numeric_values(seeded):
    """A sidecar that puts a bogus number in its prose must NOT change the
    DB-sourced numeric the trace API returns."""
    # Sidecar for the same session, carrying a misleading 'aggregate_quality'
    # in its (non-canonical) exec_stats prose.
    write_valid_sidecar(
        seeded["sessions_dir"],
        "sess-test-1",
        exec_stats={"aggregate_quality": 999.0, "note": "do not trust me"},
    )
    from fastapi.testclient import TestClient

    from omega.ops.console_server import build_console_app

    client = TestClient(
        build_console_app(db_path=seeded["db_path"], sessions_dir=str(seeded["sessions_dir"]))
    )
    detail = client.get("/api/traces/sandbox-aaa").json()
    assert detail["aggregate_quality"] == 0.85  # DB value, not 999.0
    assert "999" not in str(detail["aggregate_quality"])
