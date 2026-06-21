"""Session Review: list validated sidecars, handle malformed safely, separate
sidecar process/narrative from DB-backed numeric values."""

from __future__ import annotations

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from tests.ui.conftest import write_valid_sidecar


def _client(seeded) -> TestClient:
    return TestClient(
        build_console_app(db_path=seeded["db_path"], sessions_dir=str(seeded["sessions_dir"]))
    )


def test_valid_sidecars_are_listed(seeded):
    write_valid_sidecar(
        seeded["sessions_dir"],
        "sess-test-1",
        exec_stats={"analyses": 2},
        agent_notes="ran two analyses",
        league="NBA",
    )
    body = _client(seeded).get("/api/sessions").json()
    rows = {r["session_id"]: r for r in body["rows"]}
    assert "sess-test-1" in rows
    assert rows["sess-test-1"]["sidecar_valid"] is True
    # DB-backed trace count for this session (aaa + bbb).
    assert rows["sess-test-1"]["db_trace_count"] == 2
    assert rows["sess-test-1"]["event_count"] == 0


def test_malformed_sidecar_handled_safely(seeded):
    write_valid_sidecar(seeded["sessions_dir"], "sess-test-1")
    # A truncated / invalid JSON sidecar must not crash listing or detail.
    bad = seeded["sessions_dir"] / "sess-bad.json"
    bad.write_text('{ "session_id": "sess-bad", NOT JSON', encoding="utf-8")

    client = _client(seeded)
    listing = client.get("/api/sessions")
    assert listing.status_code == 200
    rows = {r["session_id"]: r for r in listing.json()["rows"]}
    assert rows["sess-bad"]["sidecar_valid"] is False
    assert rows["sess-bad"]["quality_gate_status"] == "unknown"

    detail = client.get("/api/sessions/sess-bad")
    assert detail.status_code == 200
    body = detail.json()
    assert body["sidecar_valid"] is False
    assert body["sidecar_error"]
    # No traces correlate to this bogus session_id; must not raise.
    assert body["db_traces"] == []


def test_session_detail_correlates_sidecar_session_id_to_db_traces(seeded):
    write_valid_sidecar(
        seeded["sessions_dir"],
        "sess-test-1",
        audit_events=[
            {
                "ts": "2026-03-21T11:30:00Z",
                "event_type": "preflight",
                "step": "bootstrap",
                "status": "ok",
                "assumptions": ["assumed-x"],
                "bugs": ["bug-y"],
                "trace_ids": ["sandbox-aaa"],
            }
        ],
    )
    body = _client(seeded).get("/api/sessions/sess-test-1").json()
    assert body["sidecar_valid"] is True
    db_ids = {t["trace_id"] for t in body["db_traces"]}
    assert db_ids == {"sandbox-aaa", "sandbox-bbb"}
    # Sidecar narrative surfaces separately.
    assert body["assumptions"] == ["assumed-x"]
    assert body["bugs"] == ["bug-y"]
    assert len(body["audit_events"]) == 1


def test_numeric_values_come_from_db_not_sidecar_prose(seeded):
    # The sidecar carries a misleading numeric in its (non-canonical) exec_stats.
    write_valid_sidecar(
        seeded["sessions_dir"],
        "sess-test-1",
        exec_stats={"aggregate_quality": 0.123, "edge_pct_claim": 77.0},
        agent_notes="totally real numbers: edge 77%",
    )
    body = _client(seeded).get("/api/sessions/sess-test-1").json()
    # Canonical numbers come from db_traces.
    qualities = {t["aggregate_quality"] for t in body["db_traces"]}
    assert qualities == {0.85}
    # Sidecar process block is labelled non-canonical and kept separate.
    assert body["field_sources"]["exec_stats"] == "sidecar_process"
    assert body["field_sources"]["db_traces"] == "db_trace_payload"
    # The misleading sidecar number is confined to exec_stats; it never becomes a
    # trace's aggregate_quality.
    assert body["exec_stats"]["aggregate_quality"] == 0.123
    assert all(t["aggregate_quality"] != 0.123 for t in body["db_traces"])


def test_legacy_sidecars_excluded_from_listing(seeded):
    write_valid_sidecar(seeded["sessions_dir"], "sess-test-1")
    # A *.legacy.json must not be surfaced as a separate session entry.
    (seeded["sessions_dir"] / "sess-test-1.legacy.json").write_text("{}", encoding="utf-8")
    body = _client(seeded).get("/api/sessions").json()
    names = {r["file_name"] for r in body["rows"]}
    assert "sess-test-1.legacy.json" not in names


def test_session_id_path_traversal_rejected(seeded):
    write_valid_sidecar(seeded["sessions_dir"], "sess-test-1")
    client = _client(seeded)
    # Encoded/relative/absolute traversal attempts must 404, never reading
    # outside the sessions dir (regex reject + resolved-parent backstop).
    for bad in (
        "..%2f..%2fomega_traces",
        "%2e%2e",
        "..%5c..%5csecret",  # encoded backslash
        "sess%2f..%2f..%2fsecret",
        "....%2f%2fsecret",
    ):
        assert client.get(f"/api/sessions/{bad}").status_code == 404, bad
