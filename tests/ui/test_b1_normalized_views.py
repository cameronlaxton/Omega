"""Milestone B.1 — UI integration of the B.0 normalization layer.

Covers the new normalized read views surfaced on the existing read-only API and
HTML pages: the trace recommendation view + evidence coverage on trace detail,
session health on session detail, and the evidence-count column on the trace
list. All read-only; raw fields/JSON stay backward compatible and collapsed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from tests.ui.conftest import make_trace, write_valid_sidecar

SCRIPT = "<script>alert('xss')</script>"
IMG = "<img src=x onerror=alert('y')>"


def _client(
    tmp_path: Path,
    setup: Callable[[TraceStore], None],
    *,
    sidecars: list[dict] | None = None,
) -> TestClient:
    """Build a console TestClient over a fresh temp DB seeded by ``setup``."""
    db = str(tmp_path / "b1.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        setup(store)
    store.close()
    for sc in sidecars or []:
        write_valid_sidecar(sessions, **sc)
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))


def _prop_handoff_trace() -> dict:
    """The handoff's motivating example: over / +165 / over_prob 0.3675."""
    return {
        "trace_id": "prop-1",
        "run_id": "r-prop-1",
        "timestamp": "2026-03-21T12:00:00Z",
        "kind": "prop",
        "session_id": "sess-b1",
        "league": "NBA",
        "matchup": "LeBron James pts 25.5",
        "execution_mode": "sandbox_prop",
        "aggregate_quality": 0.8,
        "predictions": {"over_prob": 0.3675, "under_prob": 0.6325},
        "recommendations": {
            "recommendation": "over",
            "confidence_tier": "B",
            "kelly_fraction": 0.0384,
            "recommended_units": 3.84,
            "bet_side_odds": 165,
        },
        "result": {
            "recommendation": "over",
            "line": 25.5,
            "over_prob": 0.3675,
            "under_prob": 0.6325,
        },
        "trace_quality": {"aggregate_quality": 0.8, "calibration_eligible": True},
    }


def _walk_fields(node, found: list[dict]) -> None:
    """Collect every ExtractedField-shaped dict ({value, source, source_path})."""
    if isinstance(node, dict):
        if {"value", "source", "source_path"} <= node.keys():
            found.append(node)
        for v in node.values():
            _walk_fields(v, found)
    elif isinstance(node, list):
        for v in node:
            _walk_fields(v, found)


# ---------------------------------------------------------------------------
# Trace detail API
# ---------------------------------------------------------------------------


def test_trace_detail_returns_normalized_recommendations_with_source_paths(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(_prop_handoff_trace()))
    body = client.get("/api/traces/prop-1").json()
    rv = body["recommendation_view"]
    assert rv is not None
    assert rv["trace_id"] == "prop-1"
    assert rv["kind"] == "prop"
    rec = rv["recommendations"][0]
    assert rec["is_primary"] is True
    # Selection-aware probability mapped to the over side, with a source path.
    assert rec["selection"]["value"] == "over"
    assert rec["selection"]["source_path"] == "recommendations.recommendation"
    assert rec["raw_probability"]["value"] == 0.3675
    assert rec["raw_probability"]["source_path"] == "predictions.over_prob"
    assert rec["odds"]["value"] == 165
    # Every present field carries a source path; missing ones carry None.
    fields: list[dict] = []
    _walk_fields(rv["recommendations"], fields)
    for f in fields:
        if f["value"] is None:
            assert f["source_path"] is None
        else:
            assert f["source_path"] is not None


def test_trace_detail_keeps_existing_raw_fields_unchanged(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(_prop_handoff_trace()))
    body = client.get("/api/traces/prop-1").json()
    # Backward-compatible: raw fields are untouched, new key is additive.
    assert body["recommendations"]["recommendation"] == "over"
    assert body["recommendations"]["bet_side_odds"] == 165
    assert body["predictions"]["over_prob"] == 0.3675
    assert body["payload"]["trace_id"] == "prop-1"
    assert body["field_sources"]["recommendations"] == "db_trace_payload"
    assert "recommendation_view" in body  # the only new top-level key


def test_normalized_payload_is_json_serializable(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(_prop_handoff_trace()))
    resp = client.get("/api/traces/prop-1")
    assert resp.status_code == 200
    # Round-trips cleanly (no NaN/Inf/exotic types leaked into the response).
    reserialized = json.dumps(resp.json()["recommendation_view"])
    assert json.loads(reserialized)["trace_id"] == "prop-1"


def test_computed_fields_are_marked_computed(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(_prop_handoff_trace()))
    rec = client.get("/api/traces/prop-1").json()["recommendation_view"]["recommendations"][0]
    # Implied prob from confirmed American odds; band from raw tier — both computed.
    assert rec["implied_probability"]["computed"] is True
    assert rec["implied_probability"]["source_path"] == "computed:implied_from_american_odds"
    assert rec["display_confidence_band"]["computed"] is True
    assert rec["display_confidence_band"]["source_path"] == "computed:from_raw_confidence_tier"
    # A plainly-extracted field is not marked computed.
    assert rec["raw_probability"]["computed"] is False


def test_display_confidence_band_uses_neutral_language(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(_prop_handoff_trace()))
    rec = client.get("/api/traces/prop-1").json()["recommendation_view"]["recommendations"][0]
    assert rec["display_confidence_band"]["value"] == "medium confidence"
    # Never "A-Tier"/"B-Tier" phrasing in the displayed band.
    assert "tier" not in rec["display_confidence_band"]["value"].lower()


def test_raw_confidence_tier_is_audit_source_only(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(_prop_handoff_trace()))
    rec = client.get("/api/traces/prop-1").json()["recommendation_view"]["recommendations"][0]
    # The raw engine tier is preserved verbatim for audit, not polished/computed.
    assert rec["raw_confidence_tier"]["value"] == "B"
    assert rec["raw_confidence_tier"]["computed"] is False
    assert rec["raw_confidence_tier"]["source_path"] == "recommendations.confidence_tier"


def test_no_sidecar_numeric_source_in_normalized_payload(tmp_path):
    # A sidecar with a bogus numeric in its (non-canonical) prose must never feed
    # the normalized recommendation view.
    client = _client(
        tmp_path,
        lambda s: s.persist(_prop_handoff_trace()),
        sidecars=[{"session_id": "sess-b1", "exec_stats": {"fake_edge": 999.0}, "agent_notes": "edge 999"}],
    )
    rv = client.get("/api/traces/prop-1").json()["recommendation_view"]
    fields: list[dict] = []
    _walk_fields(rv, fields)
    assert fields  # sanity
    for f in fields:
        assert f["source"] != "sidecar_process"
    assert "sidecar_process" not in json.dumps(rv)
    assert "999" not in json.dumps(rv)


def test_game_trace_normalizes_multiple_recommendations(tmp_path):
    edges = [
        {"side": "home", "team": "Lakers", "market": "moneyline", "true_prob": 0.58,
         "calibrated_prob": 0.6, "market_odds": -150, "edge_pct": 4.2, "confidence_tier": "A"},
        {"side": "away", "team": "Celtics", "market": "moneyline", "true_prob": 0.42,
         "market_odds": 130, "edge_pct": 1.1, "confidence_tier": "C"},
    ]
    trace = make_trace("game-multi", kind="game", recommendations=edges)
    client = _client(tmp_path, lambda s: s.persist(trace))
    recs = client.get("/api/traces/game-multi").json()["recommendation_view"]["recommendations"]
    assert len(recs) == 2
    assert recs[0]["is_primary"] is True and recs[0]["rank"] == 0
    assert recs[1]["is_primary"] is False and recs[1]["rank"] == 1
    # Selection-aware: home maps to the home prob (0.58), not away's 0.42.
    assert recs[0]["selection"]["value"] == "home"
    assert recs[0]["raw_probability"]["value"] == 0.58


# ---------------------------------------------------------------------------
# Session detail API
# ---------------------------------------------------------------------------


def _session_setup(store: TraceStore) -> None:
    store.persist(make_trace("s-aaa", session_id="sess-h", league="NBA", kind="game"))
    store.persist(make_trace("s-bbb", session_id="sess-h", league="NBA", kind="prop"))
    store.attach_outcome("s-aaa", 110, 100)


def test_session_detail_returns_health_with_db_counts_and_warnings(tmp_path):
    client = _client(
        tmp_path,
        _session_setup,
        sidecars=[{"session_id": "sess-h", "league": "NBA"}],
    )
    body = client.get("/api/sessions/sess-h").json()
    h = body["health"]
    assert h is not None
    # DB-backed aggregates.
    assert h["total_traces"] == 2
    assert h["traces_with_outcomes"] == 1  # s-aaa graded
    assert h["traces_zero_evidence"] == 2  # no evidence seeded
    assert h["evidence_coverage_ratio"] == 0.0
    assert h["sidecar_valid"] is True
    codes = {w["code"] for w in h["warnings"]}
    assert "traces_no_evidence" in codes
    assert "no_outcomes" not in codes  # one trace IS graded
    assert body["field_sources"]["health"] == "db_trace_payload"


def test_session_health_present_even_when_sidecar_invalid(tmp_path):
    def setup(store: TraceStore) -> None:
        store.persist(make_trace("s-aaa", session_id="sess-bad", kind="game"))

    db = str(tmp_path / "b1.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        setup(store)
    store.close()
    (sessions / "sess-bad.json").write_text('{ "session_id": "sess-bad", NOT JSON', encoding="utf-8")
    client = TestClient(build_console_app(db_path=db, sessions_dir=str(sessions)))

    h = client.get("/api/sessions/sess-bad").json()["health"]
    assert h is not None
    assert h["sidecar_valid"] is False
    assert h["total_traces"] == 1
    assert "sidecar_invalid" in {w["code"] for w in h["warnings"]}


def test_session_health_pipeline_failure_from_audit_event(tmp_path):
    client = _client(
        tmp_path,
        _session_setup,
        sidecars=[{
            "session_id": "sess-h",
            "audit_events": [
                {"ts": "2026-03-21T11:30:00Z", "event_type": "step", "step": "analysis", "status": "fail"}
            ],
        }],
    )
    h = client.get("/api/sessions/sess-h").json()["health"]
    assert "analysis" in h["pipeline_steps_failed"]
    warn = {w["code"]: w for w in h["warnings"]}
    assert warn["pipeline_failures"]["severity"] == "fail"
    assert warn["failed_audits"]["severity"] == "warn"


# ---------------------------------------------------------------------------
# Trace list evidence column
# ---------------------------------------------------------------------------


def test_trace_list_shows_evidence_counts_for_visible_rows(tmp_path):
    def setup(store: TraceStore) -> None:
        # A trace carrying one applied evidence signal.
        t = make_trace(
            "ev-1",
            kind="game",
            input_snapshot={"evidence": [{"signal_type": "injury", "confidence": 0.8, "window": "last_5"}]},
            evidence_application=[{"applied": True, "factor": 1.1}],
        )
        store.persist(t)
        store.persist(make_trace("ev-0", kind="game"))  # no evidence

    client = _client(tmp_path, setup)
    rows = {r["trace_id"]: r for r in client.get("/api/traces").json()["rows"]}
    assert rows["ev-1"]["evidence_coverage"]["total_signals"] == 1
    assert rows["ev-1"]["evidence_coverage"]["applied_signals"] == 1
    assert rows["ev-0"]["evidence_coverage"]["total_signals"] == 0
    assert rows["ev-1"]["field_sources"]["evidence_coverage"] == "evidence_signals"
    # The HTML table renders the evidence column.
    html = client.get("/traces").text
    assert "<th>evidence</th>" in html


# ---------------------------------------------------------------------------
# HTML rendering: structured cards, escaping, raw-JSON-collapsed
# ---------------------------------------------------------------------------


def test_trace_detail_structured_card_present_and_raw_json_collapsed(tmp_path):
    client = _client(tmp_path, lambda s: s.persist(_prop_handoff_trace()))
    html = client.get("/traces/prop-1").text
    # Structured card with computed badge above the raw, collapsed JSON.
    assert "(normalized)" in html
    assert "cbadge" in html  # computed badge rendered
    assert "medium confidence" in html  # neutral band
    # Raw JSON remains available but collapsed in <details>.
    assert "<details" in html
    assert "Show raw recommendations JSON" in html
    assert "Show raw trace JSON" in html
    # Structured view precedes the full raw payload.
    assert html.index("(normalized)") < html.index("Show raw trace JSON")


def test_trace_detail_card_escapes_injected_html(tmp_path):
    def setup(store: TraceStore) -> None:
        store.persist(
            make_trace("xss-card", kind="game", matchup=SCRIPT,
                       recommendations=[{"side": SCRIPT, "market": SCRIPT, "confidence_tier": "A"}])
        )

    html = _client(tmp_path, setup).get("/traces/xss-card").text
    assert SCRIPT not in html
    assert "&lt;script&gt;" in html


def test_session_health_warning_text_is_escaped(tmp_path):
    # A failing audit step whose name contains HTML flows into a health warning
    # message; it must be escaped when rendered.
    client = _client(
        tmp_path,
        _session_setup,
        sidecars=[{
            "session_id": "sess-h",
            "audit_events": [
                {"ts": "2026-03-21T11:30:00Z", "event_type": "step", "step": IMG, "status": "fail"}
            ],
        }],
    )
    # Confirm it actually reaches the warning, then confirm it is escaped in HTML.
    h = client.get("/api/sessions/sess-h").json()["health"]
    assert IMG in h["pipeline_steps_failed"]
    html = client.get("/sessions/sess-h").text
    assert IMG not in html
    assert "&lt;img" in html


# ---------------------------------------------------------------------------
# Read-only guard (no mutation routes introduced by B.1)
# ---------------------------------------------------------------------------


def test_no_mutation_routes_introduced(tmp_path):
    app = build_console_app(db_path=str(tmp_path / "x.db"), sessions_dir=str(tmp_path))
    offenders = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        for m in methods:
            if m not in {"GET", "HEAD", "OPTIONS"}:
                offenders.append(f"{getattr(route, 'path', route)}: {m}")
    assert offenders == [], f"non-read routes present: {offenders}"
