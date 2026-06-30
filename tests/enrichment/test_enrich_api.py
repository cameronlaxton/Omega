"""B6 — the enrichment write API (TestClient runs the background task inline)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.enrichment.api import build_enrichment_app
from omega.enrichment.schemas import RECOMMENDATION_TYPES

_INTENT = {"X-Omega-Enrich-Intent": "operator-console"}


def _client(traces_db: str, enrich_db: str, tmp_path: Path) -> TestClient:
    app = build_enrichment_app(
        enrich_db=enrich_db, console_db=traces_db, sessions_dir=str(tmp_path),
        provider_name="stub",
    )
    return TestClient(app)


def test_enqueue_runs_and_completes(traces_db: str, enrich_db: str, tmp_path: Path):
    client = _client(traces_db, enrich_db, tmp_path)
    assert client.post("/traces/enr-1").status_code == 403
    resp = client.post("/traces/enr-1", headers=_INTENT)
    assert resp.status_code == 200
    eid = resp.json()["enrichment_id"]
    assert resp.json()["status"] == "queued"

    # Starlette TestClient drains background tasks before returning, so it's done.
    rec = client.get(f"/enrichments/{eid}").json()
    assert rec["status"] == "completed"
    assert rec["narrative_md"]
    assert rec["result"]["recommendation_type"] in RECOMMENDATION_TYPES


def test_latest_and_list_and_feedback(traces_db: str, enrich_db: str, tmp_path: Path):
    client = _client(traces_db, enrich_db, tmp_path)
    eid = client.post("/traces/enr-1", headers=_INTENT).json()["enrichment_id"]
    other = client.post("/traces/enr-2", headers=_INTENT).json()["enrichment_id"]

    latest = client.get("/traces/enr-1/latest").json()["latest"]
    assert latest is not None and latest["id"] == eid
    assert latest["id"] != other

    listing = client.get("/traces/enr-1/enrichments").json()
    assert any(e["id"] == eid for e in listing["enrichments"])
    assert all(e["id"] != other for e in listing["enrichments"])

    assert client.post(f"/enrichments/{eid}/feedback", json={"user_rating": 1}).status_code == 403
    fb = client.post(
        f"/enrichments/{eid}/feedback",
        headers=_INTENT,
        json={"user_rating": 1, "feedback_text": "ok"},
    )
    assert fb.status_code == 200 and fb.json()["status"] == "recorded"


def test_unknown_enrichment_and_feedback_404(traces_db: str, enrich_db: str, tmp_path: Path):
    client = _client(traces_db, enrich_db, tmp_path)
    assert client.get("/enrichments/nope").status_code == 404
    assert client.post(
        "/enrichments/nope/feedback", headers=_INTENT, json={"user_rating": 1}
    ).status_code == 404


def test_latest_is_null_before_any_generation(traces_db: str, enrich_db: str, tmp_path: Path):
    client = _client(traces_db, enrich_db, tmp_path)
    assert client.get("/traces/enr-1/latest").json()["latest"] is None
