"""Milestone B.2 — Diagnostics page: read-only runtime/registry/scoring health."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus
from omega.core.calibration.registry import CalibrationRegistry
from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore
from tests.ui.conftest import make_trace


def _profile(pid: str, *, league: str = "NBA", status: ProfileStatus = ProfileStatus.PRODUCTION,
             version: int = 1, metrics: dict | None = None) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=pid, version=version, method="isotonic", league=league, status=status,
        training_window="2024-01-01/2024-12-31", sample_size=500, dataset_hash="h",
        metrics=metrics or {},
    )


def _client(tmp_path: Path, *, traces: int = 0, profiles=(), registry_path: str | None = None) -> TestClient:
    db = str(tmp_path / "b2.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    store = TraceStore(db_path=db)
    with store.autolog_suppressed():
        for i in range(traces):
            store.persist(make_trace(f"t{i}", kind="game"))
    store.close()
    reg_path = registry_path if registry_path is not None else str(tmp_path / "profiles.json")
    if profiles:
        reg = CalibrationRegistry(reg_path)
        for p in profiles:
            reg.register(p)
    return TestClient(build_console_app(db_path=db, sessions_dir=str(sessions), calibration_registry=reg_path))


def test_diagnostics_returns_db_and_registry_summary(tmp_path):
    client = _client(
        tmp_path, traces=2,
        profiles=[
            _profile("iso_nba_v2", league="NBA", status=ProfileStatus.PRODUCTION),
            _profile("iso_nba_v1", league="NBA", status=ProfileStatus.ARCHIVED),
            _profile("shr_epl_v1", league="EPL", status=ProfileStatus.CANDIDATE),
        ],
    )
    body = client.get("/api/diagnostics").json()
    assert body["status"] == "ok"
    assert body["trace_count"] == 2
    assert body["bet_count"] == 0
    cal = body["calibration"]
    assert cal["registry_available"] is True
    assert cal["total_profiles"] == 3
    assert cal["production"] == 1 and cal["candidate"] == 1 and cal["archived"] == 1
    assert cal["leagues_with_production"] == ["NBA"]
    assert body["field_sources"]["calibration"] == "calibration_registry"
    assert body["field_sources"]["bet_count"] == "bet_ledger"


def test_diagnostics_empty_registry_warns(tmp_path):
    body = _client(tmp_path, traces=1).get("/api/diagnostics").json()
    assert body["calibration"]["registry_available"] is True
    assert body["calibration"]["total_profiles"] == 0
    assert "registry_empty" in {w["code"] for w in body["warnings"]}


def test_diagnostics_never_500_on_bogus_registry_path(tmp_path):
    # A missing/garbage registry path must degrade to an empty summary, not a 500.
    client = _client(tmp_path, traces=1, registry_path=str(tmp_path / "nope" / "x.json"))
    resp = client.get("/api/diagnostics")
    assert resp.status_code == 200
    assert resp.json()["calibration"]["total_profiles"] == 0


def test_diagnostics_signal_scoring_empty_state(tmp_path):
    body = _client(tmp_path, traces=1).get("/api/diagnostics").json()
    ss = body["signal_scoring"]
    assert ss["last_scored_at"] is None
    assert ss["rows_in_latest_run"] == 0


def test_diagnostics_page_renders_with_enabled_nav(tmp_path):
    html = _client(tmp_path, traces=1).get("/diagnostics").text
    assert "<h1>Diagnostics</h1>" in html
    # Nav now links to the enabled pages (all placeholders implemented by B.3).
    assert 'href="/diagnostics"' in html
    assert 'href="/calibration"' in html
    assert 'href="/signals"' in html and 'href="/clv"' in html
