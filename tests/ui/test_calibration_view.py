"""Milestone B.2 — Calibration Status page: read-only registry profile listing."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus
from omega.core.calibration.registry import CalibrationRegistry
from omega.ops.console_server import build_console_app
from omega.trace.store import TraceStore


def _profile(
    pid: str,
    *,
    league: str = "NBA",
    status: ProfileStatus = ProfileStatus.PRODUCTION,
    version: int = 1,
    market: str = "game",
    context_slice: str | None = None,
    metrics: dict | None = None,
    promoted_at: str | None = None,
) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=pid,
        version=version,
        method="isotonic",
        league=league,
        status=status,
        market=market,
        context_slice=context_slice,
        training_window="2024-01-01/2024-12-31",
        sample_size=750,
        dataset_hash="h",
        metrics=metrics or {},
        promoted_at=promoted_at,
    )


def _client(
    tmp_path: Path, *, profiles=(), registry_path: str | None = None
) -> tuple[TestClient, str]:
    db = str(tmp_path / "b2.db")
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    TraceStore(db_path=db).close()
    reg_path = registry_path if registry_path is not None else str(tmp_path / "profiles.json")
    if profiles:
        reg = CalibrationRegistry(reg_path)
        for p in profiles:
            reg.register(p)
    app = build_console_app(db_path=db, sessions_dir=str(sessions), calibration_registry=reg_path)
    return TestClient(app), reg_path


def test_calibration_lists_profiles_and_marks_active(tmp_path):
    client, _ = _client(
        tmp_path,
        profiles=[
            _profile(
                "iso_nba_v3",
                status=ProfileStatus.PRODUCTION,
                metrics={
                    "brier_score": 0.21,
                    "calibration_error": 0.03,
                    "log_loss": 0.55,
                    "n_eval": 400,
                },
                promoted_at="2026-05-01T00:00:00Z",
            ),
            _profile(
                "iso_nba_v2",
                status=ProfileStatus.ARCHIVED,
                version=2,
                metrics={"brier_score": 0.24},
            ),
        ],
    )
    rows = {r["profile_id"]: r for r in client.get("/api/calibration").json()["rows"]}
    assert rows["iso_nba_v3"]["is_active"] is True
    assert rows["iso_nba_v2"]["is_active"] is False
    # Metrics flattened from the registry's nested metrics dict.
    assert rows["iso_nba_v3"]["brier"] == 0.21
    assert rows["iso_nba_v3"]["calibration_error"] == 0.03
    assert rows["iso_nba_v3"]["log_loss"] == 0.55
    assert rows["iso_nba_v3"]["n_eval"] == 400
    assert rows["iso_nba_v3"]["field_sources"]["profile"] == "calibration_registry"


def test_calibration_empty_registry_warns(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/calibration").json()
    assert body["rows"] == []
    assert body["registry_available"] is True
    assert "registry_empty" in {w["code"] for w in body["warnings"]}


def test_calibration_filters_by_league_and_status(tmp_path):
    client, _ = _client(
        tmp_path,
        profiles=[
            _profile("iso_nba_v3", league="NBA", status=ProfileStatus.PRODUCTION),
            _profile("shr_epl_v1", league="EPL", status=ProfileStatus.CANDIDATE),
        ],
    )
    nba = client.get("/api/calibration?league=NBA").json()["rows"]
    assert {r["profile_id"] for r in nba} == {"iso_nba_v3"}
    prod = client.get("/api/calibration?status=production").json()["rows"]
    assert {r["profile_id"] for r in prod} == {"iso_nba_v3"}


def test_calibration_no_production_profile_warning(tmp_path):
    client, _ = _client(
        tmp_path,
        profiles=[
            _profile("shr_epl_v1", league="EPL", status=ProfileStatus.CANDIDATE),
        ],
    )
    body = client.get("/api/calibration").json()
    warn = {w["code"]: w for w in body["warnings"]}
    assert "no_production_profile" in warn
    assert "EPL" in warn["no_production_profile"]["message"]


def test_calibration_page_renders_and_escapes(tmp_path):
    client, _ = _client(
        tmp_path,
        profiles=[
            _profile(
                "<script>alert(1)</script>",
                status=ProfileStatus.PRODUCTION,
                metrics={"brier_score": 0.2},
            ),
        ],
    )
    html = client.get("/calibration").text
    assert "<h1>Calibration Cockpit</h1>" in html
    assert "badge-prod" in html  # active badge rendered
    assert "pstatus-production" in html
    # Injected HTML in a profile_id is escaped.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_calibration_read_only_does_not_mutate_registry(tmp_path):
    client, reg_path = _client(tmp_path, profiles=[_profile("iso_nba_v3")])
    before = Path(reg_path).read_text(encoding="utf-8")
    client.get("/api/calibration")
    client.get("/calibration")
    after = Path(reg_path).read_text(encoding="utf-8")
    assert before == after  # listing never writes the registry
