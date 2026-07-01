"""Read-only Backtest Scorecard console view (Phase 8 Wave 3c).

Reads lab-run artifacts (LAB_RUN.json + backtest_report.json + PROMOTION_EVIDENCE.json)
off the filesystem and renders them; never mutates and never recomputes a metric.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from omega.ops.console_server import build_console_app


def _write_run(root: Path, run_id: str) -> None:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "LAB_RUN.json").write_text(
        json.dumps(
            {
                "lab_run_id": run_id,
                "league": "NFL",
                "plane": "cover",
                "promotion_status": "shadow_only",
                "created_at": "2026-06-30T00:00:00+00:00",
                "holdout_sealed": True,
            }
        ),
        encoding="utf-8",
    )
    (d / "PROMOTION_EVIDENCE.json").write_text(
        json.dumps({"clv_coherent": False}), encoding="utf-8"
    )
    (d / "backtest_report.json").write_text(
        json.dumps(
            {
                "scorecard": [
                    {
                        "market": "cover",
                        "n_calibrated": 120,
                        "raw_ece": 0.08,
                        "calibrated_ece": 0.04,
                        "n_bets": 40,
                        "roi": 0.03,
                        "avg_clv": 0.5,
                        "mean_signed_divergence": 0.02,
                        "clv_when_divergent": -0.1,
                        "divergent_beat_close_rate": 0.3,
                        "clv_coherent": False,
                        # an extra contract field the compact view ignores:
                        "raw_brier": 0.24,
                    }
                ],
                "aggregate_marginal_value": [
                    {
                        "signal_type": "recent_form_residual",
                        "brier_delta": 0.01,
                        "log_loss_delta": 0.02,
                        "n": 30,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_backtests_list_and_detail(seeded, tmp_path):
    root = tmp_path / "lab_runs"
    _write_run(root, "lab_nfl_cover")
    client = TestClient(build_console_app(db_path=seeded["db_path"], backtests_dir=str(root)))

    # HTML list page renders the run.
    r = client.get("/backtests")
    assert r.status_code == 200
    assert "lab_nfl_cover" in r.text
    assert "Backtest Scorecard" in r.text

    # HTML detail page renders the fused scorecard + marginal value.
    r = client.get("/backtests/lab_nfl_cover")
    assert r.status_code == 200
    assert "recent_form_residual" in r.text
    assert "cover" in r.text

    # JSON twins.
    runs = client.get("/api/backtests").json()["runs"]
    assert runs[0]["lab_run_id"] == "lab_nfl_cover"
    assert runs[0]["clv_coherent"] is False
    detail = client.get("/api/backtests/lab_nfl_cover").json()
    assert detail["scorecard"][0]["market"] == "cover"
    assert detail["clv_coherent"] is False
    assert detail["marginal_value"][0]["signal_type"] == "recent_form_residual"


def test_backtests_missing_and_traversal_guard(seeded, tmp_path):
    root = tmp_path / "lab_runs"
    root.mkdir()
    client = TestClient(build_console_app(db_path=seeded["db_path"], backtests_dir=str(root)))

    assert client.get("/api/backtests/missing").status_code == 404
    assert client.get("/backtests/missing").status_code == 404
    # Path-traversal id is rejected (treated as not found), never resolved on disk.
    assert client.get("/api/backtests/..%2f..%2fetc").status_code in (400, 404)


def test_backtests_empty_dir_is_graceful(seeded, tmp_path):
    client = TestClient(
        build_console_app(db_path=seeded["db_path"], backtests_dir=str(tmp_path / "absent"))
    )
    assert client.get("/backtests").status_code == 200
    assert client.get("/api/backtests").json()["runs"] == []
