from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import report_calibration  # type: ignore  # noqa: E402

from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus  # noqa: E402
from omega.core.calibration.registry import CalibrationRegistry  # noqa: E402
from omega.ops.output_modes import OutputMode  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402


def _eligible_trace(
    trace_id: str,
    kind: str,
    timestamp: str = "2026-06-04T00:00:00+00:00",
    league: str = "NBA",
) -> dict:
    """A calibration-eligible trace of the given market `kind` (game | prop)."""
    preds = (
        {"home_win_prob": 0.55, "away_win_prob": 0.45}
        if kind == "game"
        else {"over_prob": 0.6, "under_prob": 0.4}
    )
    return {
        "trace_id": trace_id,
        "run_id": "run-pm",
        "timestamp": timestamp,
        "prompt": "x",
        "league": league,
        "kind": kind,
        "result": {"status": "success"},
        "predictions": preds,
        "trace_quality": {
            "calibration_eligible": 1,
            "context_source": "provided",
            "identity_status": "complete",
        },
    }


def _prod_profile(market: str, sample_size: int, ece: float, profile_id: str) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=profile_id,
        version=1,
        method="isotonic",
        league="NBA",
        market=market,
        status=ProfileStatus.PRODUCTION,
        training_window="2026-01-01/2026-06-01",
        sample_size=sample_size,
        dataset_hash="h",
        metrics={"brier_score": 0.2, "calibration_error": ece, "log_loss": 0.6},
        promoted_at="2026-06-03T00:00:00+00:00",
    )


def test_report_counts_default_deny_legacy_calibration_rows(tmp_path):
    store = TraceStore(db_path=tmp_path / "omega.db")
    store.persist(
        {
            "trace_id": "legacy-nba",
            "run_id": "run-legacy",
            "timestamp": "2026-05-20T00:00:00+00:00",
            "prompt": "legacy",
            "league": "NBA",
            "kind": "game",
            "result": {"status": "success"},
            "predictions": {"home_win_prob": 0.55, "away_win_prob": 0.45},
        }
    )
    store.attach_outcome("legacy-nba", home_score=100, away_score=90)

    counts = report_calibration._section_counts(store, "NBA", "2026-05-01T00:00:00+00:00")
    crps = report_calibration._section_distribution_crps(
        store,
        "NBA",
        "2026-05-01T00:00:00+00:00",
    )

    assert counts["with_predictions"] == 0
    assert counts["graded_calibration"] == 0
    assert crps is None
    store.close()


def test_report_calibration_writes_derived_header(tmp_path, monkeypatch):
    db = tmp_path / "omega.db"
    out = tmp_path / "latest.md"
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    TraceStore(db_path=db).close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_calibration.py",
            "--league",
            "NBA",
            "--db",
            str(db),
            "--out",
            str(out),
            "--sessions-inbox",
            str(sessions),
        ],
    )

    assert report_calibration.main() == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("---\ncanonical: false\n")
    assert f"source_db_path: {str(db)!r}" in text
    assert "trace_count_at_generation:" in text


def test_report_lists_prop_production_profiles(tmp_path):
    registry = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    registry.register(
        CalibrationProfile(
            profile_id="iso_nba_prop_v1",
            version=1,
            method="isotonic",
            league="NBA",
            market="prop",
            status=ProfileStatus.PRODUCTION,
            training_window="2026-01-01/2026-06-01",
            sample_size=120,
            dataset_hash="abc123",
            metrics={"brier_score": 0.21, "calibration_error": 0.03, "log_loss": 0.62},
            promoted_at="2026-06-03T00:00:00+00:00",
        )
    )

    rows = report_calibration._section_production_profiles(registry, "NBA")

    assert rows == [
        {
            "profile_id": "iso_nba_prop_v1",
            "league": "NBA",
            "market": "prop",
            "context_slice": None,
            "method": "isotonic",
            "sample_size": 120,
            "metrics": {"brier_score": 0.21, "calibration_error": 0.03, "log_loss": 0.62},
            "promoted_at": "2026-06-03T00:00:00+00:00",
        }
    ]


def test_section_counts_scopes_predictions_per_market(tmp_path):
    store = TraceStore(db_path=tmp_path / "omega.db")
    store.persist(_eligible_trace("g1", "game"))
    store.persist(_eligible_trace("p1", "prop"))
    store.persist(_eligible_trace("p2", "prop"))

    counts = report_calibration._section_counts(store, "NBA", "2026-01-01T00:00:00+00:00")

    assert counts["with_predictions"] == 3
    assert counts["with_predictions_game"] == 1
    assert counts["with_predictions_prop"] == 2
    store.close()


def test_resolve_output_modes_decouples_game_and_prop():
    # A trustworthy prop profile is ACTIONABLE even though the game market has no
    # production profile at all — the original bug was props being gated on game.
    prod_by_market = {"game": None, "prop": _prod_profile("prop", 150, 0.03, "iso_prop_good")}
    coverage = {"game": 5, "prop": 8}

    modes, reasons = report_calibration._resolve_output_modes(prod_by_market, coverage)

    assert modes["game"] is OutputMode.RESEARCH_CANDIDATE
    assert modes["prop"] is OutputMode.ACTIONABLE
    assert reasons["prop"] == []
    assert reasons["game"]


def test_resolve_output_modes_quality_floor_holds_weak_prop():
    # The real NBA prop profile (n=48, ECE 0.2876) must NOT unlock props.
    prod_by_market = {
        "game": None,
        "prop": _prod_profile("prop", 48, 0.287589, "iso_nba_prop_v1_7c8018680da72efe"),
    }
    coverage = {"game": 0, "prop": 12}

    modes, reasons = report_calibration._resolve_output_modes(prod_by_market, coverage)

    assert modes["prop"] is OutputMode.RESEARCH_CANDIDATE
    assert any("sample_size" in r for r in reasons["prop"])
    assert any("ECE" in r for r in reasons["prop"])


def test_aggregate_scalar_mode_is_conservative():
    actionable, research = OutputMode.ACTIONABLE, OutputMode.RESEARCH_CANDIDATE
    assert report_calibration._aggregate_scalar_mode({"game": actionable, "prop": actionable}) is actionable
    assert report_calibration._aggregate_scalar_mode({"game": research, "prop": actionable}) is research
    assert report_calibration._aggregate_scalar_mode({}) is research


def test_report_frontmatter_emits_per_market_map(tmp_path, monkeypatch):
    db = tmp_path / "omega.db"
    out = tmp_path / "latest.md"
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    profiles = tmp_path / "profiles.json"

    store = TraceStore(db_path=db)
    store.persist(_eligible_trace("p1", "prop"))
    store.close()

    registry = CalibrationRegistry(path=str(profiles))
    registry.register(_prod_profile("prop", 150, 0.03, "iso_prop_good"))

    monkeypatch.setattr(
        report_calibration,
        "CalibrationRegistry",
        lambda: CalibrationRegistry(path=str(profiles)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_calibration.py",
            "--league",
            "NBA",
            "--db",
            str(db),
            "--out",
            str(out),
            "--sessions-inbox",
            str(sessions),
            "--window-days",
            "3650",
        ],
    )

    assert report_calibration.main() == 0
    text = out.read_text(encoding="utf-8")

    # Per-market frontmatter map: prop actionable, game research-only.
    assert "output_modes:" in text
    assert "prop: 'actionable'" in text
    assert "game: 'research_candidate'" in text
    # Conservative scalar: any research market -> research.
    assert "output_mode: 'research_candidate'" in text
    # Per-market prose directive.
    assert "Player props" in text
    assert "ACTIONABLE" in text


def test_report_main_is_overall_even_when_league_arg_is_passed(tmp_path, monkeypatch):
    db = tmp_path / "omega.db"
    out = tmp_path / "latest.md"
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    profiles = tmp_path / "profiles.json"

    store = TraceStore(db_path=db)
    store.persist(_eligible_trace("nba-game", "game", league="NBA"))
    store.persist(_eligible_trace("mlb-game", "game", league="MLB"))
    store.close()

    registry = CalibrationRegistry(path=str(profiles))
    registry.register(_prod_profile("game", 150, 0.03, "iso_nba_game"))
    registry.register(
        CalibrationProfile(
            profile_id="iso_mlb_game_candidate",
            version=1,
            method="isotonic",
            league="MLB",
            market="game",
            status=ProfileStatus.CANDIDATE,
            training_window="2026-01-01/2026-06-01",
            sample_size=80,
            dataset_hash="h",
            metrics={"brier_score": 0.22, "calibration_error": 0.06, "log_loss": 0.66},
            created_at="2026-06-05T00:00:00+00:00",
        )
    )

    monkeypatch.setattr(
        report_calibration,
        "CalibrationRegistry",
        lambda: CalibrationRegistry(path=str(profiles)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_calibration.py",
            "--league",
            "NBA",
            "--db",
            str(db),
            "--out",
            str(out),
            "--sessions-inbox",
            str(sessions),
            "--window-days",
            "3650",
        ],
    )

    assert report_calibration.main() == 0
    text = out.read_text(encoding="utf-8")

    assert "# Omega Health Report - Overall" in text
    assert "| Traces (all) | 2 |" in text
    assert "| NBA | game | base | `iso_nba_game` |" in text
    assert "| MLB | game | `iso_mlb_game_candidate` |" in text
