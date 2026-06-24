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
    assert (
        report_calibration._aggregate_scalar_mode({"game": actionable, "prop": actionable})
        is actionable
    )
    assert (
        report_calibration._aggregate_scalar_mode({"game": research, "prop": actionable})
        is research
    )
    assert report_calibration._aggregate_scalar_mode({}) is research


def test_signal_guidance_buckets_bootstrap_warnings():
    rows = [
        {
            "signal_type": "usage_role_change",
            "source": "injury_report",
            "obs_window": "matchup",
            "league": "NBA",
            "sample_size": 40,
            "direction_accuracy": 0.70,
            "calibration_gap": 0.02,
            "brier": 0.18,
        },
        {
            "signal_type": "series_avg",
            "source": "nba.com",
            "obs_window": "series",
            "league": "NBA",
            "sample_size": 35,
            "direction_accuracy": 0.40,
            "calibration_gap": 0.30,
            "brier": 0.49,
        },
        {
            "signal_type": "recent_form",
            "source": "boxscore_derived",
            "obs_window": "last_5",
            "league": "NBA",
            "sample_size": 8,
            "direction_accuracy": 0.75,
            "calibration_gap": -0.05,
            "brier": 0.20,
        },
    ]

    guidance = report_calibration._signal_guidance(rows)

    assert guidance["trusted"][0]["signal_type"] == "usage_role_change"
    assert guidance["trusted"][0]["direction_accuracy"] == 0.65
    assert guidance["trusted"][0]["brier"] == 0.15
    assert guidance["warnings"][0]["signal_type"] == "unknown"
    assert guidance["warnings"][0]["calibration_gap"] == 0.15
    assert guidance["insufficient"][0]["signal_type"] == "unknown"


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


class TestClvSignalVerdict:
    """Issue #28: the §6B verdict is CLV-primary with a direction fallback."""

    def test_clv_aligned_when_coverage_sufficient(self):
        row = {"clv_aligned": 0.62, "clv_sample": 100, "sample_size": 100,
               "direction_accuracy": 0.42, "calibration_gap": 0.0}
        assert report_calibration._signal_verdict(row) == "clv_aligned"

    def test_clv_misaligned_restates_the_line(self):
        # recent_form: strong direction but CLV says it's already in the line.
        row = {"clv_aligned": 0.46, "clv_sample": 100, "sample_size": 100,
               "direction_accuracy": 0.70, "calibration_gap": 0.0}
        assert report_calibration._signal_verdict(row) == "clv_misaligned"

    def test_falls_back_to_direction_when_clv_thin(self):
        row = {"clv_aligned": None, "clv_sample": 0, "sample_size": 50,
               "direction_accuracy": 0.58, "calibration_gap": 0.0}
        assert report_calibration._signal_verdict(row) == "predictive"

    def test_insufficient_when_no_clv_and_thin_direction(self):
        row = {"clv_aligned": None, "clv_sample": 0, "sample_size": 5,
               "direction_accuracy": 0.6, "calibration_gap": 0.0}
        assert report_calibration._signal_verdict(row) == "insufficient_n"
