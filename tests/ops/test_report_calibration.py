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

from omega.core.calibration.profiles import (  # noqa: E402
    CalibrationBackendBinding,
    CalibrationProfile,
    ProfileStatus,
)
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
            backend_binding=CalibrationBackendBinding(
                backend_name="prop_neg_binom",
                backend_component_version="prop_nb_v1",
                param_profile_id="prop_neg_binom__NBA__PTS__v1",
            ),
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
            "binding_status": "bound",
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
    # The real NBA prop profile (n=48, ECE 0.2876) must NOT unlock full ACTIONABLE
    # output. It is a real profile, so it lands in research+ (numbers shown under a
    # capped stake, with the failing floor in the reasons), never actionable.
    prod_by_market = {
        "game": None,
        "prop": _prod_profile("prop", 48, 0.287589, "iso_nba_prop_v1_7c8018680da72efe"),
    }
    coverage = {"game": 0, "prop": 12}

    modes, reasons = report_calibration._resolve_output_modes(prod_by_market, coverage)

    assert modes["prop"] is OutputMode.RESEARCH_PLUS
    assert modes["prop"] is not OutputMode.ACTIONABLE
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


def test_report_main_scopes_to_league_arg(tmp_path, monkeypatch):
    """Phase 0: --league is functional again — every section is league-scoped.

    (Reverses the earlier deprecation where --league was accepted but ignored;
    latest.md remains overall only when --league is omitted.)
    """
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

    assert "# Omega Health Report - NBA" in text
    # Only the NBA trace is counted; the MLB trace is out of scope.
    assert "| Traces (all) | 1 |" in text
    assert "| NBA | game | base | `iso_nba_game` | isotonic | legacy |" in text
    # The MLB candidate must not leak into an NBA-scoped report.
    assert "iso_mlb_game_candidate" not in text


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


class TestUnifiedScorecard:
    """Issue #28 WS5: §6B reliability column + recommendations/market-aware tail."""

    def test_reliability_map_reads_production_policy(self, monkeypatch):
        from omega.core.calibration import adjustment_policy as ap

        class _Pol:
            coefficients = {
                "recent_form": {"reliability_weight": 0.0, "cap": 0.1},
                "usage_spike": {"cap": 0.2},  # no reliability_weight -> excluded
            }

        class _Reg:
            def __init__(self, *a, **k):
                pass

            def get_production_policy(self):
                return _Pol()

        monkeypatch.setattr(ap, "AdjustmentPolicyRegistry", _Reg)
        assert report_calibration._production_reliability_map() == {"recent_form": 0.0}

    def test_scorecard_lines_render_recommendations_and_market(self, monkeypatch):
        from omega.core.calibration import adjustment_policy as ap
        from omega.core.calibration import registry as reg_mod

        class _Cand:
            version = 3
            lifecycle_recommendations = {"recent_form": "deprecated", "stale_line": "rejected"}

        class _APReg:
            def __init__(self, *a, **k):
                pass

            def list_policies(self, status=None):
                return [_Cand()]

        class _Prof:
            league = "NBA"
            market = "game"
            method = "market_aware"
            profile_id = "market_nba_v1"
            params = {"market_weight": 0.8}

        class _CReg:
            def __init__(self, *a, **k):
                pass

            def list_profiles(self, status=None):
                return [_Prof()]

        monkeypatch.setattr(ap, "AdjustmentPolicyRegistry", _APReg)
        monkeypatch.setattr(reg_mod, "CalibrationRegistry", _CReg)

        text = "\n".join(report_calibration._evidence_scorecard_lines())
        assert "`recent_form` -> **deprecated**" in text
        assert "`stale_line` -> **rejected**" in text
        assert "Market-aware deference" in text
        assert "market_weight=0.80" in text

    def test_scorecard_lines_empty_without_data(self, monkeypatch):
        from omega.core.calibration import adjustment_policy as ap
        from omega.core.calibration import registry as reg_mod

        class _APReg:
            def __init__(self, *a, **k):
                pass

            def list_policies(self, status=None):
                return []

        class _CReg:
            def __init__(self, *a, **k):
                pass

            def list_profiles(self, status=None):
                return []

        monkeypatch.setattr(ap, "AdjustmentPolicyRegistry", _APReg)
        monkeypatch.setattr(reg_mod, "CalibrationRegistry", _CReg)
        assert report_calibration._evidence_scorecard_lines() == []
