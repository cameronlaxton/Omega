"""Unit tests for the read-only B.0 normalization layer (omega.ui.normalizers).

These exercise the trustworthy interpretation contract: selection-aware
probability mapping, implied/computed-edge derivation from confirmed American
odds only, confidence-band doctrine, scalar-only display values, evidence
coverage counting (not scoring), session-health aggregation, and the warning
vocabulary — all read-only, all sourced from DB-backed payloads.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Any

import pytest

import omega.ui.normalizers as norm
from omega.ui.normalizers import (
    VALID_SEVERITIES,
    EvidenceCoverage,
    ExtractedField,
    NormalizedRecommendation,
    SessionTraceFacts,
    build_evidence_coverage,
    build_session_health_view,
    build_trace_recommendation_view,
    computed_edge_value,
    confidence_band,
    implied_probability_from_american,
)
from omega.ui.schemas import Source

# ---------------------------------------------------------------------------
# Builders for realistic trace payloads (shapes verified against
# omega.core.contracts.schemas + omega.trace.persistable)
# ---------------------------------------------------------------------------


def prop_trace(
    *,
    recommendation: str = "over",
    over_prob: float | None = 0.3675,
    under_prob: float | None = 0.6325,
    bet_side_odds: float | None = 165,
    confidence_tier: str | None = "B",
    kelly_fraction: float | None = None,
    recommended_units: float | None = None,
    over_calibrated: float | None = None,
    under_calibrated: float | None = None,
    line: float | None = 25.5,
    **rec_extra: Any,
) -> dict[str, Any]:
    """A persisted prop trace: predictions = {over_prob, under_prob};
    recommendations = single flat dict."""
    predictions: dict[str, Any] = {}
    if over_prob is not None:
        predictions["over_prob"] = over_prob
    if under_prob is not None:
        predictions["under_prob"] = under_prob
    rec: dict[str, Any] = {
        "recommendation": recommendation,
        "confidence_tier": confidence_tier,
    }
    if kelly_fraction is not None:
        rec["kelly_fraction"] = kelly_fraction
    if recommended_units is not None:
        rec["recommended_units"] = recommended_units
    if bet_side_odds is not None:
        rec["bet_side_odds"] = bet_side_odds
    rec.update(rec_extra)
    result: dict[str, Any] = {"recommendation": recommendation, "line": line}
    if over_calibrated is not None:
        result["over_calibration_audit"] = {"market": "over", "calibrated_prob": over_calibrated}
    if under_calibrated is not None:
        result["under_calibration_audit"] = {"market": "under", "calibrated_prob": under_calibrated}
    return {
        "trace_id": "prop-1",
        "kind": "prop",
        "predictions": predictions or None,
        "recommendations": rec,
        "result": result,
        "input_snapshot": {"line": line},
    }


def game_edge(
    side: str,
    *,
    true_prob: float,
    calibrated_prob: float | None = None,
    market_implied: float | None = None,
    market_odds: float | None = -150,
    edge_pct: float | None = 4.2,
    confidence_tier: str | None = "A",
    recommended_units: float | None = 1.5,
    market: str = "moneyline",
    line: float | None = None,
) -> dict[str, Any]:
    edge: dict[str, Any] = {
        "side": side,
        "team": f"{side}_team",
        "market": market,
        "true_prob": true_prob,
        "market_odds": market_odds,
        "confidence_tier": confidence_tier,
        "recommended_units": recommended_units,
    }
    if calibrated_prob is not None:
        edge["calibrated_prob"] = calibrated_prob
        edge["calibration_audit"] = {"market": side, "calibrated_prob": calibrated_prob}
    if market_implied is not None:
        edge["market_implied"] = market_implied
    if edge_pct is not None:
        edge["edge_pct"] = edge_pct
    if line is not None:
        edge["line"] = line
    return edge


def game_trace(
    edges: list[dict[str, Any]], *, simulation: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A persisted game trace: predictions = simulation block (0–100 scale);
    recommendations = list of edge dicts."""
    return {
        "trace_id": "game-1",
        "kind": "game",
        "predictions": simulation,
        "recommendations": edges,
        "result": {"edges": edges, "simulation": simulation},
        "odds_snapshot": {"moneyline_home": -150, "moneyline_away": 130},
    }


def _primary(view) -> NormalizedRecommendation:
    return view.recommendations[0]


def _iter_fields(rec: NormalizedRecommendation):
    for f in dataclasses.fields(rec):
        val = getattr(rec, f.name)
        if isinstance(val, ExtractedField):
            yield f.name, val


# ---------------------------------------------------------------------------
# Selection-aware probability
# ---------------------------------------------------------------------------


def test_prop_over_maps_to_over_prob():
    view = build_trace_recommendation_view(prop_trace(recommendation="over"))
    rec = _primary(view)
    assert rec.raw_probability.value == pytest.approx(0.3675)
    assert rec.raw_probability.source_path == "predictions.over_prob"


def test_prop_under_maps_to_under_prob():
    view = build_trace_recommendation_view(prop_trace(recommendation="under"))
    rec = _primary(view)
    # Must pick the under side, never the first available (over) probability.
    assert rec.raw_probability.value == pytest.approx(0.6325)
    assert rec.raw_probability.source_path == "predictions.under_prob"


def test_game_home_maps_to_home_win_prob():
    edges = [
        game_edge("home", true_prob=0.58),
        game_edge("away", true_prob=0.42),
    ]
    # Simulation block is on a 0–100 scale; the edge-native 0–1 true_prob wins.
    view = build_trace_recommendation_view(
        game_trace(edges, simulation={"home_win_prob": 58.0, "away_win_prob": 42.0})
    )
    home = view.recommendations[0]
    assert home.selection.value == "home"
    assert home.raw_probability.value == pytest.approx(0.58)
    assert home.raw_probability.value != pytest.approx(0.42)


def test_game_home_falls_back_to_predictions_when_no_edge_prob():
    # Edge without true_prob → selection-aware fallback to predictions.home_win_prob.
    edge = {"side": "home", "team": "home_team", "market": "moneyline", "market_odds": -150}
    view = build_trace_recommendation_view(
        game_trace([edge], simulation={"home_win_prob": 0.61, "away_win_prob": 0.39})
    )
    home = _primary(view)
    assert home.raw_probability.value == pytest.approx(0.61)
    assert home.raw_probability.source_path == "predictions.home_win_prob"


# ---------------------------------------------------------------------------
# Implied probability
# ---------------------------------------------------------------------------


def test_positive_american_odds_implied_probability():
    assert implied_probability_from_american(165) == pytest.approx(0.37736, abs=1e-4)
    view = build_trace_recommendation_view(prop_trace(bet_side_odds=165))
    rec = _primary(view)
    assert rec.implied_probability.value == pytest.approx(0.37736, abs=1e-4)
    assert rec.implied_probability.computed is True
    assert rec.implied_probability.source_path == "computed:implied_from_american_odds"


def test_negative_american_odds_implied_probability():
    assert implied_probability_from_american(-150) == pytest.approx(0.60)
    view = build_trace_recommendation_view(prop_trace(bet_side_odds=-150))
    assert _primary(view).implied_probability.value == pytest.approx(0.60)


def test_zero_odds_gives_null_implied():
    assert implied_probability_from_american(0) is None
    view = build_trace_recommendation_view(prop_trace(bet_side_odds=0))
    assert _primary(view).implied_probability.value is None


def test_missing_odds_gives_null_implied():
    view = build_trace_recommendation_view(prop_trace(bet_side_odds=None))
    rec = _primary(view)
    assert rec.odds.value is None
    assert rec.odds.source_path is None
    assert rec.implied_probability.value is None
    assert "missing_odds" in {w.code for w in rec.warnings}


def test_ambiguous_odds_format_warning():
    # No confirmed-American odds on the rec; only the heuristic snapshot carries a
    # value → surfaced but flagged, implied not computed.
    edge = {"side": "home", "team": "home_team", "market": "moneyline", "true_prob": 0.55}
    view = build_trace_recommendation_view(game_trace([edge]))
    rec = _primary(view)
    codes = {w.code for w in rec.warnings}
    assert "ambiguous_odds_format" in codes
    assert rec.odds.value == -150  # surfaced from odds_snapshot.moneyline_home
    assert rec.odds.source_path == "odds_snapshot.moneyline_home"
    assert rec.implied_probability.value is None


# ---------------------------------------------------------------------------
# Edge handling
# ---------------------------------------------------------------------------


def test_engine_edge_extracted_from_edge_pct():
    view = build_trace_recommendation_view(
        game_trace([game_edge("home", true_prob=0.58, edge_pct=4.2)])
    )
    rec = _primary(view)
    assert rec.engine_edge.value == pytest.approx(4.2)
    assert rec.engine_edge.computed is False
    assert rec.engine_edge.source_path == "recommendations[0].edge_pct"


def test_computed_edge_only_when_calibrated_present():
    # calibrated 0.62 + confirmed American odds -150 (implied 0.60) → edge 0.02.
    view = build_trace_recommendation_view(
        game_trace([game_edge("home", true_prob=0.58, calibrated_prob=0.62, market_odds=-150)])
    )
    rec = _primary(view)
    assert rec.calibrated_probability.value == pytest.approx(0.62)
    assert rec.computed_edge.value == pytest.approx(0.62 - 0.60)
    assert rec.computed_edge.computed is True
    assert rec.computed_edge.source_path == "computed:calibrated_minus_implied"


def test_computed_edge_null_when_calibrated_missing():
    view = build_trace_recommendation_view(
        game_trace([game_edge("home", true_prob=0.58, calibrated_prob=None, market_odds=-150)])
    )
    rec = _primary(view)
    assert rec.calibrated_probability.value is None
    assert rec.computed_edge.value is None
    assert rec.computed_edge.source_path is None


def test_computed_edge_helper_requires_both_operands():
    assert computed_edge_value(0.62, 0.60) == pytest.approx(0.02)
    assert computed_edge_value(None, 0.60) is None
    assert computed_edge_value(0.62, None) is None


def test_raw_below_implied_warning():
    # over_prob 0.3675 < implied(+165)=0.3774, and no calibrated probability.
    view = build_trace_recommendation_view(
        prop_trace(recommendation="over", over_prob=0.3675, bet_side_odds=165)
    )
    rec = _primary(view)
    assert rec.calibrated_probability.value is None
    assert "raw_below_implied" in {w.code for w in rec.warnings}


def test_kelly_no_edge_warning():
    # Positive Kelly, but no engine edge and no computable edge (odds missing).
    view = build_trace_recommendation_view(prop_trace(kelly_fraction=0.0384, bet_side_odds=None))
    rec = _primary(view)
    assert rec.engine_edge.value is None
    assert rec.computed_edge.value is None
    assert "kelly_no_edge" in {w.code for w in rec.warnings}


def test_units_no_kelly_warning():
    view = build_trace_recommendation_view(prop_trace(recommended_units=3.84, kelly_fraction=None))
    rec = _primary(view)
    assert rec.recommended_units.value == pytest.approx(3.84)
    assert rec.kelly_fraction.value is None
    assert "units_no_kelly" in {w.code for w in rec.warnings}


# ---------------------------------------------------------------------------
# Multi-recommendation shapes
# ---------------------------------------------------------------------------


def test_list_recommendations_normalize_multiple():
    edges = [
        game_edge("home", true_prob=0.58),
        game_edge("away", true_prob=0.42),
        game_edge("draw", true_prob=0.10, market="draw"),
    ]
    view = build_trace_recommendation_view(game_trace(edges))
    assert len(view.recommendations) == 3
    assert view.recommendations[0].is_primary is True
    assert view.recommendations[0].rank == 0
    assert view.recommendations[1].is_primary is False
    assert view.recommendations[1].rank == 1
    assert view.recommendations[2].rank == 2


def test_best_bet_normalizes_as_primary():
    # result.best_bet alone (no recommendations / edges).
    trace = {
        "trace_id": "g2",
        "kind": "game",
        "result": {
            "best_bet": {
                "selection": "Lakers -3.5",
                "odds": -110,
                "edge_pct": 3.0,
                "confidence_tier": "B",
                "recommended_units": 1.0,
                "kelly_fraction": 0.02,
            }
        },
    }
    view = build_trace_recommendation_view(trace)
    assert len(view.recommendations) == 1
    rec = _primary(view)
    assert rec.is_primary is True
    assert rec.rank is None
    assert rec.selection.value == "Lakers -3.5"
    assert rec.odds.value == -110  # BetSlip.odds is confirmed American


def test_dict_recommendation_single_item():
    view = build_trace_recommendation_view(prop_trace())
    assert len(view.recommendations) == 1
    rec = _primary(view)
    assert rec.is_primary is True
    assert rec.rank is None


def test_trace_without_recommendations_returns_empty_list():
    view = build_trace_recommendation_view({"trace_id": "x", "kind": "game", "result": {}})
    assert view.recommendations == []
    assert view.raw_payload_available is True
    assert view.kind == "game"


# ---------------------------------------------------------------------------
# Field / provenance invariants
# ---------------------------------------------------------------------------


def test_missing_fields_return_none_source_path():
    view = build_trace_recommendation_view(
        prop_trace(line=None, bet_side_odds=None, confidence_tier=None)
    )
    rec = _primary(view)
    assert rec.line.value is None and rec.line.source_path is None
    assert rec.odds.value is None and rec.odds.source_path is None
    assert rec.raw_confidence_tier.value is None and rec.raw_confidence_tier.source_path is None


def test_missing_values_have_none_source_path():
    # Sparse rec: scan every ExtractedField — value None ⇒ source_path None.
    view = build_trace_recommendation_view(
        {"trace_id": "z", "kind": "prop", "recommendations": {"recommendation": "over"}}
    )
    rec = _primary(view)
    for name, fld in _iter_fields(rec):
        if fld.value is None:
            assert fld.source_path is None, f"{name} has source_path for a None value"


def test_confidence_band_mapping():
    assert confidence_band("A") == "high confidence"
    assert confidence_band("B") == "medium confidence"
    assert confidence_band("C") == "low confidence"
    assert confidence_band("Pass") == "tracked lean"
    assert confidence_band(None) == "unrated"
    # Wired through a recommendation, marked computed.
    view = build_trace_recommendation_view(prop_trace(confidence_tier="A"))
    rec = _primary(view)
    assert rec.display_confidence_band.value == "high confidence"
    assert rec.display_confidence_band.computed is True
    # Raw tier preserved for audit; never "A-Tier" phrasing in display.
    assert rec.raw_confidence_tier.value == "A"
    assert "tier" not in rec.display_confidence_band.value.lower()


def test_unrecognized_confidence_tier():
    assert confidence_band("Z") == "source confidence: Z"
    view = build_trace_recommendation_view(prop_trace(confidence_tier="Z"))
    assert _primary(view).display_confidence_band.value == "source confidence: Z"


def test_scalar_value_only():
    # Direct guard + a dict/list candidate collapses to a missing field.
    assert norm._scalarize({"a": 1}) is None
    assert norm._scalarize([1, 2]) is None
    assert norm._scalarize("x") == "x"
    # A dict-valued field candidate collapses to a missing field (no fallback).
    view = build_trace_recommendation_view(prop_trace(kelly_fraction={"nested": 1}))
    rec = _primary(view)
    assert rec.kelly_fraction.value is None  # dict candidate rejected
    assert rec.kelly_fraction.source_path is None


def test_recommendation_value_never_dict_or_list():
    view = build_trace_recommendation_view(prop_trace())
    rec = _primary(view)
    for name, fld in _iter_fields(rec):
        assert not isinstance(fld.value, (dict, list)), f"{name} leaked a container"


def test_computed_fields_are_marked_computed():
    view = build_trace_recommendation_view(
        game_trace([game_edge("home", true_prob=0.58, calibrated_prob=0.62, market_odds=-150)])
    )
    rec = _primary(view)
    assert rec.implied_probability.computed is True
    assert rec.computed_edge.computed is True
    assert rec.display_confidence_band.computed is True


def test_computed_source_paths_start_with_computed():
    view = build_trace_recommendation_view(
        game_trace([game_edge("home", true_prob=0.58, calibrated_prob=0.62, market_odds=-150)])
    )
    rec = _primary(view)
    for fld in (rec.implied_probability, rec.computed_edge, rec.display_confidence_band):
        assert fld.source_path is not None
        assert fld.source_path.startswith("computed:")


def test_no_sidecar_numeric_sources():
    # No extracted field in a normalized trace is sourced from the sidecar.
    view = build_trace_recommendation_view(
        game_trace([game_edge("home", true_prob=0.58, calibrated_prob=0.62)])
    )
    for rec in view.recommendations:
        for _name, fld in _iter_fields(rec):
            assert fld.source != Source.SIDECAR_PROCESS


# ---------------------------------------------------------------------------
# Evidence coverage
# ---------------------------------------------------------------------------


def test_evidence_zero_creates_warning():
    cov = build_evidence_coverage([])
    assert isinstance(cov, EvidenceCoverage)
    assert cov.total_signals == 0
    assert "zero_evidence" in {w.code for w in cov.warnings}


def test_evidence_applied_shadow_counts():
    rows = [
        {"applied": 1, "signal_type": "injury", "confidence": 0.8},
        {"applied": 0, "signal_type": "weather", "confidence": 0.4},
        {"applied": 0, "applied_factor": 0.5, "signal_type": "form"},
    ]
    cov = build_evidence_coverage(rows)
    assert cov.total_signals == 3
    assert cov.applied_signals == 2  # applied=1 and the non-zero applied_factor
    assert cov.shadow_signals == 1
    assert cov.signal_types_present == ["form", "injury", "weather"]
    assert "no_applied_evidence" not in {w.code for w in cov.warnings}


def test_evidence_avg_confidence():
    rows = [
        {"signal_type": "a", "confidence": 0.8},
        {"signal_type": "b", "confidence": 0.4},
        {"signal_type": "c", "confidence": None},
    ]
    cov = build_evidence_coverage(rows)
    assert cov.signals_with_confidence == 2
    assert cov.avg_confidence == pytest.approx(0.6)
    # No confidence values at all → None, not 0.
    assert build_evidence_coverage([{"signal_type": "x"}]).avg_confidence is None


def test_evidence_low_avg_confidence_info_warning():
    cov = build_evidence_coverage(
        [
            {"applied": 1, "signal_type": "a", "confidence": 0.1},
            {"applied": 1, "signal_type": "b", "confidence": 0.2},
        ]
    )
    assert cov.avg_confidence == pytest.approx(0.15)
    warn = {w.code: w for w in cov.warnings}
    assert "low_avg_confidence" in warn
    assert warn["low_avg_confidence"].severity == "info"


# ---------------------------------------------------------------------------
# Session health
# ---------------------------------------------------------------------------


def _session(**overrides: Any):
    kwargs: dict[str, Any] = dict(
        session_id="sess-1",
        quality_gate_status="pass",
        trace_facts=[
            SessionTraceFacts("a", evidence_signal_count=2, has_outcome=True, has_bet=True),
            SessionTraceFacts("b", evidence_signal_count=1, has_outcome=False, has_bet=False),
        ],
        sidecar_valid=True,
        assumption_count=0,
        bug_count=0,
        audit_events=[],
        pipeline_status=None,
    )
    kwargs.update(overrides)
    return build_session_health_view(**kwargs)


def test_session_zero_evidence_traces_warns():
    view = _session(
        trace_facts=[
            SessionTraceFacts("a", evidence_signal_count=0),
            SessionTraceFacts("b", evidence_signal_count=3),
        ]
    )
    assert view.traces_zero_evidence == 1
    assert view.total_traces == 2
    assert "traces_no_evidence" in {w.code for w in view.warnings}


def test_session_sidecar_invalid_warns():
    view = _session(sidecar_valid=False)
    codes = {w.code for w in view.warnings}
    assert "sidecar_invalid" in codes


def test_session_bugs_logged_warning():
    view = _session(bug_count=2)
    warn = {w.code: w for w in view.warnings}
    assert "bugs_logged" in warn
    assert warn["bugs_logged"].severity == "warn"
    assert "2" in warn["bugs_logged"].message


def test_session_pipeline_failure_warning():
    view = _session(audit_events=[{"step": "analysis", "status": "fail"}])
    assert view.pipeline_steps_failed == ["analysis"]
    warn = {w.code: w for w in view.warnings}
    assert "pipeline_failures" in warn
    assert warn["pipeline_failures"].severity == "fail"


def test_session_failed_audit_events_counted():
    view = _session(
        audit_events=[
            {"step": "s1", "status": "ok"},
            {"step": "s2", "status": "error"},
            {"step": "s3", "status": None},
        ]
    )
    assert view.audit_event_count == 3
    assert view.failed_audit_events == 1  # only the "error" event
    assert "failed_audits" in {w.code for w in view.warnings}


def test_session_health_numbers_come_from_trace_facts_not_prose():
    # Evidence/outcome/bet counts derive only from the DB-backed trace facts;
    # changing narrative inputs (assumptions/bugs) never alters them.
    facts = [
        SessionTraceFacts("a", evidence_signal_count=2, has_outcome=True, has_bet=True),
        SessionTraceFacts("b", evidence_signal_count=4, has_outcome=False, has_bet=False),
    ]
    base = build_session_health_view(
        session_id="s",
        quality_gate_status="pass",
        trace_facts=facts,
        sidecar_valid=True,
        assumption_count=0,
        bug_count=0,
    )
    noisy = build_session_health_view(
        session_id="s",
        quality_gate_status="pass",
        trace_facts=facts,
        sidecar_valid=True,
        assumption_count=99,
        bug_count=99,
    )
    assert base.total_evidence_signals == noisy.total_evidence_signals == 6
    assert base.traces_with_outcomes == noisy.traces_with_outcomes == 1
    assert base.traces_with_bets == noisy.traces_with_bets == 1
    assert base.avg_evidence_signals_per_trace == pytest.approx(3.0)
    assert base.evidence_coverage_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Warning vocabulary + read-only guard
# ---------------------------------------------------------------------------


def test_warning_severity_levels():
    # Gather warnings from a rich recommendation + a session view; every severity
    # must be in the valid vocabulary.
    view = build_trace_recommendation_view(
        prop_trace(
            recommendation="over",
            over_prob=0.30,
            bet_side_odds=165,
            kelly_fraction=0.04,
            recommended_units=3.0,
            confidence_tier=None,
        )
    )
    session = _session(
        trace_facts=[SessionTraceFacts("a", evidence_signal_count=0)],
        sidecar_valid=False,
        bug_count=1,
        audit_events=[{"step": "x", "status": "fail"}],
    )
    all_warnings = list(_primary(view).warnings) + list(session.warnings)
    assert all_warnings  # sanity: we actually produced some
    for w in all_warnings:
        assert w.severity in VALID_SEVERITIES


def test_normalizers_do_not_import_mutation_modules():
    """Static guard: normalizers.py imports only stdlib + omega.ui.schemas."""
    path = Path(norm.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    forbidden_substrings = (
        "settle_bets",
        "ingest_traces",
        "promote_profile",
        "quarantine_",
        "backfill_",
        "omega.mcp",
        "omega.trace.store",
        "omega.trace.repository",
    )
    approved_omega = {"omega.ui.schemas"}

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(sub in module for sub in forbidden_substrings):
                offenders.append(f"from {module}")
            if module.startswith("omega.") and module not in approved_omega:
                offenders.append(f"from {module} (unapproved)")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(sub in alias.name for sub in forbidden_substrings):
                    offenders.append(f"import {alias.name}")
                if alias.name.startswith("omega.") and alias.name not in approved_omega:
                    offenders.append(f"import {alias.name} (unapproved)")
    assert offenders == [], f"normalizers.py has disallowed imports: {offenders}"
