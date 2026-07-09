"""Unit tests for the RSVG Roster & Situational Verification Gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from omega.core.calibration.adjustment_policy import AdjustmentPolicyRegistry
from omega.core.contracts.evidence import EvidenceSignal
from omega.core.contracts.schemas import BatchAnalysisEntry, ReasoningPresentation
from omega.core.gates.rsvg import (
    _PROTECTED_QUANT_FIELDS,
    MAX_KEY_ABSENCES_PER_TEAM,
    RosterContextPayload,
    RsvgProtectedFieldError,
    RsvgResult,
    _assert_no_protected_fields,
    evaluate_roster_context,
)
from omega.core.simulation.evidence_handlers import compute_game_adjustment
from omega.core.simulation.evidence_to_modifier import signals_to_transition_modifiers

_NOW = datetime(2026, 7, 3, 15, 0, 0, tzinfo=timezone.utc)


def _payload(**overrides: Any) -> RosterContextPayload:
    """A complete, fresh, verified payload; overrides poke holes per-test."""
    base: dict[str, Any] = {
        "home_team": "Denver Nuggets",
        "away_team": "Boston Celtics",
        "league": "NBA",
        "game_date": "2026-07-03",
        "source_summaries": [
            {
                "source": "espn.com",
                "summary": "Both lineups confirmed; no late scratches expected.",
                "retrieved_at": (_NOW - timedelta(hours=1)).isoformat(),
            }
        ],
        "home_status": {"lineup_status": "confirmed", "injury_report_checked": True},
        "away_status": {"lineup_status": "confirmed", "injury_report_checked": True},
        "absences": [],
        "motivation_notes": "Regular-season game; no seeding implications.",
        "gathered_at": (_NOW - timedelta(hours=1)).isoformat(),
        "roster_context_complete": True,
    }
    base.update(overrides)
    return RosterContextPayload(**base)


def _absence(player: str, side: str = "home", status: str = "out", key: bool = False) -> dict:
    return {"player": player, "team_side": side, "status": status, "is_key_player": key}


# --- Complete context passes -------------------------------------------------


def test_complete_roster_context_passes() -> None:
    result = evaluate_roster_context(_payload(), evaluated_at=_NOW)
    assert result.status == "pass"
    assert result.formal_output_allowed is True
    assert result.reasoning_downgrade_rationale is None
    assert result.evidence == []
    assert result.gate_audit.missing_context == []
    assert result.gate_audit.output_mode_ceiling == "unrestricted"
    assert result.reasoning_presentation.verdict.startswith("PASS")
    # AGENTS.md RSVG persistence rule: news summaries live in the presentation.
    assert "no late scratches" in (result.reasoning_presentation.why or "")


# --- Duplicate-summary content-quality downgrade ------------------------------


def test_duplicate_summary_detected_forces_research_candidate() -> None:
    """A caller (omega_run_batch) that detects the exact same source_summaries
    text reused across different matchups must be able to downgrade an
    otherwise-passing entry -- boilerplate reused verbatim was never
    independently verified for this specific matchup."""
    result = evaluate_roster_context(_payload(), evaluated_at=_NOW, duplicate_summary_detected=True)
    assert result.status == "research_candidate"
    assert result.formal_output_allowed is False
    assert "reused verbatim" in (result.reasoning_downgrade_rationale or "")
    assert result.gate_audit.output_mode_ceiling == "research_candidate"


def test_duplicate_flag_false_by_default_does_not_downgrade() -> None:
    result = evaluate_roster_context(_payload(), evaluated_at=_NOW)
    assert result.status == "pass"


def test_duplicate_flag_irrelevant_when_source_summaries_already_empty() -> None:
    """Empty source_summaries is already caught by the missing-context check;
    the duplicate flag must not double-count or change that verdict's shape."""
    result = evaluate_roster_context(
        _payload(source_summaries=[]), evaluated_at=_NOW, duplicate_summary_detected=True
    )
    assert result.status == "research_candidate"
    assert "source_summaries" in result.gate_audit.missing_context
    assert "reused verbatim" not in (result.reasoning_downgrade_rationale or "")


# --- Non-key absences warn but do not block -----------------------------------


def test_non_key_absences_pass_with_notes() -> None:
    payload = _payload(absences=[_absence("Role Player", side="away", status="out", key=False)])
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "pass"
    assert result.evidence == []  # no signal for non-key absences
    assert result.gate_audit.non_key_absences["away"] == ["Role Player"]
    assert any("Role Player" in n for n in result.gate_audit.notes)
    assert "Role Player" in (result.reasoning_presentation.risks or "")


# --- Key absence emits usage_role_change --------------------------------------


def test_one_key_absence_emits_usage_role_change_for_absent_team() -> None:
    payload = _payload(
        absences=[
            {
                "player": "Nikola Jokic",
                "team_side": "home",
                "status": "out",
                "is_key_player": True,
                "role_note": "starting center",
            }
        ]
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "pass"  # <=2 key absences with verified context stays pass
    assert len(result.evidence) == 1
    sig = result.evidence[0]
    assert isinstance(sig, EvidenceSignal)
    assert sig.signal_type == "usage_role_change"
    assert sig.category == "situational"
    assert sig.plane == "game"
    assert sig.direction == "home"
    assert sig.value == "bench"
    assert sig.source == "injury_report"
    assert sig.confidence == pytest.approx(0.9)
    assert "Nikola Jokic" in (sig.note or "")
    assert "starting center" in (sig.note or "")


def test_key_absence_targets_absent_team_in_deterministic_handlers() -> None:
    payload = _payload(
        absences=[_absence("Nikola Jokic", side="home", status="out", key=True)]
    )
    sig = evaluate_roster_context(payload, evaluated_at=_NOW).evidence[0]

    game_adjustment = compute_game_adjustment(
        evidence=[sig],
        league="NBA",
        policy=AdjustmentPolicyRegistry().get_production_policy(),
        evidence_mode="bounded_live",
    )
    assert game_adjustment.home_factor < 1.0
    assert game_adjustment.away_factor == 1.0

    markov_modifiers = signals_to_transition_modifiers([sig], home_team="Denver Nuggets")
    assert markov_modifiers["home_score_rate_scalar"] == pytest.approx(0.93)
    assert "away_score_rate_scalar" not in markov_modifiers


def test_away_key_absence_direction_and_doubtful_confidence() -> None:
    payload = _payload(
        absences=[_absence("Jayson Tatum", side="away", status="doubtful", key=True)]
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    sig = result.evidence[0]
    assert sig.direction == "away"
    assert sig.confidence == pytest.approx(0.6)


# --- >2 key absences forces research_candidate --------------------------------


def test_more_than_two_key_absences_forces_research_candidate() -> None:
    payload = _payload(
        absences=[
            _absence("Star A", side="home", status="out", key=True),
            _absence("Star B", side="home", status="out", key=True),
            _absence("Star C", side="home", status="suspended", key=True),
        ]
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "research_candidate"
    assert result.formal_output_allowed is False
    assert result.gate_audit.output_mode_ceiling == "research_candidate"
    assert len(result.evidence) == 3  # every key absence still emits its signal
    assert result.reasoning_downgrade_rationale is not None
    assert "3 missing key players" in result.reasoning_downgrade_rationale
    assert str(MAX_KEY_ABSENCES_PER_TEAM) in result.reasoning_downgrade_rationale


def test_two_key_absences_on_one_team_still_passes() -> None:
    payload = _payload(
        absences=[
            _absence("Star A", side="home", status="out", key=True),
            _absence("Star B", side="home", status="out", key=True),
        ]
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "pass"
    assert len(result.evidence) == 2


def test_threshold_is_per_team_not_total() -> None:
    payload = _payload(
        absences=[
            _absence("Star A", side="home", status="out", key=True),
            _absence("Star B", side="home", status="out", key=True),
            _absence("Star C", side="away", status="out", key=True),
            _absence("Star D", side="away", status="out", key=True),
        ]
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "pass"  # 2 per team, threshold never exceeded


# --- Missing lineup/injury context ---------------------------------------------


def test_missing_lineup_context_produces_downgrade_rationale() -> None:
    payload = _payload(
        home_status={"lineup_status": "unknown", "injury_report_checked": True}
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "research_candidate"
    assert "home.lineup_status" in result.gate_audit.missing_context
    assert result.reasoning_downgrade_rationale is not None
    assert "home.lineup_status" in result.reasoning_downgrade_rationale


def test_unchecked_injury_report_downgrades() -> None:
    payload = _payload(
        away_status={"lineup_status": "confirmed", "injury_report_checked": False}
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "research_candidate"
    assert "away.injury_report" in result.gate_audit.missing_context


def test_fully_dark_team_blocks() -> None:
    payload = _payload(
        home_status={"lineup_status": "unknown", "injury_report_checked": False}
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "blocked"
    assert result.formal_output_allowed is False
    assert result.gate_audit.output_mode_ceiling == "blocked"
    assert result.reasoning_downgrade_rationale is not None
    assert "Denver Nuggets" in result.reasoning_downgrade_rationale
    assert result.reasoning_presentation.verdict.startswith("BLOCKED")


def test_incomplete_attestation_downgrades() -> None:
    result = evaluate_roster_context(
        _payload(roster_context_complete=False), evaluated_at=_NOW
    )
    assert result.status == "research_candidate"
    assert "roster_context_complete" in result.gate_audit.missing_context


def test_stale_context_downgrades() -> None:
    payload = _payload(gathered_at=(_NOW - timedelta(hours=48)).isoformat())
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    assert result.status == "research_candidate"
    assert result.reasoning_downgrade_rationale is not None
    assert "stale" in result.reasoning_downgrade_rationale
    assert result.gate_audit.context_age_hours == pytest.approx(48.0)


# --- Contract compatibility -----------------------------------------------------


def test_output_validates_as_evidence_and_presentation_and_batch_entry() -> None:
    payload = _payload(
        absences=[_absence("Nikola Jokic", side="home", status="out", key=True)]
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)
    fields = result.to_batch_entry_fields()

    # Evidence dicts round-trip through the typed contract.
    for d in fields["evidence"]:
        EvidenceSignal(**d)
    # Presentation dict round-trips, and carries all five analyst-note keys.
    pres = ReasoningPresentation(**fields["reasoning_presentation"])
    assert set(fields["reasoning_presentation"]) == {
        "thesis",
        "market_read",
        "why",
        "risks",
        "verdict",
    }
    assert pres.thesis and pres.why and pres.risks and pres.verdict
    # The whole fragment builds a valid BatchAnalysisEntry.
    entry = BatchAnalysisEntry(
        kind="game",
        league="NBA",
        home_team="Denver Nuggets",
        away_team="Boston Celtics",
        roster_context=payload.model_dump(),
        **fields,
    )
    assert entry.evidence == fields["evidence"]
    # trace_quality fragment is a plain persistable dict under the 'rsvg' key.
    tq = result.trace_quality_fragment()
    assert tq["rsvg"]["gate"] == "rsvg"
    assert tq["rsvg"]["status"] == "pass"


# --- Protected betting fields -----------------------------------------------------


def test_payload_rejects_protected_betting_fields() -> None:
    with pytest.raises(ValidationError):
        RosterContextPayload(
            home_team="A",
            away_team="B",
            league="NBA",
            game_date="2026-07-03",
            edge_pct=4.2,
        )
    with pytest.raises(ValidationError):
        RosterContextPayload(
            home_team="A",
            away_team="B",
            league="NBA",
            game_date="2026-07-03",
            kelly_fraction=0.1,
        )


def test_emitted_payloads_are_free_of_protected_fields() -> None:
    payload = _payload(
        absences=[
            _absence("Star A", side="home", status="out", key=True),
            _absence("Star B", side="home", status="out", key=True),
            _absence("Star C", side="home", status="out", key=True),
            _absence("Bench Guy", side="away", status="doubtful", key=False),
        ]
    )
    result = evaluate_roster_context(payload, evaluated_at=_NOW)

    def _walk_keys(value: Any) -> set[str]:
        keys: set[str] = set()
        if isinstance(value, dict):
            for k, v in value.items():
                keys.add(str(k))
                keys |= _walk_keys(v)
        elif isinstance(value, (list, tuple)):
            for item in value:
                keys |= _walk_keys(item)
        return keys

    emitted = (
        _walk_keys(result.gate_audit.model_dump())
        | _walk_keys(result.reasoning_presentation.model_dump())
        | _walk_keys(result.evidence_dicts())
        | _walk_keys(result.trace_quality_fragment())
    )
    assert emitted.isdisjoint(_PROTECTED_QUANT_FIELDS)


def test_protected_scan_fails_closed() -> None:
    with pytest.raises(RsvgProtectedFieldError):
        _assert_no_protected_fields("test", {"nested": [{"edge_pct": 4.2}]})


def test_result_model_rejects_protected_fields_on_direct_construction() -> None:
    result = evaluate_roster_context(_payload(), evaluated_at=_NOW)
    audit = result.gate_audit.model_copy(update={"policy": {"edge_pct": 4.2}})

    with pytest.raises(ValidationError, match="edge_pct"):
        RsvgResult(
            status=result.status,
            evidence=result.evidence,
            reasoning_presentation=result.reasoning_presentation,
            reasoning_downgrade_rationale=result.reasoning_downgrade_rationale,
            reasoning_sources=result.reasoning_sources,
            gate_audit=audit,
        )


def test_protected_set_superset_of_sidecar_set() -> None:
    # The gate's protected list must never fall behind the sidecar's canonical
    # one (core cannot import trace, so the sync is enforced here instead).
    from omega.trace.session_sidecar import _PROTECTED_QUANT_FIELDS as sidecar_fields

    assert sidecar_fields <= _PROTECTED_QUANT_FIELDS
