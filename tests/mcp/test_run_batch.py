"""Unit tests for omega_run_batch."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from omega.mcp.server import TOOL_NAMES, omega_run_batch


def _make_trace(trace_id: str = "sandbox-batch-001") -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "session_id": "sess-test",
        "bankroll": 1000.0,
        "league": "MLB",
        "kind": "prop",
        "result": {"status": "success"},
        "trace_quality": {"aggregate_quality": 0.85},
    }


def _prop_entry(**kwargs: Any) -> dict[str, Any]:
    base = {
        "kind": "prop",
        "league": "MLB",
        "player_name": "Julio Rodríguez",
        "prop_type": "hits",
        "home_team": "Seattle Mariners",
        "away_team": "New York Mets",
        "game_date": "2026-06-03",
        "player_context": {"hits_mean": 1.15, "hits_std": 0.8},
        "game_context": {"is_playoff": False, "rest_days": 1},
    }
    base.update(kwargs)
    return base


def _game_entry(**kwargs: Any) -> dict[str, Any]:
    base = {
        "kind": "game",
        "league": "MLB",
        "home_team": "New York Yankees",
        "away_team": "Cleveland Guardians",
        "game_date": "2026-06-03",
        "home_context": {"off_rating": 5.15, "def_rating": 3.60, "starter_era": 0.71},
        "away_context": {"off_rating": 4.16, "def_rating": 4.06, "starter_era": 3.07},
        "game_context": {"is_playoff": False, "rest_days": 1},
    }
    base.update(kwargs)
    return base


def _mock_resolve_odds_ok(kind: str, **kwargs: Any) -> dict[str, Any]:
    if kind == "prop":
        return {
            "status": "success",
            "request_patch": {"line": 0.5, "odds_over": -115, "odds_under": -105},
        }
    return {
        "status": "success",
        "request_patch": {"odds": {"moneyline_home": -180, "moneyline_away": 150}},
    }


# --- Gate-blocked ---


def test_gate_failure_blocks_entire_batch(tmp_path: Path) -> None:
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=["preflight_failed"]),
    ):
        result = omega_run_batch(
            entries=[_prop_entry()],
            bankroll=1000.0,
            session_id="sess-test",
        )
    assert result["status"] == "error"
    assert result["error_code"] == "formal_output_blocked"


# --- Happy path ---


def test_happy_path_prop_writes_export_block(tmp_path: Path) -> None:
    trace = _make_trace("sandbox-prop-001")
    presentation = {
        "thesis": "Contact form supports the over.",
        "market_read": "BetMGM line is still playable.",
        "why": "Recent role and matchup both point up.",
        "risks": "Weather can suppress contact.",
        "verdict": "Research lean if the line holds.",
    }
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_mock_resolve_odds_ok),
        patch("omega.core.contracts.service.analyze", return_value=trace),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[
                _prop_entry(
                    reasoning_narrative="Julio's contact profile fits this matchup.",
                    reasoning_presentation=presentation,
                )
            ],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["status"] == "ok"
    assert result["entries_ok"] == 1
    assert result["entries_skipped"] == 0
    assert result["entries_error"] == 0
    assert "sandbox-prop-001" in result["trace_ids"]
    # Export file should exist
    export_path = Path(result["export_paths"][0])
    assert export_path.exists()
    import json

    block = json.loads(export_path.read_text())
    assert block["export_schema_version"] == 2
    assert block["trace"]["trace_id"] == "sandbox-prop-001"
    assert "reasoning_inputs" not in block
    assert "reasoning_narrative" not in block
    assert "reasoning_presentation" not in block
    assert block["trace"]["reasoning_inputs"]["market_context"]["prop_type"] == "hits"
    assert block["trace"]["reasoning_narrative"] == "Julio's contact profile fits this matchup."
    assert block["trace"]["reasoning_presentation"] == presentation
    # A4: export block carries a top-level session_id so the prediction->session
    # link survives outside the DB and passes the strict export validator.
    assert block["session_id"] == "sess-test"


def test_batch_reasoning_presentation_rejects_extra_or_numeric_fields() -> None:
    with patch("omega.mcp.server._formal_output_gate_failures", return_value=[]):
        result = omega_run_batch(
            entries=[
                _prop_entry(
                    reasoning_presentation={
                        "thesis": "Qualitative only.",
                        "edge_pct": "not allowed",
                        "why": 12,
                    }
                )
            ],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["status"] == "error"
    assert result["entries_error"] == 1
    assert "reasoning_presentation" in result["errors"][0]["error"]


def test_batch_reasoning_inputs_rejects_protected_market_context(tmp_path: Path) -> None:
    trace = _make_trace("sandbox-game-protected-ri")
    trace["kind"] = "game"
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.core.contracts.service.analyze", return_value=trace),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[
                _game_entry(
                    odds={
                        "moneyline_home": -120,
                        "moneyline_away": 100,
                        "edge_pct": 4.2,
                    }
                )
            ],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["status"] == "error"
    assert result["entries_error"] == 1
    assert result["export_paths"] == []
    assert "reasoning_inputs" in result["errors"][0]["error"]
    assert "edge_pct" in result["errors"][0]["error"]


def test_happy_path_game_writes_export_block(tmp_path: Path) -> None:
    trace = _make_trace("sandbox-game-001")
    trace["kind"] = "game"
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_mock_resolve_odds_ok),
        patch("omega.core.contracts.service.analyze", return_value=trace),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_game_entry()],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_ok"] == 1
    assert "sandbox-game-001" in result["trace_ids"]


def test_seed_derivation_is_independent_of_session_id(tmp_path: Path) -> None:
    seen_requests: list[dict[str, Any]] = []

    def _analyze(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        seen_requests.append(dict(request))
        return _make_trace(f"sandbox-seed-{len(seen_requests)}")

    entry = _prop_entry(line=0.5, odds_over=-115, odds_under=-105)
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.core.contracts.service.analyze", side_effect=_analyze),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        first = omega_run_batch(entries=[entry], bankroll=1000.0, session_id="sess-one")
        second = omega_run_batch(entries=[entry], bankroll=1000.0, session_id="sess-two")

    assert first["entries_ok"] == 1
    assert second["entries_ok"] == 1
    assert seen_requests[0]["seed"] == seen_requests[1]["seed"]


def test_resolve_odds_uses_entry_game_date_window(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def _resolve(kind: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"kind": kind, **kwargs})
        return _mock_resolve_odds_ok(kind, **kwargs)

    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_resolve),
        patch("omega.core.contracts.service.analyze", return_value=_make_trace("sandbox-date-001")),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_prop_entry(game_date="2026-06-07")],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_ok"] == 1
    assert calls
    # The commence window is the league-local day (default America/New_York),
    # converted to UTC, so late local games are preserved (P1 late-slate fix).
    # Midnight EDT on 2026-06-07 is 04:00Z; the window runs to the next local midnight.
    assert calls[0]["commence_time_from"] == "2026-06-07T04:00:00Z"
    assert calls[0]["commence_time_to"] == "2026-06-08T04:00:00Z"


# --- Odds unavailable ---


def test_odds_unavailable_skips_entry(tmp_path: Path) -> None:
    def _no_odds(kind: str, **kwargs: Any) -> dict[str, Any]:
        return {"status": "error", "message": "no_markets"}

    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_no_odds),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_prop_entry()],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_skipped"] == 1
    assert result["entries_ok"] == 0
    # all-skipped with no errors → status is "ok" (nothing went wrong, just no markets)
    assert result["status"] == "ok"


# --- prop_type fallback chain ---


def test_prop_type_fallback_chain(tmp_path: Path) -> None:
    trace = _make_trace("sandbox-fallback-001")

    call_log: list[str] = []

    def _fallback_resolve(kind: str, prop_type: str | None = None, **kwargs: Any) -> dict[str, Any]:
        call_log.append(str(prop_type))
        if prop_type == "hits":
            return {"status": "error"}
        return {
            "status": "success",
            "request_patch": {"line": 1.5, "odds_over": -110, "odds_under": -110},
        }

    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_fallback_resolve),
        patch("omega.core.contracts.service.analyze", return_value=trace),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_prop_entry(prop_type=["hits", "total_bases"])],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert "hits" in call_log
    assert "total_bases" in call_log
    assert result["entries_ok"] == 1


# --- Per-entry exception does not abort batch ---


def test_per_entry_error_continues_batch(tmp_path: Path) -> None:
    traces = [_make_trace("sandbox-ok-001"), _make_trace("sandbox-ok-002")]
    call_count = 0

    def _analyze_with_error(request: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated engine error")
        return traces[call_count - 2]

    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_mock_resolve_odds_ok),
        patch("omega.core.contracts.service.analyze", side_effect=_analyze_with_error),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_prop_entry(), _prop_entry(player_name="Rafael Devers")],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_error"] == 1
    assert result["entries_ok"] == 1
    assert result["status"] == "partial"


# --- Invalid entry validation error ---


def test_invalid_entry_captured_as_error(tmp_path: Path) -> None:
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[{"kind": "invalid_kind", "league": "MLB"}],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_error"] == 1
    assert len(result["errors"]) == 1


# --- RSVG roster gate ---


def _rsvg_context(**overrides: Any) -> dict[str, Any]:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    base: dict[str, Any] = {
        "home_team": "New York Yankees",
        "away_team": "Cleveland Guardians",
        "league": "MLB",
        "game_date": "2026-06-03",
        "source_summaries": [{"source": "espn.com", "summary": "Lineups posted.", "retrieved_at": now}],
        "home_status": {"lineup_status": "confirmed", "injury_report_checked": True},
        "away_status": {"lineup_status": "confirmed", "injury_report_checked": True},
        "absences": [],
        "gathered_at": now,
        "roster_context_complete": True,
    }
    base.update(overrides)
    return base


def test_rsvg_blocked_entry_skips_before_analyze(tmp_path: Path) -> None:
    analyze_calls: list[Any] = []

    def _analyze(request: Any, **kwargs: Any) -> dict[str, Any]:
        analyze_calls.append(request)
        return _make_trace("sandbox-rsvg-blocked")

    blocked_context = _rsvg_context(
        home_status={"lineup_status": "unknown", "injury_report_checked": False},
        roster_context_complete=False,
    )
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_mock_resolve_odds_ok),
        patch("omega.core.contracts.service.analyze", side_effect=_analyze),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_game_entry(roster_context=blocked_context)],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_skipped"] == 1
    assert result["entries_ok"] == 0
    assert analyze_calls == []  # blocked entries never reach analyze()
    row = result["results"][0]
    assert row["reason"] == "rsvg_blocked"
    assert row["rsvg"]["status"] == "blocked"
    assert row["rsvg"]["gate"] == "rsvg"


def test_rsvg_key_absence_merges_evidence_and_stamps_trace(tmp_path: Path) -> None:
    seen_requests: list[dict[str, Any]] = []

    def _analyze(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        seen_requests.append(dict(request))
        trace = _make_trace("sandbox-rsvg-pass")
        trace["kind"] = "game"
        return trace

    context = _rsvg_context(
        absences=[
            {
                "player": "Aaron Judge",
                "team_side": "home",
                "status": "out",
                "is_key_player": True,
            }
        ]
    )
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_mock_resolve_odds_ok),
        patch("omega.core.contracts.service.analyze", side_effect=_analyze),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_game_entry(roster_context=context)],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_ok"] == 1
    assert result["results"][0]["rsvg_status"] == "pass"
    # The usage_role_change signal reached the analyze() request.
    evidence = seen_requests[0]["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["signal_type"] == "usage_role_change"
    assert evidence[0]["direction"] == "away"
    # And the export block carries the audit + no downgrade for a pass.
    import json

    block = json.loads(Path(result["export_paths"][0]).read_text())
    assert block["trace"]["trace_quality"]["rsvg"]["status"] == "pass"
    assert block["trace"]["trace_quality"]["rsvg"]["key_absences"]["home"] == ["Aaron Judge"]
    assert block["trace"]["reasoning_downgrade_rationale"] is None
    # RSVG presentation fills in when the entry supplied none.
    assert block["trace"]["reasoning_presentation"]["verdict"].startswith("PASS")
    assert "espn.com" in block["trace"]["reasoning_inputs"]["sources"]


def test_rsvg_research_candidate_stamps_downgrade_rationale(tmp_path: Path) -> None:
    context = _rsvg_context(
        absences=[
            {"player": p, "team_side": "home", "status": "out", "is_key_player": True}
            for p in ("Star A", "Star B", "Star C")
        ]
    )
    trace = _make_trace("sandbox-rsvg-research")
    trace["kind"] = "game"
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_mock_resolve_odds_ok),
        patch("omega.core.contracts.service.analyze", return_value=trace),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_game_entry(roster_context=context)],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_ok"] == 1
    assert result["results"][0]["rsvg_status"] == "research_candidate"
    import json

    block = json.loads(Path(result["export_paths"][0]).read_text())
    assert block["trace"]["trace_quality"]["rsvg"]["status"] == "research_candidate"
    assert block["trace"]["trace_quality"]["rsvg"]["formal_output_allowed"] is False
    rationale = block["trace"]["reasoning_downgrade_rationale"]
    assert rationale is not None and "RSVG research_candidate" in rationale


def test_rsvg_export_block_passes_export_validator(tmp_path: Path) -> None:
    """No top-level shape drift: RSVG-stamped exports (trace_quality.rsvg,
    reasoning_presentation, downgrade rationale) stay valid for the shared
    export validator that guards ingest."""
    context = _rsvg_context(
        absences=[
            {"player": p, "team_side": "home", "status": "out", "is_key_player": True}
            for p in ("Star A", "Star B", "Star C")
        ]
    )
    trace = _make_trace("sandbox-rsvg-shape")
    trace["kind"] = "game"
    # A real analyze() return carries timestamp + identity; the minimal mock
    # doesn't, and those gaps are what the validator would (correctly) flag.
    trace["ran_at"] = "2026-07-03T18:00:00Z"
    trace["input_snapshot"] = {
        "home_team": "New York Yankees",
        "away_team": "Cleveland Guardians",
        "league": "MLB",
        "game_date": "2026-06-03",
        "seed": 42,
    }
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_mock_resolve_odds_ok),
        patch("omega.core.contracts.service.analyze", return_value=trace),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_game_entry(roster_context=context)],
            bankroll=1000.0,
            session_id="sess-test",
        )

    import json

    from omega.trace.export_validator import validate_export_block

    block = json.loads(Path(result["export_paths"][0]).read_text())
    report = validate_export_block(block, strict=False)
    assert report.ok, f"export validator errors: {report.errors}"
    # And the reasoning surface survived intact inside the inner trace.
    assert block["trace"]["reasoning_presentation"]["verdict"].startswith("RESEARCH_CANDIDATE")
    assert block["trace"]["trace_quality"]["rsvg"]["output_mode_ceiling"] == "research_candidate"


def test_rsvg_invalid_payload_is_entry_error(tmp_path: Path) -> None:
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_game_entry(roster_context={"home_team": "Only"})],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_error"] == 1
    assert "rsvg_invalid" in result["errors"][0]["error"]


def test_rsvg_payload_with_protected_field_is_rejected(tmp_path: Path) -> None:
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_game_entry(roster_context=_rsvg_context(edge_pct=4.2))],
            bankroll=1000.0,
            session_id="sess-test",
        )

    assert result["entries_error"] == 1
    assert "rsvg_invalid" in result["errors"][0]["error"]


# --- Tool registered ---


def test_omega_run_batch_in_tool_names() -> None:
    assert "omega_run_batch" in TOOL_NAMES
