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
    with (
        patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
        patch("omega.integrations.odds_resolver.resolve_odds", side_effect=_mock_resolve_odds_ok),
        patch("omega.core.contracts.service.analyze", return_value=trace),
        patch("omega.paths.repo_root", return_value=tmp_path),
    ):
        result = omega_run_batch(
            entries=[_prop_entry()],
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
    assert block["trace"]["trace_id"] == "sandbox-prop-001"
    assert "reasoning_inputs" in block
    assert "reasoning_narrative" in block
    # A4: export block carries a top-level session_id so the prediction->session
    # link survives outside the DB and passes the strict export validator.
    assert block["session_id"] == "sess-test"


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


# --- Tool registered ---


def test_omega_run_batch_in_tool_names() -> None:
    assert "omega_run_batch" in TOOL_NAMES
