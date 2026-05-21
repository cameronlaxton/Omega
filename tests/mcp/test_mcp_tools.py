from __future__ import annotations

from typing import Any

from omega.mcp.server import (
    PROMPT_NAMES,
    RESOURCE_URIS,
    TOOL_NAMES,
    _read_repo_file,
    omega_analyze_game,
    omega_analyze_prop,
    omega_calibration_fit_preview,
    omega_chat_orchestrate,
    omega_replay_bundle,
    omega_resolve_odds,
    omega_trace_attach_outcome,
    omega_trace_get,
    omega_trace_query,
)


def _trace_payload(trace_id: str = "mcp-trace-1") -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": "run-1",
        "timestamp": "2026-05-16T12:00:00Z",
        "prompt": "Celtics vs Lakers NBA",
        "league": "NBA",
        "matchup": "Celtics @ Lakers",
        "execution_mode": "native_sim",
        "simulation_seed": 42,
        "aggregate_quality": 1.0,
        "predictions": {"home_win_prob": 0.58, "away_win_prob": 0.42},
        "recommendations": [],
        "odds_snapshot": {"moneyline_home": -150, "moneyline_away": 130},
        "downgrades": [],
    }


def test_mcp_manifest_lists_expected_surface():
    assert "omega_analyze_game" in TOOL_NAMES
    assert "omega_replay_bundle" in TOOL_NAMES
    assert "omega_calibration_fit_preview" in TOOL_NAMES
    assert "omega_resolve_odds" in TOOL_NAMES
    assert "omega://docs/llm-mcp-interface" in RESOURCE_URIS
    assert "omega_runtime_prompt" in PROMPT_NAMES


def test_calibration_resource_path_is_readable():
    data = _read_repo_file("config/calibration/universal_latest.json")

    assert "universal_v1.0" in data


def test_analyze_game_tool_delegates_to_core_service():
    result = omega_analyze_game(
        {
            "home_team": "Boston Celtics",
            "away_team": "Indiana Pacers",
            "league": "NBA",
            "n_iterations": 1000,
            "seed": 42,
            "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
            "odds": {"moneyline_home": -160, "moneyline_away": 140},
        },
        bankroll=2500.0,
        session_id="sess-20260518-mcp",
    )

    assert result["schema_version"] == 1
    assert result["tool"] == "omega_analyze_game"
    assert result["status"] == "success"
    assert result["trace"]["trace_id"].startswith("sandbox-")
    assert result["trace"]["kind"] == "game"
    assert result["trace"]["session_id"] == "sess-20260518-mcp"
    assert result["trace"]["bankroll"] == 2500.0
    assert result["trace"]["model_version"] == "omega-core-phase6h"
    assert result["result"]["status"] == "success"


def test_analyze_prop_tool_returns_validation_errors():
    result = omega_analyze_prop(
        {
            "player_name": "Jayson Tatum",
            "league": "NBA",
            "prop_type": "pts",
            "line": 25.5,
        },
        bankroll=1000.0,
        session_id="sess-20260518-mcp",
    )

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_request"
    missing = {tuple(err["loc"]) for err in result["detail"]}
    assert ("home_team",) in missing
    assert ("away_team",) in missing
    assert ("game_date",) in missing


def test_replay_bundle_tool_marks_replay_mode_without_live_fetch():
    result = omega_replay_bundle(
        {
            "prompt": "Audit an NBA prop decision",
            "facts": [{"key": "injury_status", "filled": True, "source": "fixture"}],
            "source_trace_id": "trace-fixture",
            "decision_date": "2026-05-01",
            "simulation_seed": 123,
            "expected_outputs": ["downgrade_discipline", "trace_completeness"],
        }
    )

    assert result["status"] == "success"
    response = result["response"]
    assert response["mode"] == "replay_audit"
    assert response["live_fetch_enabled"] is False
    assert response["quant_benchmark"] is False
    trace = response["trace"]
    assert trace["facts_summary"]["replay_mode"] is True
    assert trace["facts_summary"]["source_trace_id"] == "trace-fixture"


def test_replay_bundle_rejects_live_fetch_flags():
    result = omega_replay_bundle(
        {
            "prompt": "Bad replay",
            "facts": [{"key": "odds", "live_fetch": True}],
        }
    )

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_replay_bundle"


def test_chat_orchestrate_refuses_missing_current_orchestrator():
    result = omega_chat_orchestrate("Run the whole agent")

    assert result["status"] == "error"
    assert result["error_code"] == "unsupported_current_repo"
    assert "no MCP chat orchestrator" in result["detail"]["message"]


def test_trace_tools_round_trip(tmp_path):
    from omega.trace.store import TraceStore

    db_path = str(tmp_path / "traces.db")
    store = TraceStore(db_path=db_path)
    store.persist(_trace_payload())
    store.close()

    query = omega_trace_query(db_path=db_path, league="NBA")
    assert query["status"] == "success"
    assert len(query["traces"]) == 1

    got = omega_trace_get("mcp-trace-1", db_path=db_path)
    assert got["status"] == "success"
    assert got["trace"]["trace_id"] == "mcp-trace-1"

    attached = omega_trace_attach_outcome(
        "mcp-trace-1",
        home_score=112,
        away_score=105,
        source="test",
        db_path=db_path,
    )
    assert attached["status"] == "success"

    graded = omega_trace_query(db_path=db_path, has_outcome=True)
    assert len(graded["traces"]) == 1
    assert graded["traces"][0]["_outcome"]["result"] == "home_win"


def test_calibration_preview_is_dry_run_when_insufficient_data(tmp_path):
    result = omega_calibration_fit_preview(db_path=str(tmp_path / "traces.db"), plane="prop")

    assert result["status"] == "success"
    assert result["result"]["status"] == "skipped"
    assert result["result"]["dry_run"] is True
    assert result["result"]["sample_size"] == 0
    assert result["result"]["plane"] == "prop"
    assert result["result"]["pair_type"] == "prop probability/outcome"


def test_resolve_odds_tool_returns_input_prep_result(monkeypatch):
    import scripts.resolve_odds as resolver

    def fake_resolve_odds(**kwargs):
        assert kwargs["bookmaker"] == "betmgm"
        return {"status": "unavailable", "request_patch": None, "skipped_reasons": ["fixture"]}

    monkeypatch.setattr(resolver, "resolve_odds", fake_resolve_odds)

    result = omega_resolve_odds(kind="game", league="NBA", event_id="evt-1")

    assert result["status"] == "success"
    assert result["result"]["status"] == "unavailable"
