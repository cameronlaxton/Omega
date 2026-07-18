"""Phase 0 MCP tests: run_batch mode stamping + the safe matchup-brief tool."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from omega.mcp.server import TOOL_NAMES, omega_get_matchup_brief, omega_run_batch
from omega.trace.store import TraceStore


def _make_trace(trace_id: str = "sandbox-p0-001") -> dict[str, Any]:
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
        "line": 0.5,
        "odds_over": -115,
        "odds_under": -105,
        "player_context": {"hits_mean": 1.15, "hits_std": 0.8},
        "game_context": {"is_playoff": False, "rest_days": 1},
    }
    base.update(kwargs)
    return base


def test_matchup_brief_tool_is_registered():
    assert "omega_get_matchup_brief" in TOOL_NAMES


class TestModeValidation:
    def test_invalid_presentation_mode_is_rejected(self):
        result = omega_run_batch(
            entries=[], bankroll=1000.0, session_id="s", presentation_mode="bogus"
        )
        assert result["status"] == "error"
        assert result["error_code"] == "INVALID_INPUT"
        assert "presentation_mode" in result["detail"]

    def test_invalid_ledger_mode_is_rejected(self):
        result = omega_run_batch(
            entries=[], bankroll=1000.0, session_id="s", engine_auto_ledger_mode="on"
        )
        assert result["status"] == "error"
        assert result["error_code"] == "INVALID_INPUT"
        assert "engine_auto_ledger_mode" in result["detail"]


class TestBatchStamping:
    def _run(self, tmp_path: Path, entry: dict[str, Any], **modes: Any) -> dict[str, Any]:
        with (
            patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
            patch("omega.core.contracts.service.analyze", return_value=_make_trace()),
            patch("omega.paths.repo_root", return_value=tmp_path),
        ):
            result = omega_run_batch(
                entries=[entry], bankroll=1000.0, session_id="sess-test", **modes
            )
        assert result["status"] == "ok", result
        return json.loads(Path(result["export_paths"][0]).read_text())

    def test_defaults_stamp_decision_support_and_disabled(self, tmp_path: Path):
        block = self._run(tmp_path, _prop_entry())
        trace = block["trace"]
        assert trace["schema_version"] == 2
        assert trace["presentation_mode"] == "decision_support"
        assert trace["engine_auto_ledger_mode"] == "disabled"
        assert trace["event_identity"] is None

    def test_event_id_builds_shared_event_identity(self, tmp_path: Path):
        block = self._run(tmp_path, _prop_entry(event_id="ev-42"))
        identity = block["trace"]["event_identity"]
        assert identity["provider"] == "the-odds-api"
        assert identity["provider_event_id"] == "ev-42"
        assert identity["event_key"] == "MLB::the-odds-api::ev-42"
        assert identity["home_team"] == "Seattle Mariners"
        assert identity["game_date"] == "2026-06-03"

    def test_explicit_modes_are_stamped(self, tmp_path: Path):
        block = self._run(
            tmp_path,
            _prop_entry(),
            presentation_mode="recommendation_lab",
            engine_auto_ledger_mode="shadow",
        )
        trace = block["trace"]
        assert trace["presentation_mode"] == "recommendation_lab"
        assert trace["engine_auto_ledger_mode"] == "shadow"

    def test_decision_support_presentation_rides_the_trace(self, tmp_path: Path):
        payload = {
            "matchup_summary": "Contact-heavy lineup against a fly-ball starter.",
            "market_context": "Line has held at 0.5 across books.",
            "outcome_cases": [
                {
                    "market_key": "hits",
                    "outcome_key": "over",
                    "label": "Over 0.5 hits",
                    "supporting": ["High contact rate."],
                    "challenging": ["Tough lefty matchup."],
                    "data_status": "complete",
                },
                {
                    "market_key": "hits",
                    "outcome_key": "under",
                    "label": "Under 0.5 hits",
                    "supporting": ["Tough lefty matchup."],
                    "challenging": ["High contact rate."],
                    "data_status": "complete",
                },
            ],
            "uncertainties": ["Lineup slot not confirmed."],
        }
        block = self._run(tmp_path, _prop_entry(decision_support_presentation=payload))
        stamped = block["trace"]["decision_support_presentation"]
        assert stamped["matchup_summary"] == payload["matchup_summary"]
        assert len(stamped["outcome_cases"]) == 2

    def test_blocked_language_in_presentation_errors_the_entry(self, tmp_path: Path):
        payload = {
            "matchup_summary": "This is our best bet tonight.",
            "market_context": "x",
        }
        with (
            patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
            patch("omega.core.contracts.service.analyze", return_value=_make_trace()),
            patch("omega.paths.repo_root", return_value=tmp_path),
        ):
            result = omega_run_batch(
                entries=[_prop_entry(decision_support_presentation=payload)],
                bankroll=1000.0,
                session_id="sess-test",
            )
        assert result["entries_error"] == 1
        assert "blocked language" in result["errors"][0]["error"]


def _mock_resolver(event_id: str = "prov-ev-1", commence: str = "2026-06-03T23:10:00Z"):
    def _resolve(kind: str, **kwargs: Any) -> dict[str, Any]:
        patch_payload = (
            {"line": 0.5, "odds_over": -115, "odds_under": -105}
            if kind == "prop"
            else {"odds": {"moneyline_home": -180, "moneyline_away": 150}}
        )
        return {
            "status": "success",
            "event_id": event_id,
            "commence_time": commence,
            "request_patch": patch_payload,
        }

    return _resolve


class TestResolverEventIdentity:
    """Phase 1: live odds resolution anchors the provider event identity."""

    def _run(self, tmp_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
        with (
            patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
            patch(
                "omega.integrations.odds_resolver.resolve_odds",
                side_effect=_mock_resolver(),
            ),
            patch("omega.core.contracts.service.analyze", return_value=_make_trace()),
            patch("omega.paths.repo_root", return_value=tmp_path),
        ):
            return omega_run_batch(
                entries=[entry], bankroll=1000.0, session_id="sess-test"
            )

    def _entry_needing_resolution(self, **kwargs: Any) -> dict[str, Any]:
        entry = _prop_entry(**kwargs)
        # Strip pre-supplied odds so the batch tool resolves them live.
        for key in ("line", "odds_over", "odds_under"):
            entry.pop(key, None)
        return entry

    def test_resolved_identity_is_provider_anchored(self, tmp_path: Path):
        result = self._run(tmp_path, self._entry_needing_resolution())
        assert result["status"] == "ok", result
        block = json.loads(Path(result["export_paths"][0]).read_text())
        identity = block["trace"]["event_identity"]
        assert identity["provider"] == "the-odds-api"
        assert identity["provider_event_id"] == "prov-ev-1"
        assert identity["event_key"] == "MLB::the-odds-api::prov-ev-1"
        assert identity["commence_time"] == "2026-06-03T23:10:00Z"

    def test_matching_caller_id_confirmed_by_resolver(self, tmp_path: Path):
        result = self._run(
            tmp_path, self._entry_needing_resolution(event_id="prov-ev-1")
        )
        assert result["status"] == "ok", result
        block = json.loads(Path(result["export_paths"][0]).read_text())
        assert block["trace"]["event_identity"]["provider_event_id"] == "prov-ev-1"

    def test_conflicting_caller_id_errors_the_entry(self, tmp_path: Path):
        result = self._run(
            tmp_path, self._entry_needing_resolution(event_id="some-other-event")
        )
        assert result["entries_error"] == 1
        assert "event_identity_mismatch" in result["errors"][0]["error"]


class TestRsvgIdentityAgreement:
    """Phase 0: the RSVG payload must describe the entry's own matchup."""

    def _roster_context(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "home_team": "Seattle Mariners",
            "away_team": "New York Mets",
            "league": "MLB",
            "game_date": "2026-06-03",
            "home_status": {
                "lineup_status": "confirmed",
                "injury_report_checked": True,
            },
            "away_status": {
                "lineup_status": "confirmed",
                "injury_report_checked": True,
            },
            "source_summaries": [
                {"source": "mlb.com", "summary": "Both lineups confirmed for tonight."}
            ],
            "roster_context_complete": True,
        }
        base.update(overrides)
        return base

    def _run(self, roster_context: dict[str, Any]) -> dict[str, Any]:
        import tempfile

        with (
            patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
            patch("omega.core.contracts.service.analyze", return_value=_make_trace()),
            patch(
                "omega.paths.repo_root",
                return_value=Path(tempfile.mkdtemp()),
            ),
        ):
            return omega_run_batch(
                entries=[_prop_entry(roster_context=roster_context)],
                bankroll=1000.0,
                session_id="sess-test",
            )

    def test_cross_league_context_is_rejected(self):
        result = self._run(self._roster_context(league="NBA"))
        assert result["entries_error"] == 1
        assert "rsvg_identity_mismatch" in result["errors"][0]["error"]
        assert "league" in result["errors"][0]["error"]

    def test_wrong_matchup_context_is_rejected(self):
        result = self._run(self._roster_context(home_team="New York Yankees"))
        assert result["entries_error"] == 1
        assert "rsvg_identity_mismatch" in result["errors"][0]["error"]

    def test_wrong_date_context_is_rejected(self):
        result = self._run(self._roster_context(game_date="2026-06-04"))
        assert result["entries_error"] == 1
        assert "game_date" in result["errors"][0]["error"]

    def test_matching_identity_passes_the_check(self):
        result = self._run(self._roster_context())
        assert result["status"] == "ok", result
        assert result["entries_ok"] == 1


class TestMatchupBriefTool:
    def _seed(self) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = TraceStore(db_path=tmp.name)
        with store.autolog_suppressed():
            store.persist(
                {
                    "trace_id": "sandbox-brief-1",
                    "run_id": "r-1",
                    "timestamp": "2026-07-16T00:00:00Z",
                    "kind": "game",
                    "prompt": "MLB game",
                    "league": "MLB",
                    "matchup": "Red Sox @ Yankees",
                    "execution_mode": "sandbox_game",
                    "aggregate_quality": 0.9,
                    "downgrades": [],
                    "input_snapshot": {
                        "league": "MLB",
                        "home_team": "Yankees",
                        "away_team": "Red Sox",
                    },
                    "result": {"status": "success", "edges": []},
                    "trace_quality": {"aggregate_quality": 0.9},
                }
            )
        store.close()
        return tmp.name

    def test_brief_by_trace_key(self):
        db = self._seed()
        result = omega_get_matchup_brief("trace:sandbox-brief-1", db_path=db)
        assert result["status"] == "success"
        brief = result["brief"]
        assert brief["identity_warning"] is True
        assert brief["markets"][0]["kind"] == "game"
        dumped = json.dumps(brief)
        for key in ("edge_pct", "ev_pct", "kelly_fraction", "confidence_tier", "best_bet"):
            assert f'"{key}"' not in dumped

    def test_not_found(self):
        db = self._seed()
        result = omega_get_matchup_brief("trace:missing", db_path=db)
        assert result["status"] == "error"
        assert result["error_code"] == "matchup_not_found"
