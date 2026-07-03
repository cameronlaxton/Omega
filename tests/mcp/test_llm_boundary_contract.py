"""Contract tests for the LLM ⊥ engine boundary at the MCP seam.

Protected engine-owned values (edge/EV/Kelly/units/confidence tier/
probabilities) must be rejected wherever an LLM-facing payload could smuggle
them in — the caller-supplied ``trace_quality`` on the analyze tools is the one
open dict that flows into the persisted quality block.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from omega.core.contracts.protected_fields import PROTECTED_QUANT_FIELDS, find_protected_key
from omega.mcp.server import omega_analyze_game, omega_analyze_prop, omega_analyze_slate


def _game_request() -> dict[str, Any]:
    return {
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "league": "NBA",
        "home_context": {"off_rating": 119.2, "def_rating": 108.1, "pace": 96.5},
        "away_context": {"off_rating": 115.8, "def_rating": 110.3, "pace": 94.1},
        "game_context": {"is_playoff": False, "rest_days": 2},
    }


class TestTraceQualityProtectedGuard:
    def test_analyze_game_rejects_protected_trace_quality(self) -> None:
        with patch("omega.core.contracts.service.analyze") as analyze:
            result = omega_analyze_game(
                request=_game_request(),
                bankroll=1000.0,
                session_id="sess-test",
                trace_quality={"aggregate_quality": 0.8, "edge_pct": 4.2},
            )
        assert result["status"] == "error"
        assert result["error_code"] == "invalid_request"
        assert "edge_pct" in str(result["detail"])
        analyze.assert_not_called()

    def test_analyze_prop_rejects_nested_protected_key(self) -> None:
        with patch("omega.core.contracts.service.analyze") as analyze:
            result = omega_analyze_prop(
                request={},
                bankroll=1000.0,
                session_id="sess-test",
                trace_quality={"notes": [{"kelly_fraction": 0.05}]},
            )
        assert result["status"] == "error"
        assert result["error_code"] == "invalid_request"
        assert "kelly_fraction" in str(result["detail"])
        analyze.assert_not_called()

    def test_analyze_slate_rejects_protected_trace_quality(self) -> None:
        with patch("omega.core.contracts.service.analyze") as analyze:
            result = omega_analyze_slate(
                request={"league": "NBA"},
                bankroll=1000.0,
                session_id="sess-test",
                trace_quality={"confidence_tier": "A"},
            )
        assert result["status"] == "error"
        assert result["error_code"] == "invalid_request"
        analyze.assert_not_called()

    def test_clean_quality_metadata_is_accepted(self) -> None:
        """aggregate_quality / downgrades are quality metadata, not protected."""
        trace = {"trace_id": "sandbox-clean", "result": {}, "trace_quality": {}}
        with (
            patch("omega.mcp.server._formal_output_gate_failures", return_value=[]),
            patch("omega.core.contracts.service.analyze", return_value=trace) as analyze,
        ):
            result = omega_analyze_game(
                request=_game_request(),
                bankroll=1000.0,
                session_id="sess-test",
                trace_quality={"aggregate_quality": 0.74, "downgrades": ["thin_sample"]},
            )
        assert result["status"] == "success"
        analyze.assert_called_once()

    @pytest.mark.parametrize("field", sorted(PROTECTED_QUANT_FIELDS))
    def test_every_canonical_protected_field_is_rejected(self, field: str) -> None:
        with patch("omega.core.contracts.service.analyze") as analyze:
            result = omega_analyze_game(
                request=_game_request(),
                bankroll=1000.0,
                session_id="sess-test",
                trace_quality={field: 1.0},
            )
        assert result["status"] == "error"
        analyze.assert_not_called()


class TestProtectedFieldsCanonicalSet:
    def test_core_set_matches_sidecar_set(self) -> None:
        """One definition of 'protected': the sidecar's enforcement set and the
        core contracts set must never drift apart (core cannot import trace, so
        the sync is enforced here)."""
        from omega.trace.session_sidecar import _PROTECTED_QUANT_FIELDS as sidecar_fields

        assert sidecar_fields == PROTECTED_QUANT_FIELDS

    def test_find_protected_key_scans_keys_not_values(self) -> None:
        # Prose VALUES may mention a field name; only KEYS are structural.
        assert find_protected_key({"notes": "Null fields: result.edge_pct"}) is None
        assert find_protected_key({"a": [{"b": {"ev_pct": 3.0}}]}) == "ev_pct"
        assert find_protected_key(None) is None
