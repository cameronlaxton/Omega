"""
Agent proposal channel (issue #28 WS3): propose_signal validation + store
round-trip, and the omega_propose_signal MCP tool.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from omega.mcp.server import omega_propose_signal
from omega.ops.propose_signal import propose_signal
from omega.trace.store import TraceStore

_SPEC = {
    "kind": "predicate",
    "when": {"feature": "usage", "op": ">", "value": 0.30},
    "true_factor": 1.05,
}


def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


class TestProposeSignalValidation:
    def test_collision_with_builtin_rejected(self):
        with pytest.raises(ValueError):
            propose_signal(name="recent_form", feature_combo=_SPEC, persist=False)

    def test_bad_plane_rejected(self):
        with pytest.raises(ValueError):
            propose_signal(name="p1", feature_combo=_SPEC, plane="galaxy", persist=False)

    def test_bad_direction_rejected(self):
        with pytest.raises(ValueError):
            propose_signal(name="p1", feature_combo=_SPEC, direction_rule="sideways", persist=False)

    def test_off_whitelist_feature_rejected(self):
        from omega.core.simulation.feature_combo_eval import FeatureComboError

        bad = {"kind": "predicate", "when": {"feature": "x", "op": ">", "value": 1}, "true_factor": 1.1}
        with pytest.raises(FeatureComboError):
            propose_signal(name="p1", feature_combo=bad, persist=False)


class TestProposeSignalStore:
    def test_roundtrip_as_probation(self):
        db = _tmp_db()
        propose_signal(
            name="usage_when_star_out",
            feature_combo=_SPEC,
            thesis="Usage spikes when the star sits",
            plane="player",
            direction_rule="over",
            db_path=db,
        )
        store = TraceStore(db_path=db)
        try:
            props = store.get_signal_proposals()
            assert len(props) == 1
            assert props[0]["name"] == "usage_when_star_out"
            assert props[0]["lifecycle"] == "probation"  # always enters probation
            assert props[0]["feature_combo"] == _SPEC
        finally:
            store.close()


class TestProposeSignalMcpTool:
    def test_success(self):
        db = _tmp_db()
        res = omega_propose_signal(
            name="steam_fade",
            feature_combo={"kind": "linear", "terms": [{"feature": "edge", "weight": 0.2}]},
            plane="game",
            direction_rule="home",
            db_path=db,
        )
        assert res["status"] == "success"
        assert res["lifecycle"] == "probation"

    def test_invalid_spec_returns_error(self):
        res = omega_propose_signal(name="bad", feature_combo={"kind": "nope"})
        assert res["status"] == "error"
        assert res["error_code"] == "invalid_proposal"

    def test_collision_returns_error(self):
        res = omega_propose_signal(name="recent_form", feature_combo=_SPEC)
        assert res["status"] == "error"
        assert res["error_code"] == "invalid_proposal"
