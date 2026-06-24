"""
Signal lifecycle (issue #28 WS3): resolver helpers, the application gate
(probation/deprecated never move a live prediction), and lifecycle-filtered
agent vocabulary.
"""

from __future__ import annotations

import warnings

from omega.core.calibration.adjustment_policy import AdjustmentPolicyRegistry
from omega.core.contracts.evidence import (
    EvidenceSignal,
    declared_lifecycle,
    effective_lifecycle,
    is_applicable_lifecycle,
    is_vocabulary_visible,
)
from omega.core.simulation.evidence_handlers import compute_player_adjustment
from omega.core.simulation.evidence_to_modifier import build_markov_vocabulary_table

_POLICY = AdjustmentPolicyRegistry().get_production_policy()


class TestLifecycleResolver:
    def test_declared_defaults_active(self):
        assert declared_lifecycle("recent_form") == "active"
        assert declared_lifecycle("unknown_signal_xyz") == "active"

    def test_override_takes_precedence(self):
        assert effective_lifecycle("recent_form", {"recent_form": "deprecated"}) == "deprecated"

    def test_invalid_override_ignored(self):
        # A malformed value must never silently un-gate; fall back to declared.
        assert effective_lifecycle("recent_form", {"recent_form": "bogus"}) == "active"

    def test_no_overrides(self):
        assert effective_lifecycle("recent_form", None) == "active"

    def test_vocabulary_visibility(self):
        assert is_vocabulary_visible("active") is True
        assert is_vocabulary_visible("probation") is True
        assert is_vocabulary_visible("deprecated") is False
        assert is_vocabulary_visible("rejected") is False

    def test_applicability(self):
        assert is_applicable_lifecycle("active") is True
        for lc in ("probation", "deprecated", "rejected"):
            assert is_applicable_lifecycle(lc) is False


def _usage_signal() -> EvidenceSignal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvidenceSignal(
            signal_type="usage_spike",
            category="player_form",
            plane="player",
            value=0.5,
            source="agent_reasoning",
            confidence=0.9,
            window="matchup",
        )


def _adjust_bounded_live(overrides: dict[str, str] | None = None):
    pol = _POLICY
    if overrides is not None:
        pol = pol.model_copy(update={"signal_lifecycle": overrides})
    pol = pol.bounded_live_effective()
    return compute_player_adjustment(
        player_context={"pts_mean": 25.0, "pts_std": 6.0},
        evidence=[_usage_signal()],
        league="NBA",
        prop_type="pts",
        policy=pol,
        evidence_mode="bounded_live",
    )


class TestApplicationGate:
    def test_active_signal_applies_in_bounded_live(self):
        # Control: with no override usage_spike is active and moves the math.
        adj = _adjust_bounded_live(None)
        assert adj.mean_factor != 1.0
        assert adj.records[0].applied is True

    def test_probation_override_suppresses_application(self):
        adj = _adjust_bounded_live({"usage_spike": "probation"})
        assert adj.mean_factor == 1.0  # NEVER applied, even in bounded_live
        rec = adj.records[0]
        assert rec.applied is False
        assert rec.factor != 1.0  # still evaluated + recorded for CLV scoring/audit
        assert "lifecycle=probation" in rec.reason

    def test_deprecated_override_suppresses_application(self):
        adj = _adjust_bounded_live({"usage_spike": "deprecated"})
        assert adj.mean_factor == 1.0
        assert adj.records[0].applied is False


class TestLifecycleFilteredVocabulary:
    @staticmethod
    def _vocab_rows(table: str) -> dict[str, str]:
        """Map signal_type -> its vocab ROW line (excludes the static footer text)."""
        rows: dict[str, str] = {}
        for ln in table.splitlines():
            s = ln.strip()
            for sig in ("pace_up", "pace_down", "rest_advantage", "b2b_fatigue"):
                if s.startswith(sig + " "):
                    rows[sig] = ln
        return rows

    def test_active_signals_present_by_default(self):
        rows = self._vocab_rows(build_markov_vocabulary_table())
        assert "pace_up" in rows
        assert "rest_advantage" in rows

    def test_deprecated_dropped(self):
        rows = self._vocab_rows(build_markov_vocabulary_table({"rest_advantage": "deprecated"}))
        assert "rest_advantage" not in rows  # the vocab row is gone
        assert "pace_up" in rows  # untouched

    def test_probation_flagged_not_dropped(self):
        rows = self._vocab_rows(build_markov_vocabulary_table({"pace_up": "probation"}))
        assert "pace_up" in rows
        assert "probation" in rows["pace_up"]
