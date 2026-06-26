"""
Safe feature-combo evaluator (issue #28 WS3): validation, evaluation of the
boolean/threshold predicate + linear grammars, the whitelist guard, and the
generic handler that applies a graduated proposal (and stays unapplied in
probation).
"""

from __future__ import annotations

import warnings

import pytest

from omega.core.calibration.adjustment_policy import AdjustmentPolicyRegistry
from omega.core.contracts.evidence import EvidenceSignal
from omega.core.simulation.evidence_handlers import compute_player_adjustment
from omega.core.simulation.feature_combo_eval import (
    FeatureComboError,
    evaluate_feature_combo,
    validate_feature_combo,
)

_PRED = {
    "kind": "predicate",
    "when": {
        "op": "AND",
        "terms": [
            {"feature": "usage", "op": ">", "value": 0.30},
            {"feature": "teammate_injured", "op": "==", "value": True},
            {"feature": "opponent_scheme", "op": "==", "value": "ZONE"},
        ],
    },
    "true_factor": 1.06,
}


class TestValidate:
    def test_valid_predicate(self):
        validate_feature_combo(_PRED)  # no raise

    def test_valid_linear(self):
        validate_feature_combo(
            {"kind": "linear", "terms": [{"feature": "recent_residual", "weight": 0.1}], "bias": 1.0}
        )

    def test_off_whitelist_feature_rejected(self):
        with pytest.raises(FeatureComboError):
            validate_feature_combo(
                {"kind": "predicate", "when": {"feature": "secret", "op": ">", "value": 1}, "true_factor": 1.1}
            )

    def test_unknown_kind_rejected(self):
        with pytest.raises(FeatureComboError):
            validate_feature_combo({"kind": "exec", "code": "boom"})

    def test_unknown_operator_rejected(self):
        with pytest.raises(FeatureComboError):
            validate_feature_combo(
                {"kind": "predicate", "when": {"feature": "usage", "op": "~=", "value": 1}, "true_factor": 1.1}
            )

    def test_factor_out_of_bounds_rejected(self):
        with pytest.raises(FeatureComboError):
            validate_feature_combo({**_PRED, "true_factor": 5.0})

    def test_excessive_depth_rejected(self):
        node = {"feature": "usage", "op": ">", "value": 0.1}
        for _ in range(8):  # nest beyond _MAX_DEPTH
            node = {"op": "NOT", "term": node}
        with pytest.raises(FeatureComboError):
            validate_feature_combo({"kind": "predicate", "when": node, "true_factor": 1.1})


class TestEvaluatePredicate:
    def test_fires(self):
        f = evaluate_feature_combo(
            _PRED, {"usage": 0.35, "teammate_injured": True, "opponent_scheme": "zone"}
        )
        assert f == 1.06  # case-insensitive string match

    def test_does_not_fire(self):
        f = evaluate_feature_combo(
            _PRED, {"usage": 0.20, "teammate_injured": True, "opponent_scheme": "ZONE"}
        )
        assert f == 1.0  # default false_factor

    def test_missing_feature_is_safe_no_fire(self):
        assert evaluate_feature_combo(_PRED, {"usage": 0.35}) == 1.0

    def test_or_and_not(self):
        spec = {
            "kind": "predicate",
            "when": {
                "op": "OR",
                "terms": [
                    {"feature": "rest_days", "op": ">=", "value": 2},
                    {"op": "NOT", "term": {"feature": "b2b", "op": "==", "value": True}},
                ],
            },
            "true_factor": 1.03,
        }
        assert evaluate_feature_combo(spec, {"rest_days": 0, "b2b": False}) == 1.03  # NOT b2b
        assert evaluate_feature_combo(spec, {"rest_days": 3, "b2b": True}) == 1.03  # rest>=2
        assert evaluate_feature_combo(spec, {"rest_days": 0, "b2b": True}) == 1.0


class TestEvaluateLinear:
    def test_linear_sum(self):
        spec = {
            "kind": "linear",
            "terms": [
                {"feature": "recent_residual", "weight": 0.2},
                {"feature": "edge", "weight": 0.5},
            ],
            "bias": 1.0,
        }
        assert evaluate_feature_combo(spec, {"recent_residual": 0.5, "edge": 0.1}) == pytest.approx(
            1.0 + 0.2 * 0.5 + 0.5 * 0.1
        )

    def test_missing_feature_contributes_zero(self):
        spec = {"kind": "linear", "terms": [{"feature": "edge", "weight": 0.5}], "bias": 1.0}
        assert evaluate_feature_combo(spec, {}) == 1.0


_POLICY = AdjustmentPolicyRegistry().get_production_policy()


def _proposal_signal() -> EvidenceSignal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unknown signal_type warns by design
        return EvidenceSignal(
            signal_type="usage_when_star_out",
            category="player_form",
            plane="player",
            value={"usage": 0.35, "teammate_injured": True, "opponent_scheme": "ZONE"},
            source="llm",
            confidence=0.8,
            window="matchup",
            direction="over",
        )


class TestGenericHandlerApplication:
    def _adjust(self, policy, mode="bounded_live"):
        return compute_player_adjustment(
            player_context={"pts_mean": 25.0, "pts_std": 6.0},
            evidence=[_proposal_signal()],
            league="NBA",
            prop_type="pts",
            policy=policy,
            evidence_mode=mode,
        )

    def test_graduated_proposal_applies(self):
        # omega-promote-adjustment-policy binds a graduated proposal's coefficient
        # WITH a reliability_weight (it cleared the CLV bar): full trust (1.0) when
        # no scored CLV is on record, so it applies at its full magnitude — not the
        # unproven-prior sliver the handler would otherwise fall back to.
        pol = _POLICY.model_copy(
            update={
                "coefficients": {
                    **_POLICY.coefficients,
                    "usage_when_star_out": {
                        "feature_combo": _PRED,
                        "cap": 0.10,
                        "reliability_weight": 1.0,
                    },
                },
                "signal_lifecycle": {"usage_when_star_out": "active"},
            }
        ).bounded_live_effective()
        adj = self._adjust(pol)
        assert adj.records[0].applied is True
        # raw 1.06 at full weight (deviation 0.06 < the 0.10 bounded_live cap) -> 1.06.
        assert adj.mean_factor == pytest.approx(1.06)

    def test_unweighted_feature_combo_falls_back_to_prior(self):
        # A feature_combo coefficient with NO reliability_weight (e.g. a hand-bound
        # coeff, or the pre-fix promotion path) inherits the seed's conservative
        # unfitted prior (0.25): raw 1.06 -> 1 + 0.25*(0.06) = 1.015, a sliver. The
        # promote path now binds a weight so a genuinely graduated proposal applies
        # at full/measured magnitude instead (see test_graduated_proposal_applies).
        pol = _POLICY.model_copy(
            update={
                "coefficients": {
                    **_POLICY.coefficients,
                    "usage_when_star_out": {"feature_combo": _PRED, "cap": 0.10},
                },
                "signal_lifecycle": {"usage_when_star_out": "active"},
            }
        ).bounded_live_effective()
        adj = self._adjust(pol)
        assert adj.records[0].applied is True
        assert adj.mean_factor == pytest.approx(1.015)

    def test_probation_proposal_not_applied(self):
        # No coefficients for the proposal (the probation state) -> skip record.
        pol = _POLICY.bounded_live_effective()
        adj = self._adjust(pol)
        assert adj.mean_factor == 1.0
        assert adj.records[0].applied is False
