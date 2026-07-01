"""Per-signal marginal value via injected counterfactual (issue #28 WS1).

The counterfactual is injected, so the collection + aggregation are tested exactly
without invoking the engine; the engine binding is tested only for its defensive
contract (never raises on a malformed trace).
"""

from __future__ import annotations

from omega.historical.marginal_value import (
    aggregate_marginal_values,
    collect_signal_pairs,
    engine_counterfactual_prob,
)


def _trace(home_win_prob, result, applied_signals):
    return {
        "predictions": {"home_win_prob": home_win_prob},
        "_outcome": {"result": result},
        "evidence_application": [{"signal_type": s, "applied": True} for s in applied_signals],
    }


def _home(t):
    p = t["predictions"]["home_win_prob"]
    return p / 100.0 if p > 1 else p


def test_collect_signal_pairs_only_applied_and_graded():
    traces = [
        _trace(60, "home_win", ["recent_form_residual"]),
        _trace(55, "away_win", ["recent_form_residual"]),
        # ungraded result → skipped
        {
            "predictions": {"home_win_prob": 70},
            "_outcome": {"result": "scheduled"},
            "evidence_application": [{"signal_type": "recent_form_residual", "applied": True}],
        },
        # a signal that was NOT applied → skipped
        {
            "predictions": {"home_win_prob": 50},
            "_outcome": {"result": "home_win"},
            "evidence_application": [{"signal_type": "noise", "applied": False}],
        },
    ]
    pairs = collect_signal_pairs(traces, lambda t, s: _home(t) - 0.10)
    assert set(pairs) == {"recent_form_residual"}
    with_p, without_p, outs = pairs["recent_form_residual"]
    assert len(with_p) == 2
    assert outs == [1, 0]  # home_win → 1, away_win → 0


def test_aggregate_rewards_helpful_signal():
    # WITH the signal the forecast is sharp toward the realized result; the
    # counterfactual (signal removed) is a coin flip → the signal lowered Brier.
    traces = [
        _trace(80, "home_win", ["good"]),
        _trace(20, "away_win", ["good"]),
    ]
    blocks = aggregate_marginal_values(traces, lambda t, s: 0.5)
    assert len(blocks) == 1
    assert blocks[0].signal_type == "good"
    assert blocks[0].brier_delta > 0  # improves (lowers) Brier
    assert blocks[0].n == 2


def test_aggregate_empty_when_no_applied_evidence():
    traces = [_trace(60, "home_win", [])]
    assert aggregate_marginal_values(traces, lambda t, s: 0.5) == []


def test_counterfactual_none_skips_that_signal_only():
    traces = [_trace(60, "home_win", ["a", "b"])]
    blocks = aggregate_marginal_values(traces, lambda t, s: None if s == "b" else 0.5)
    assert [b.signal_type for b in blocks] == ["a"]


def test_engine_counterfactual_prob_is_defensive():
    # A malformed trace must return None, never raise (keeps a lab run alive).
    assert engine_counterfactual_prob({}, "x") is None
    assert engine_counterfactual_prob({"input_snapshot": {}}, "x") is None
