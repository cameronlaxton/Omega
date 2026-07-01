"""Batch-level aggregate_quality heuristic in omega-validate-trace-export.

Repo-visible symptom detection only: it flags the fingerprint of a hardcoded
quality value (every trace in a batch identical) without claiming to police the
external scratch script that may have produced it.
"""

from __future__ import annotations

from omega.ops.validate_trace_export import (
    _extract_aggregate_quality,
    _uniform_quality_warning,
)


def _trace(agg, *, key="trace_quality"):
    return {"trace_id": "t", "kind": "game", key: {"aggregate_quality": agg}}


def test_extract_from_trace_quality():
    assert _extract_aggregate_quality(_trace(0.82)) == 0.82


def test_extract_tolerates_pre_rename_quality_gate_key():
    assert _extract_aggregate_quality(_trace(0.5, key="quality_gate")) == 0.5


def test_extract_handles_wrapped_shape():
    wrapped = {"trace": {"trace_id": "t", "kind": "game", "trace_quality": {"aggregate_quality": 1.0}}}
    assert _extract_aggregate_quality(wrapped) == 1.0


def test_extract_missing_or_nonnumeric_returns_none():
    assert _extract_aggregate_quality({"trace_id": "t", "kind": "game"}) is None
    assert _extract_aggregate_quality(_trace("high")) is None
    assert _extract_aggregate_quality("not a dict") is None


def test_uniform_warning_fires_when_all_identical():
    warn = _uniform_quality_warning([1.0, 1.0, 1.0, 1.0, 1.0])
    assert warn is not None
    assert "identical" in warn and "aggregate_quality=1" in warn


def test_uniform_warning_silent_when_values_vary():
    assert _uniform_quality_warning([1.0, 0.9, 0.8, 0.7, 0.6]) is None
    # Even a single outlier suppresses the all-identical signal.
    assert _uniform_quality_warning([1.0, 1.0, 1.0, 1.0, 0.75]) is None


def test_uniform_warning_needs_minimum_sample():
    # A tiny batch that happens to match is not enough evidence to warn.
    assert _uniform_quality_warning([1.0, 1.0, 1.0]) is None
