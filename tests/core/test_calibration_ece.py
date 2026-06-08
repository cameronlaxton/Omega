"""Tests for the adaptive (equal-frequency) ECE estimator (#4).

Motivation: the promotion gate reads ``calibration_error`` from a profile's
holdout metrics. With ~20 holdout points and 10 fixed equal-width bins, most bins
held 0-2 samples and ECE was dominated by 1-2 singleton bins — high variance that
failed genuinely well-calibrated profiles. The adaptive estimator uses
quantile bins whose count scales with n and never splits tied predictions.
"""

from __future__ import annotations

import random

import pytest

from omega.core.calibration.fitter import (
    ECE_MAX_BINS,
    ECE_MIN_PER_BIN,
    _adaptive_calibration_error,
)


def _fixed_bin_ece(preds, outs, n_bins=10):
    """Frozen copy of the previous equal-width ECE, for comparison only."""
    n = len(preds)
    if n == 0:
        return 0.0
    bc = [0] * n_bins
    bp = [0.0] * n_bins
    bo = [0.0] * n_bins
    for p, o in zip(preds, outs):
        i = min(int(p * n_bins), n_bins - 1)
        bc[i] += 1
        bp[i] += p
        bo[i] += o
    e = 0.0
    for i in range(n_bins):
        if bc[i] > 0:
            e += abs(bp[i] / bc[i] - bo[i] / bc[i]) * bc[i] / n
    return e


def test_empty_returns_zero():
    assert _adaptive_calibration_error([], []) == 0.0


def test_perfectly_calibrated_is_zero():
    # One predicted value, realized rate equals it.
    preds = [0.5] * 20
    outs = [1, 0] * 10
    assert _adaptive_calibration_error(preds, outs) == pytest.approx(0.0, abs=1e-12)


def test_gross_miscalibration_is_detected():
    # Predicts 0.9 everywhere but realizes 0.5 -> ECE == 0.4.
    preds = [0.9] * 20
    outs = [1, 0] * 10
    assert _adaptive_calibration_error(preds, outs) == pytest.approx(0.4, abs=1e-12)


def test_in_unit_interval():
    preds = [0.1, 0.9, 0.5, 0.3, 0.7, 0.6, 0.4, 0.8, 0.55, 0.62, 0.66, 0.71]
    outs = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    assert 0.0 <= _adaptive_calibration_error(preds, outs) <= 1.0


def test_order_independent_even_with_tied_predictions():
    # Tied values (0.52, 0.66) must not be split/sorted by outcome.
    preds = [0.52] * 9 + [0.66] * 9 + [0.36, 0.82]
    outs = [1, 1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 0, 0, 0] + [0, 0]
    base = _adaptive_calibration_error(preds, outs)
    for seed in range(6):
        idx = list(range(len(preds)))
        random.Random(seed).shuffle(idx)
        sp = [preds[i] for i in idx]
        so = [outs[i] for i in idx]
        assert _adaptive_calibration_error(sp, so) == pytest.approx(base)


def test_tied_predictions_not_split_by_outcome():
    # All predictions identical; a naive sort-by-outcome would put every 0 in one
    # bin and every 1 in another, inventing ~0.5 ECE. Correct answer: |0.6-0.5|.
    preds = [0.6] * 40
    outs = [1] * 20 + [0] * 20
    assert _adaptive_calibration_error(preds, outs) == pytest.approx(0.1, abs=1e-12)


def test_reduces_singleton_bin_variance_vs_fixed_width():
    # 18 well-calibrated mid predictions + 2 'unlucky' extreme singletons. With
    # fixed equal-width bins the singletons sit alone and dominate ECE; quantile
    # bins absorb them into populated neighbors.
    preds = [0.52] * 9 + [0.66] * 9 + [0.36, 0.82]
    outs = [1, 1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 0, 0, 0] + [0, 0]
    fixed = _fixed_bin_ece(preds, outs, n_bins=10)
    adaptive = _adaptive_calibration_error(preds, outs)
    assert adaptive < fixed
    assert adaptive < 0.05  # clears the gate floor; fixed-width (~0.078) would not


def test_small_n_collapses_to_single_bin():
    # n below min_per_bin -> 1 bin -> calibration-in-the-large only.
    preds = [0.6, 0.7, 0.8]
    outs = [1, 1, 0]
    expected = abs(sum(preds) / 3 - sum(outs) / 3)
    assert _adaptive_calibration_error(preds, outs) == pytest.approx(expected)


def test_bin_count_scales_with_sample_size():
    # Distinct values so each could occupy its own bin; the number of *populated*
    # bins is capped by n // min_per_bin. With 100 distinct points -> up to 10.
    preds = [i / 100 for i in range(100)]
    outs = [1 if i % 2 == 0 else 0 for i in range(100)]
    # Should run without error and stay bounded; resolution is full at n=100.
    val = _adaptive_calibration_error(preds, outs)
    assert 0.0 <= val <= 1.0
    assert ECE_MIN_PER_BIN == 10
    assert ECE_MAX_BINS == 10
