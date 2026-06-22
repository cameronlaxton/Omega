"""Tests for omega-cv-calibration-diagnostic pure CV logic.

Covers the novel logic this script adds — stratified folds, repeated-k-fold
cross-validation around the existing fitter, and the out-of-sample raw baseline.
The DB-loading plumbing reuses TraceStore (covered elsewhere) and is not retested.
"""

from __future__ import annotations

from omega.core.calibration.fitter import CalibrationFitter, stratified_folds
from omega.ops.cv_calibration_diagnostic import (
    cross_validate,
    raw_oos,
)


def _calibrated_dataset() -> tuple[list[float], list[int]]:
    """A well-calibrated set: each prediction bucket's empirical rate == bucket value."""
    preds: list[float] = []
    outs: list[int] = []
    for bucket in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        ones = round(bucket * 20)
        for _ in range(20):
            preds.append(bucket)
        outs.extend([1] * ones + [0] * (20 - ones))
    return preds, outs


def _miscalibrated_dataset() -> tuple[list[float], list[int]]:
    """Confident-but-wrong: predicts ~0.85 while true rate is ~0.40."""
    preds = [0.85] * 200
    outs = [1] * 80 + [0] * 120
    return preds, outs


def test_stratified_folds_partition_and_balance():
    outcomes = [1] * 40 + [0] * 160
    folds = stratified_folds(outcomes, k=5, seed=7)
    # every index used exactly once
    flat = sorted(i for f in folds for i in f)
    assert flat == list(range(len(outcomes)))
    # class balance preserved per fold (each fold ~8 positives, ~32 negatives)
    for f in folds:
        pos = sum(outcomes[i] for i in f)
        assert 6 <= pos <= 10


def test_stratified_folds_deterministic():
    outcomes = [1, 0] * 50
    assert stratified_folds(outcomes, 5, seed=99) == stratified_folds(outcomes, 5, seed=99)


def test_cross_validate_calibrated_low_ece():
    preds, outs = _calibrated_dataset()
    fitter = CalibrationFitter()
    res = cross_validate(
        fitter,
        preds,
        outs,
        "TEST",
        "game",
        "isotonic",
        folds=5,
        repeats=3,
        ece_floor=0.05,
        base_seed=1,
    )
    assert res.n_pairs == len(preds)
    assert res.n_folds_total == 15  # 5 folds x 3 repeats
    # A genuinely calibrated model should sit well under a loose bound out-of-sample.
    assert res.cal_ece_mean < 0.10
    assert 0.0 <= res.pass_rate <= 1.0
    lo, hi = res.cal_ece_ci95
    assert lo <= res.cal_ece_mean <= hi


def test_cross_validate_miscalibrated_high_ece():
    preds, outs = _miscalibrated_dataset()
    fitter = CalibrationFitter()
    res = cross_validate(
        fitter,
        preds,
        outs,
        "TEST",
        "game",
        "isotonic",
        folds=5,
        repeats=3,
        ece_floor=0.05,
        base_seed=1,
    )
    # Raw is confidently wrong; even after calibration the held-out signal is weak,
    # but the RAW out-of-sample ECE must clearly exceed the floor.
    raw_mean, raw_pass = raw_oos(preds, outs, folds=5, repeats=3, ece_floor=0.05, base_seed=1)
    assert raw_mean > 0.05
    assert raw_pass == 0.0
    import math

    assert math.isclose(res.raw_ece, raw_mean, rel_tol=1e-9)


def test_raw_oos_in_range():
    preds, outs = _calibrated_dataset()
    mean, pass_rate = raw_oos(preds, outs, folds=5, repeats=2, ece_floor=0.05, base_seed=3)
    assert 0.0 <= mean <= 1.0
    assert 0.0 <= pass_rate <= 1.0
