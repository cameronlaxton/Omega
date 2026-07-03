"""Static-fallback calibration policy: monotonicity and continuity.

The v1 static policy (gate on raw ∉ [0.10, 0.90], then shrink 0.7 + cap)
was non-monotone at both activation boundaries: raw 0.905 calibrated to
~0.784 while raw 0.890 passed through untouched, so a *higher* raw
probability produced a *lower* calibrated probability and edge/EV/tier
rankings could invert between two candidates. The v2 policy
(``static_tail_compress``) is a monotone, continuous tail compression —
these tests pin that property so a regression to gate-then-shrink fails
loudly.

No profile is registered in any of these tests, so ``apply_calibration``
resolves to the static path (league=None skips profile lookup entirely).
"""

from __future__ import annotations

import pytest

from omega.core.calibration.probability import (
    _POLICY_TAIL_HI,
    _POLICY_TAIL_LO,
    apply_calibration,
    apply_calibration_audited,
    static_tail_compress,
)


class TestStaticTailCompress:
    def test_identity_in_mid_range(self):
        for p in (0.10, 0.25, 0.50, 0.65, 0.85, 0.90):
            assert static_tail_compress(p) == p

    def test_monotone_non_decreasing_over_full_range(self):
        # Dense grid including the v1 failure points around both boundaries.
        grid = [i / 1000.0 for i in range(1001)]
        values = [static_tail_compress(p) for p in grid]
        for lower, higher in zip(values, values[1:]):
            assert higher >= lower

    def test_strictly_increasing_across_v1_failure_points(self):
        # v1: 0.890 → 0.890 but 0.905 → ~0.784 (inversion). v2 must order these.
        upper = [0.88, 0.89, 0.90, 0.905, 0.92, 0.95, 0.99, 1.0]
        cal = [static_tail_compress(p) for p in upper]
        assert cal == sorted(cal)
        assert len(set(cal)) == len(cal)  # strictly increasing

        # v1: 0.110 → 0.110 but 0.095 → ~0.217 (inversion, low side).
        lower = [0.0, 0.01, 0.05, 0.095, 0.10, 0.11, 0.12]
        cal_lo = [static_tail_compress(p) for p in lower]
        assert cal_lo == sorted(cal_lo)
        assert len(set(cal_lo)) == len(cal_lo)

    def test_continuous_at_activation_boundaries(self):
        eps = 1e-9
        assert static_tail_compress(_POLICY_TAIL_HI + eps) == pytest.approx(
            _POLICY_TAIL_HI, abs=1e-6
        )
        assert static_tail_compress(_POLICY_TAIL_LO - eps) == pytest.approx(
            _POLICY_TAIL_LO, abs=1e-6
        )

    def test_softens_extremes_and_stays_in_range(self):
        assert static_tail_compress(1.0) < 1.0
        assert static_tail_compress(0.0) > 0.0
        assert static_tail_compress(0.95) < 0.95
        assert static_tail_compress(0.05) > 0.05
        for p in (0.0, 0.03, 0.97, 1.0):
            assert 0.0 < static_tail_compress(p) < 1.0

    def test_clips_out_of_range_input(self):
        assert static_tail_compress(1.5) == static_tail_compress(1.0)
        assert static_tail_compress(-0.5) == static_tail_compress(0.0)


class TestStaticPathThroughApplyCalibration:
    def test_apply_calibration_static_path_is_monotone(self):
        grid = [i / 200.0 for i in range(201)]
        values = [apply_calibration(p) for p in grid]
        for lower, higher in zip(values, values[1:]):
            assert higher >= lower

    def test_audit_paths_still_distinguish_identity_from_calibrated(self):
        cal_mid, audit_mid = apply_calibration_audited(0.65)
        assert cal_mid == 0.65
        assert audit_mid["path"] == "static_identity"
        assert audit_mid["method_resolved"] is None

        cal_tail, audit_tail = apply_calibration_audited(0.95)
        assert cal_tail < 0.95
        assert audit_tail["path"] == "static_calibrated"
        assert audit_tail["method_resolved"] == "combined_v2"
        assert audit_tail["raw_prob"] == 0.95
        assert audit_tail["calibrated_prob"] == cal_tail
