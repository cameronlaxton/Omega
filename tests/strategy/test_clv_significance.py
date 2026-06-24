"""
Tests for the CLV probation statistics (issue #28 Dual-Gate + FDR framework).

Covers the three gates (bootstrap lower bound, Benjamini–Hochberg FDR, power
N_min), the p-value helpers, deterministic seeding, and the composed
graduation decision.
"""

from __future__ import annotations

import pytest

from omega.strategy.clv_significance import (
    ProbationStats,
    alignment_pvalue,
    benjamini_hochberg,
    bootstrap_clv_lower_bound,
    clv_pvalue,
    compute_probation_stats,
    graduation_mask,
    min_samples_for_power,
    seed_from_dataset_hash,
)


class TestSeed:
    def test_deterministic(self):
        assert seed_from_dataset_hash("abc123") == seed_from_dataset_hash("abc123")

    def test_salt_changes_seed(self):
        assert seed_from_dataset_hash("abc123", "recent_form") != seed_from_dataset_hash(
            "abc123", "stale_line"
        )

    def test_non_negative(self):
        assert seed_from_dataset_hash("") >= 0
        assert seed_from_dataset_hash("x" * 100, "y" * 100) >= 0


class TestBootstrap:
    def test_empty_is_neg_inf(self):
        assert bootstrap_clv_lower_bound([], seed=1) == float("-inf")

    def test_single_value(self):
        assert bootstrap_clv_lower_bound([4.2], seed=1) == 4.2

    def test_deterministic_same_seed(self):
        vals = [1.0, -2.0, 3.0, 0.5, -1.5, 2.2, 0.1]
        a = bootstrap_clv_lower_bound(vals, seed=42)
        b = bootstrap_clv_lower_bound(vals, seed=42)
        assert a == b

    def test_clearly_positive_clears_gate(self):
        lb = bootstrap_clv_lower_bound([3.0, 4.0, 5.0, 6.0, 7.0], seed=7)
        assert lb > 0.0

    def test_centered_at_zero_fails_gate(self):
        lb = bootstrap_clv_lower_bound([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0], seed=7)
        assert lb <= 0.0


class TestBenjaminiHochberg:
    def test_empty(self):
        assert benjamini_hochberg([]) == []

    def test_all_reject_when_all_significant(self):
        # p_(k) <= (k/m)*q for every k.
        mask = benjamini_hochberg([0.001, 0.002, 0.003, 0.004], q=0.05)
        assert mask == [True, True, True, True]

    def test_only_smallest_rejected(self):
        mask = benjamini_hochberg([0.001, 0.5, 0.6, 0.7, 0.8], q=0.05)
        assert mask == [True, False, False, False, False]

    def test_none_rejected_when_all_large(self):
        assert benjamini_hochberg([0.4, 0.5, 0.6], q=0.05) == [False, False, False]

    def test_alignment_to_input_order(self):
        # Same multiset, shuffled — the True must follow the small p-value.
        mask = benjamini_hochberg([0.8, 0.001, 0.6], q=0.05)
        assert mask == [False, True, False]


class TestPower:
    def test_default_in_plausible_range(self):
        n = min_samples_for_power()  # edge=0.02, power=0.80, alpha=0.05
        assert 3000 < n < 5000

    def test_larger_edge_needs_fewer_samples(self):
        assert min_samples_for_power(edge=0.05) < min_samples_for_power(edge=0.02)

    def test_higher_power_needs_more_samples(self):
        assert min_samples_for_power(power=0.90) > min_samples_for_power(power=0.80)

    def test_non_positive_edge_raises(self):
        with pytest.raises(ValueError):
            min_samples_for_power(edge=0.0)


class TestClvPvalue:
    def test_too_few_samples(self):
        assert clv_pvalue([1.0]) == 1.0

    def test_strongly_positive_is_significant(self):
        assert clv_pvalue([3.0, 4.0, 5.0, 6.0, 7.0]) < 0.05

    def test_centered_is_not_significant(self):
        assert clv_pvalue([-2.0, -1.0, 0.0, 1.0, 2.0]) > 0.4

    def test_zero_variance_positive(self):
        assert clv_pvalue([2.0, 2.0, 2.0]) == 0.0

    def test_zero_variance_nonpositive(self):
        assert clv_pvalue([0.0, 0.0, 0.0]) == 1.0


class TestAlignmentPvalue:
    def test_no_samples(self):
        assert alignment_pvalue(0, 0) == 1.0

    def test_high_rate_significant(self):
        assert alignment_pvalue(90, 100) < 0.05

    def test_coin_flip_not_significant(self):
        assert alignment_pvalue(50, 100) > 0.4


class TestGraduationMask:
    def test_empty(self):
        assert graduation_mask({}) == {}

    def test_ineligible_keys_never_graduate(self):
        stats = {
            "low_n": ProbationStats(
                n=3, n_min=10, clv_mean=5.0, boot_lower_bound=2.0,
                pvalue=0.0001, meets_n_min=False, boot_positive=True,
            ),
            "boot_neg": ProbationStats(
                n=50, n_min=10, clv_mean=0.1, boot_lower_bound=-1.0,
                pvalue=0.0001, meets_n_min=True, boot_positive=False,
            ),
        }
        mask = graduation_mask(stats)
        assert mask == {"low_n": False, "boot_neg": False}

    def test_fdr_applied_among_eligible(self):
        # Five eligible candidates; BH at q=0.10 (m=5) admits the two smallest.
        def mk(p: float) -> ProbationStats:
            return ProbationStats(
                n=50, n_min=10, clv_mean=1.0, boot_lower_bound=0.5,
                pvalue=p, meets_n_min=True, boot_positive=True,
            )

        stats = {
            "a": mk(0.001),
            "b": mk(0.04),
            "c": mk(0.5),
            "d": mk(0.6),
            "e": mk(0.7),
        }
        mask = graduation_mask(stats, q=0.10)
        assert mask == {"a": True, "b": True, "c": False, "d": False, "e": False}

    def test_real_signal_graduates_end_to_end(self):
        real = compute_probation_stats(
            [3.0, 4.0, 5.0, 6.0, 7.0] * 4, n_min=10, seed=seed_from_dataset_hash("d", "real")
        )
        noise = compute_probation_stats(
            [-3.0, -1.0, 1.0, 3.0] * 5, n_min=10, seed=seed_from_dataset_hash("d", "noise")
        )
        mask = graduation_mask({"real": real, "noise": noise})
        assert mask["real"] is True
        assert mask["noise"] is False


class TestPooledStats:
    def test_pooled_matches_concatenation(self):
        import statistics

        from omega.strategy.clv_significance import pooled_mean_std

        a = [1.0, 2.0, 3.0, 4.0]
        b = [5.0, 6.0, 7.0, 8.0, 9.0]
        groups = [
            (len(a), statistics.fmean(a), statistics.stdev(a)),
            (len(b), statistics.fmean(b), statistics.stdev(b)),
        ]
        n, mean, std = pooled_mean_std(groups)
        assert n == 9
        assert abs(mean - statistics.fmean(a + b)) < 1e-9
        assert abs(std - statistics.stdev(a + b)) < 1e-9

    def test_pooled_empty(self):
        from omega.strategy.clv_significance import pooled_mean_std

        assert pooled_mean_std([]) == (0, 0.0, 0.0)

    def test_normal_lower_bound_below_mean(self):
        from omega.strategy.clv_significance import normal_lower_bound

        lb = normal_lower_bound(mean=10.0, std=2.0, n=100)
        assert abs(lb - (10.0 - 1.6448536 * 0.2)) < 1e-4

    def test_normal_lower_bound_zero_std_is_mean(self):
        from omega.strategy.clv_significance import normal_lower_bound

        assert normal_lower_bound(mean=3.0, std=0.0, n=50) == 3.0

    def test_pvalue_from_stats(self):
        from omega.strategy.clv_significance import clv_pvalue_from_stats

        assert clv_pvalue_from_stats(2.0, 1.0, 100) < 0.001  # strongly positive
        assert abs(clv_pvalue_from_stats(0.0, 1.0, 100) - 0.5) < 1e-6  # centered
        assert clv_pvalue_from_stats(-2.0, 1.0, 100) > 0.999  # negative
