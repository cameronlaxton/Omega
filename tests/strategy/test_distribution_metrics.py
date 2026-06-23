from __future__ import annotations

import json

import pytest

from omega.strategy.distribution_metrics import (
    METRIC_VERSION,
    crps_from_distribution_row,
    crps_normal,
    crps_poisson,
)


def test_crps_normal_fixed_fixture():
    assert crps_normal(mu=0.0, sigma=1.0, observed=0.0) == pytest.approx(
        0.233695,
        abs=1e-6,
    )


def test_crps_poisson_is_lower_near_distribution_center():
    assert crps_poisson(lam=5.0, observed=5.0) < crps_poisson(lam=5.0, observed=12.0)


def test_crps_from_distribution_row_parses_json_params():
    row = {
        "distribution_type": "normal",
        "distribution_params": json.dumps({"mu": 10.0, "sigma": 2.5}),
    }

    metric = crps_from_distribution_row(row, observed=11.0)

    assert metric["metric_version"] == METRIC_VERSION
    assert metric["metric"] == "crps"
    assert metric["distribution_type"] == "normal"
    assert metric["value"] >= 0.0


def test_crps_empirical_markov_uses_normal_approximation():
    # empirical_markov uses sample_mean/sample_std for Normal-moment CRPS.
    # Verify it returns a finite non-negative value and matches the equivalent
    # direct Normal call.
    row = {
        "distribution_type": "empirical_markov",
        "distribution_params": {"source": "markov_terminal_scores"},
        "sample_mean": 108.5,
        "sample_std": 9.2,
    }
    observed = 115.0
    metric = crps_from_distribution_row(row, observed=observed)

    assert metric["metric_version"] == METRIC_VERSION
    assert metric["metric"] == "crps"
    assert metric["distribution_type"] == "empirical_markov"
    assert metric["value"] >= 0.0

    # Must match closed-form normal CRPS with the same moments
    from omega.strategy.distribution_metrics import crps_normal

    expected = round(crps_normal(108.5, 9.2, observed), 6)
    assert metric["value"] == pytest.approx(expected, abs=1e-9)


def test_crps_empirical_fast_score_uses_normal_approximation():
    """fast_score backend emits distribution_type='empirical'; must also work."""
    row = {
        "distribution_type": "empirical",
        "distribution_params": {"source": "monte_carlo_scores"},
        "sample_mean": 112.0,
        "sample_std": 8.5,
    }
    metric = crps_from_distribution_row(row, observed=110.0)
    assert metric["distribution_type"] == "empirical"
    assert metric["value"] >= 0.0


def test_crps_empirical_markov_unknown_type_still_raises():
    row = {"distribution_type": "future_unknown_type", "distribution_params": {}}
    with pytest.raises(ValueError, match="CRPS is not implemented"):
        crps_from_distribution_row(row, observed=5.0)
