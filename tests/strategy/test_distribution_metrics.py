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
