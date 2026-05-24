"""Dynamic scoring helpers for persisted simulation distributions.

Metrics are intentionally computed from raw V10 distribution rows and realized
outcomes at report time. They are not stored as transactional truth because the
metric implementations can evolve independently of the ledger schema.
"""

from __future__ import annotations

import json
import math
from typing import Any

METRIC_VERSION = "distribution_metrics_v1"


def crps_normal(mu: float, sigma: float, observed: float) -> float:
    """Closed-form CRPS for a normal distribution."""
    sigma = max(float(sigma), 1e-9)
    z = (float(observed) - float(mu)) / sigma
    phi = math.exp(-(z**2) / 2.0) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def crps_poisson(lam: float, observed: float) -> float:
    """Numerically compute CRPS for a Poisson distribution by CDF summation."""
    lam = max(float(lam), 1e-9)
    y = int(math.floor(float(observed)))
    upper = max(y + 50, int(lam + 10.0 * math.sqrt(lam) + 10.0))
    pmf = math.exp(-lam)
    cdf = 0.0
    total = 0.0
    for k in range(upper + 1):
        if k > 0:
            pmf *= lam / k
        cdf += pmf
        indicator = 1.0 if k >= y else 0.0
        total += (cdf - indicator) ** 2
        if k > y and 1.0 - cdf < 1e-10:
            break
    return total


def crps_from_distribution_row(row: dict[str, Any], observed: float) -> dict[str, Any]:
    """Return a versioned CRPS payload for one V10 distribution row."""
    dist = str(row.get("distribution_type") or "").lower()
    params = row.get("distribution_params") or {}
    if isinstance(params, str):
        params = json.loads(params)
    if dist == "normal":
        value = crps_normal(params["mu"], params["sigma"], observed)
    elif dist == "poisson":
        value = crps_poisson(params["lambda"], observed)
    elif dist in ("empirical", "empirical_markov"):
        # Normal-moment approximation using stored sample_mean/sample_std.
        # Valid when N is large enough for CLT (both fast_score and markov backends
        # default to ≥100 iterations). "empirical" is fast_score monte-carlo;
        # "empirical_markov" is the Markov possession simulator.
        mean = float(row.get("sample_mean") or 0.0)
        std = float(row.get("sample_std") or 1.0)
        value = crps_normal(mean, std, observed)
    else:
        raise ValueError(f"CRPS is not implemented for distribution_type={dist!r}")
    return {
        "metric_version": METRIC_VERSION,
        "metric": "crps",
        "value": round(value, 6),
        "distribution_type": dist,
    }
