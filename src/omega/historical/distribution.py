"""Distribution-shift helpers for the historical-vs-live parity gate.

PSI (Population Stability Index) with a Jensen-Shannon cross-check. Conventional
PSI reading: < 0.10 no shift, 0.10-0.25 moderate, > 0.25 significant.
"""

from __future__ import annotations

import math

PROB_BINS = [i / 10 for i in range(11)]  # 10 equal-width [0,1] bins


def histogram(values: list[float], bins: list[float]) -> list[int]:
    counts = [0] * (len(bins) - 1)
    last = len(bins) - 2
    for v in values:
        for i in range(len(bins) - 1):
            upper_ok = v < bins[i + 1] or (i == last and v <= bins[i + 1])
            if v >= bins[i] and upper_ok:
                counts[i] += 1
                break
    return counts


def _dist(counts: list[int], eps: float = 1e-6) -> list[float]:
    total = sum(counts)
    if total == 0:
        n = len(counts) or 1
        return [1.0 / n] * len(counts)
    return [max(c / total, eps) for c in counts]


def psi(expected: list[float], actual: list[float], bins: list[float] = PROB_BINS) -> float:
    """PSI between an expected (live) and actual (historical) numeric sample."""
    e = _dist(histogram(expected, bins))
    a = _dist(histogram(actual, bins))
    return sum((a_i - e_i) * math.log(a_i / e_i) for e_i, a_i in zip(e, a))


def js_divergence(p_vals: list[float], q_vals: list[float], bins: list[float] = PROB_BINS) -> float:
    p = _dist(histogram(p_vals, bins))
    q = _dist(histogram(q_vals, bins))
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def category_psi(expected_labels: list, actual_labels: list) -> float:
    """PSI over a categorical label distribution (e.g. context_source mix)."""
    cats = sorted({str(x) for x in expected_labels} | {str(x) for x in actual_labels})
    if not cats:
        return 0.0
    e = _dist([sum(1 for x in expected_labels if str(x) == c) for c in cats])
    a = _dist([sum(1 for x in actual_labels if str(x) == c) for c in cats])
    return sum((a_i - e_i) * math.log(a_i / e_i) for e_i, a_i in zip(e, a))
