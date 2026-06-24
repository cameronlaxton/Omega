"""
omega.strategy.clv_significance — the statistical bar for graduating CLV-scored signals.

Pure math (no I/O, no DB), mirroring ``signal_performance.py``. Implements the
"Dual-Gate + FDR" probation framework decided for issue #28 — the rule a signal
(or an LLM-proposed hypothesis) must clear before it earns trust, so we never
overfit to short-term CLV noise:

1. **Bootstrapped CLV confidence interval.** ``clv_cents_when_followed`` carries
   variance — a signal can look incredible over a small sample purely because it
   caught a few late-steam moves by accident. We resample the per-trace CLV
   observations with replacement and require the 5th percentile of the
   bootstrapped *mean* CLV to be strictly > 0. (The sign of the mean and of the
   cumulative sum agree for any n > 0, so this is the "cumulative CLV > 0" test,
   expressed per-observation so it is comparable across signals.)
2. **Benjamini–Hochberg FDR.** When many hypotheses are tested in one run (the
   expanded LLM feature grammar makes this a real multiple-testing problem), we
   control the false-discovery rate so an enlarged search space does not
   manufacture spurious "edges". The more hypotheses tested, the stricter the
   per-signal threshold.
3. **Power-derived N_min.** Even with a positive bound we enforce an absolute
   minimum sample size derived from a power analysis (default: 80% power to
   detect a 2% edge), so low-volume anomalies cannot graduate.

A signal graduates iff it clears **all three** gates (see :func:`graduation_mask`).

Determinism: bootstrap resampling is seeded; the caller derives the seed from the
``dataset_hash`` (see :func:`seed_from_dataset_hash`) so a rerun on identical data
yields identical bounds — required by the repo determinism invariant.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

__all__ = [
    "DEFAULT_FDR_Q",
    "DEFAULT_POWER_EDGE",
    "ProbationStats",
    "alignment_pvalue",
    "benjamini_hochberg",
    "bootstrap_clv_lower_bound",
    "clv_pvalue",
    "clv_pvalue_from_stats",
    "compute_probation_stats",
    "graduation_mask",
    "min_samples_for_power",
    "normal_lower_bound",
    "pooled_mean_std",
    "seed_from_dataset_hash",
]

# Defaults — the architect's settled framework (issue #28 comments).
DEFAULT_FDR_Q = 0.10  # Benjamini–Hochberg target false-discovery rate.
DEFAULT_POWER_EDGE = 0.02  # detect a 2% edge ...
DEFAULT_POWER = 0.80  # ... at 80% power ...
DEFAULT_ALPHA = 0.05  # ... one-sided at 5%.
_DEFAULT_RESAMPLES = 2000
_BOOTSTRAP_PCT = 5.0  # 5th percentile lower bound.

_STD_NORMAL = NormalDist()


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------


def seed_from_dataset_hash(dataset_hash: str, salt: str = "") -> int:
    """Derive a stable 63-bit bootstrap seed from a dataset hash (+ optional salt).

    Salting with the signal key decorrelates each signal's resampling stream while
    keeping the whole run reproducible from the dataset hash. Empty/garbage hashes
    are tolerated (they hash to a fixed seed) so scoring never crashes on thin data.
    """
    digest = hashlib.sha256(f"{dataset_hash}|{salt}".encode()).digest()
    # 63 bits keeps it a non-negative signed int on every platform.
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Gate 1 — bootstrapped CLV lower bound
# ---------------------------------------------------------------------------


def bootstrap_clv_lower_bound(
    values: Sequence[float],
    *,
    seed: int,
    n_resamples: int = _DEFAULT_RESAMPLES,
    pct: float = _BOOTSTRAP_PCT,
) -> float:
    """Return the ``pct`` percentile of the bootstrapped mean of ``values``.

    Resamples ``values`` with replacement ``n_resamples`` times, takes each
    resample's mean, and returns the requested lower percentile. A signal clears
    gate 1 iff this is strictly > 0 (the edge is distributed across the data, not
    driven by a few outliers).

    Empty input returns ``-inf`` (no evidence — never clears the gate). A single
    observation returns that value (degenerate but well-defined).
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return float("-inf")
    if n == 1:
        return float(arr[0])
    rng = np.random.default_rng(seed)
    # (n_resamples, n) index matrix → resampled means, vectorised + deterministic.
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, pct))


# ---------------------------------------------------------------------------
# Gate 2 — Benjamini–Hochberg FDR control
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvalues: Sequence[float], q: float = DEFAULT_FDR_Q) -> list[bool]:
    """Benjamini–Hochberg step-up FDR control.

    Returns a boolean mask (aligned to input order): True where the hypothesis is
    rejected (i.e. the signal's CLV edge is accepted as real) at FDR level ``q``.
    The classic step-up: sort p ascending, find the largest rank ``k`` (1-indexed)
    with ``p_(k) <= (k/m) * q``, and reject the ``k`` smallest p-values.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    max_k = 0
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= (rank / m) * q:
            max_k = rank
    mask = [False] * m
    for rank, i in enumerate(order, start=1):
        if rank <= max_k:
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Gate 3 — power-derived minimum sample size
# ---------------------------------------------------------------------------


def min_samples_for_power(
    edge: float = DEFAULT_POWER_EDGE,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
    baseline: float = 0.5,
) -> int:
    """Minimum n to detect ``edge`` at ``power`` (one-sided ``alpha``).

    Normal-approximation sample size for a one-sample proportion test of
    ``p1 = baseline + edge`` against ``p0 = baseline``. With the defaults this is
    the "80% power to detect a 2% edge" floor from the issue. Returned as a
    ceiling-rounded int; always >= 1.
    """
    if edge <= 0:
        raise ValueError("edge must be > 0")
    p0 = baseline
    p1 = min(1.0, max(0.0, baseline + edge))
    z_alpha = _STD_NORMAL.inv_cdf(1.0 - alpha)
    z_beta = _STD_NORMAL.inv_cdf(power)
    numerator = z_alpha * math.sqrt(p0 * (1.0 - p0)) + z_beta * math.sqrt(p1 * (1.0 - p1))
    n = (numerator * numerator) / (edge * edge)
    return max(1, math.ceil(n))


# ---------------------------------------------------------------------------
# p-values for the CLV hypotheses
# ---------------------------------------------------------------------------


def clv_pvalue(values: Sequence[float]) -> float:
    """One-sided p-value for H1: mean(CLV) > 0 via a one-sample z-test.

    Returns 1.0 (no evidence) for n < 2 or zero variance with a non-positive mean;
    0.0 for zero variance with a strictly positive mean. p in [0, 1].
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 2:
        return 1.0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return 0.0 if mean > 0.0 else 1.0
    z = mean / (sd / math.sqrt(n))
    return 1.0 - _STD_NORMAL.cdf(z)


def pooled_mean_std(stats: Sequence[tuple[int, float, float]]) -> tuple[int, float, float]:
    """Pool per-group ``(n, mean, std)`` into the combined ``(N, mean, sample_std)``.

    Reconstructs the exact pooled sample SD from sufficient statistics (no raw
    data): total SS = within-group SS + between-group SS. Lets the fit combine a
    signal type's per-(source, window, league) rows into one signal_type-level
    distribution. Groups with n < 1 are ignored; returns ``(0, 0.0, 0.0)`` when
    empty and ``std = 0.0`` when the pooled N < 2.
    """
    groups = [(int(n), float(m), float(s or 0.0)) for n, m, s in stats if int(n) >= 1]
    total_n = sum(n for n, _, _ in groups)
    if total_n == 0:
        return 0, 0.0, 0.0
    grand_mean = sum(n * m for n, m, _ in groups) / total_n
    if total_n < 2:
        return total_n, grand_mean, 0.0
    ss_within = sum((n - 1) * s * s for n, _, s in groups)
    ss_between = sum(n * (m - grand_mean) ** 2 for n, m, _ in groups)
    combined_var = (ss_within + ss_between) / (total_n - 1)
    return total_n, grand_mean, math.sqrt(max(0.0, combined_var))


def normal_lower_bound(mean: float, std: float, n: int, pct: float = _BOOTSTRAP_PCT) -> float:
    """Lower ``pct`` percentile confidence bound on the mean (normal approximation).

    The poolable analogue of :func:`bootstrap_clv_lower_bound` for aggregated
    signal_type rows where only ``(n, mean, std)`` survive: ``mean + z_pct * SE``.
    At the power-derived N_min sizes the CLT makes this ~equal to the bootstrap
    percentile. Returns ``-inf`` for n < 1 and ``mean`` when std == 0.
    """
    if n < 1:
        return float("-inf")
    if std <= 0.0:
        return mean
    z = _STD_NORMAL.inv_cdf(pct / 100.0)  # negative for pct < 50
    return mean + z * (std / math.sqrt(n))


def clv_pvalue_from_stats(mean: float, std: float, n: int) -> float:
    """One-sided p-value for H1: mean > 0 from sufficient stats (z-test).

    The poolable analogue of :func:`clv_pvalue`. Returns 1.0 for n < 2; for zero
    variance returns 0.0 if mean > 0 else 1.0.
    """
    if n < 2:
        return 1.0
    if std <= 0.0:
        return 0.0 if mean > 0.0 else 1.0
    z = mean / (std / math.sqrt(n))
    return 1.0 - _STD_NORMAL.cdf(z)


def alignment_pvalue(n_aligned: int, n_total: int, baseline: float = 0.5) -> float:
    """One-sided p-value for H1: alignment rate > ``baseline`` (normal approx to binomial).

    Used when the per-signal evidence is the count of CLV-aligned traces rather
    than continuous CLV cents. Returns 1.0 for n_total < 1.
    """
    if n_total < 1:
        return 1.0
    rate = n_aligned / n_total
    se = math.sqrt(baseline * (1.0 - baseline) / n_total)
    if se == 0.0:
        return 0.0 if rate > baseline else 1.0
    z = (rate - baseline) / se
    return 1.0 - _STD_NORMAL.cdf(z)


# ---------------------------------------------------------------------------
# Composed per-signal verdict + run-level FDR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbationStats:
    """All three gates' raw measurements for one signal/proposal key.

    ``graduates`` is decided at the run level by :func:`graduation_mask` because
    the FDR gate is inherently multiple-testing (it depends on the other keys).
    """

    n: int
    n_min: int
    clv_mean: float
    boot_lower_bound: float
    pvalue: float
    meets_n_min: bool
    boot_positive: bool


def compute_probation_stats(
    clv_values: Sequence[float],
    *,
    n_min: int,
    seed: int,
    n_resamples: int = _DEFAULT_RESAMPLES,
) -> ProbationStats:
    """Measure gates 1 and 3 (and the p-value feeding gate 2) for one signal.

    ``clv_values`` are the per-trace CLV-cents observations on traces where the
    signal fired. ``n_min`` is the power-derived floor (doubled by the caller for
    toxic signals such as ``stale_line``).
    """
    arr = np.asarray(clv_values, dtype=float)
    n = int(arr.size)
    clv_mean = float(arr.mean()) if n else 0.0
    boot_lb = bootstrap_clv_lower_bound(clv_values, seed=seed, n_resamples=n_resamples)
    return ProbationStats(
        n=n,
        n_min=n_min,
        clv_mean=clv_mean,
        boot_lower_bound=boot_lb,
        pvalue=clv_pvalue(clv_values),
        meets_n_min=(n >= n_min),
        boot_positive=(boot_lb > 0.0),
    )


def graduation_mask(
    stats_by_key: Mapping[Hashable, ProbationStats],
    q: float = DEFAULT_FDR_Q,
) -> dict[Hashable, bool]:
    """Decide which signals clear ALL THREE gates.

    A key graduates iff it (1) clears the bootstrap lower bound, (3) meets N_min,
    and (2) survives Benjamini–Hochberg FDR control over the p-values of the keys
    that already passed gates 1 and 3. FDR is applied only among those eligible
    candidates so failing keys do not dilute the correction.
    """
    keys = list(stats_by_key.keys())
    eligible = [k for k in keys if stats_by_key[k].meets_n_min and stats_by_key[k].boot_positive]
    result: dict[Hashable, bool] = {k: False for k in keys}
    if not eligible:
        return result
    fdr_pass = benjamini_hochberg([stats_by_key[k].pvalue for k in eligible], q=q)
    for k, passed in zip(eligible, fdr_pass, strict=True):
        result[k] = passed
    return result
