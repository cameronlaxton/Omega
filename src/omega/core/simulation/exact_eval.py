"""Exact analytic market evaluation for parametric simulation backends.

The ``fast_score`` Poisson archetypes (soccer, baseball, hockey) and the
negative-binomial prop backend build a closed-form outcome distribution and then
Monte-Carlo *sample* it. This module evaluates the same markets **exactly** by
summing that distribution, removing the MC sampling noise that the backtest's
``edge >= edge_threshold`` filter selects on (the optimizer's-curse bias the
exact-eval plan targets).

Design rules:

* Pure leaf module — numpy + stdlib only, no project imports — so it cannot
  introduce an import cycle and is trivially unit-testable.
* It consumes an *already-built, normalized* joint score grid. Grid construction
  stays the single source of truth in ``engine.py`` (shared by the MC sampler and
  this evaluator), so MC and exact can never drift in the model itself — only in
  evaluation, which the parity tests pin to within MC standard error.
* The market formulas here mirror ``engine._build_team_score_result`` cell for
  cell. Any change to one must change the other; ``tests/core/test_exact_eval.py``
  guards the equivalence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def normal_cdf(x: float) -> float:
    """Standard-normal CDF via the stdlib error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pmf_stats(values: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return ``(mean, std, p10, p50, p90)`` of a discrete distribution.

    Quantiles are the smallest value whose cumulative mass reaches the target —
    the exact-distribution analogue of the nearest-rank percentile the MC path
    computes over sorted samples.
    """
    probs = np.asarray(probs, dtype=float)
    values = np.asarray(values, dtype=float)
    total = float(probs.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    probs = probs / total
    mean = float((values * probs).sum())
    var = float((probs * (values - mean) ** 2).sum())
    std = math.sqrt(max(0.0, var))
    cum = np.cumsum(probs)

    def _q(q: float) -> float:
        idx = int(np.searchsorted(cum, q, side="left"))
        idx = min(idx, values.size - 1)
        return float(values[idx])

    return mean, std, _q(0.10), _q(0.50), _q(0.90)


def discrete_joint_market_probs(
    grid: np.ndarray,
    *,
    supports_draw: bool,
    spread_home: float | None,
    over_under: float | None,
) -> dict[str, Any]:
    """Exact market probabilities from a normalized joint score grid.

    ``grid[h, a]`` is ``P(home scores h, away scores a)`` and must sum to ~1.
    Returns raw (un-rounded, 0-1 scale) probabilities and predicted means; the
    caller applies the same percent rounding the MC builder uses. Mirrors the
    market formulas in ``engine._build_team_score_result``.
    """
    grid = np.asarray(grid, dtype=float)
    g = grid.shape[0]
    idx = np.arange(g)
    home_goals = idx.reshape(-1, 1)
    away_goals = idx.reshape(1, -1)
    margin = home_goals - away_goals  # margin[h, a] = h - a
    total = home_goals + away_goals

    home_marg = grid.sum(axis=1)
    away_marg = grid.sum(axis=0)

    home_win = float(grid[margin > 0].sum())
    away_win = float(grid[margin < 0].sum())
    draw = float(grid[margin == 0].sum())

    # Pre-reallocation values feed the (draw-supporting) 3-way derived markets.
    home_win_orig, away_win_orig, draw_orig = home_win, away_win, draw

    if not supports_draw and draw > 0.0:
        # Continuous analogue of engine._allocate_ties: split tie mass in
        # proportion to decisive outcomes.
        decisive = home_win + away_win
        home_share = (home_win / decisive) if decisive > 0 else 0.5
        home_win += draw * home_share
        away_win += draw * (1.0 - home_share)
        draw = 0.0

    home_mean = float((idx * home_marg).sum())
    away_mean = float((idx * away_marg).sum())

    out: dict[str, Any] = {
        "home_win": home_win,
        "away_win": away_win,
        "draw": draw,
        "home_mean": home_mean,
        "away_mean": away_mean,
    }

    if supports_draw:
        both_score = float(grid[(home_goals > 0) & (away_goals > 0)].sum())
        out["double_chance_home_draw"] = home_win_orig + draw_orig
        out["double_chance_home_away"] = home_win_orig + away_win_orig
        out["double_chance_away_draw"] = away_win_orig + draw_orig
        decisive = home_win_orig + away_win_orig
        if decisive > 0:
            out["dnb_home"] = home_win_orig / decisive
            out["dnb_away"] = away_win_orig / decisive
        else:
            out["dnb_home"] = 0.0
            out["dnb_away"] = 0.0
        out["btts_yes"] = both_score
        out["btts_no"] = 1.0 - both_score

    if spread_home is not None:
        threshold = -spread_home
        out["home_cover"] = float(grid[margin > threshold].sum())
        out["away_cover"] = float(grid[margin < threshold].sum())

    if over_under is not None:
        out["over"] = float(grid[total > over_under].sum())
        out["under"] = float(grid[total < over_under].sum())

    return out


def correct_score_probs(grid: np.ndarray, max_goals: int = 5) -> dict[str, float]:
    """Exact correct-score map ``{"h-a": pct}`` matching
    ``engine._correct_score_distribution`` (scorelines above ``max_goals`` bucket
    into ``"other"``; percentages rounded to 0.1).
    """
    grid = np.asarray(grid, dtype=float)
    g = grid.shape[0]
    counts: dict[str, float] = {}
    for h in range(g):
        for a in range(g):
            p = float(grid[h, a])
            if p == 0.0:
                continue
            key = f"{h}-{a}" if h <= max_goals and a <= max_goals else "other"
            counts[key] = counts.get(key, 0.0) + p
    return {k: round(v * 100, 1) for k, v in sorted(counts.items())}


def margin_total_pmfs(grid: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Exact pmfs of goal margin and full-time total as ``{str(int): prob}`` maps,
    matching the ``margin_counts`` / ``total_counts`` shape the soccer-derivatives
    edge path normalizes (``_normalize_counts`` divides by the sum, so emitting
    probabilities rather than integer counts is exact).
    """
    grid = np.asarray(grid, dtype=float)
    g = grid.shape[0]
    margin_pmf: dict[str, float] = {}
    total_pmf: dict[str, float] = {}
    for h in range(g):
        for a in range(g):
            p = float(grid[h, a])
            if p == 0.0:
                continue
            mk = str(h - a)
            tk = str(h + a)
            margin_pmf[mk] = margin_pmf.get(mk, 0.0) + p
            total_pmf[tk] = total_pmf.get(tk, 0.0) + p
    return margin_pmf, total_pmf


def thinned_total_pmf(grid: np.ndarray, share: float) -> dict[str, float]:
    """Exact pmf of an independently-thinned total (e.g. first-half goals).

    If ``T`` is the full total and each goal lands in the bucket with probability
    ``share`` independently, ``P(K=k) = sum_t P(T=t) * Binom(k; t, share)``. This
    is the exact analogue of the soccer backend's
    ``rng.binomial(total, share)`` thinning.
    """
    grid = np.asarray(grid, dtype=float)
    g = grid.shape[0]
    max_total = 2 * (g - 1)
    total_pmf = np.zeros(max_total + 1)
    for h in range(g):
        for a in range(g):
            total_pmf[h + a] += float(grid[h, a])

    out: dict[str, float] = {}
    for t in range(max_total + 1):
        pt = total_pmf[t]
        if pt == 0.0:
            continue
        ks = np.arange(t + 1)
        # Binomial(t, share) pmf via lgamma for numerical stability.
        log_coeff = _lgamma_arr(t + 1) - _lgamma_arr(ks + 1) - _lgamma_arr(t - ks + 1)
        with np.errstate(divide="ignore"):
            log_p = log_coeff + ks * math.log(share) + (t - ks) * math.log1p(-share)
        binom = np.exp(log_p)
        for k in range(t + 1):
            out[str(k)] = out.get(str(k), 0.0) + pt * float(binom[k])
    return out


def _phi(x: float) -> float:
    """Standard-normal pdf."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def censored_normal_mean(mu: float, sigma: float) -> float:
    """``E[max(0, X)]`` for ``X ~ N(mu, sigma)`` — the closed form for the mean of
    the clipped-normal score the Normal archetypes sample (``max(0, s)``)."""
    if sigma <= 0:
        return max(0.0, mu)
    z = mu / sigma
    return mu * normal_cdf(z) + sigma * _phi(z)


def censored_normal_grid(
    mu: float, sigma: float, step: float, max_val: float
) -> tuple[np.ndarray, np.ndarray]:
    """Discretized clipped-normal ``S = max(0, N(mu, sigma))`` on ``0, step, ...``.

    ``S`` is rounded to the nearest grid point: the 0 bin carries all mass below
    ``step/2`` (the clip atom plus the (0, step/2) sliver); bin ``k*step`` carries
    ``P(S in ((k-0.5)step, (k+0.5)step])``. With a fine ``step`` the rounding error
    is far below MC standard error. Returns ``(values, probs)``.
    """
    n = int(math.ceil(max_val / step)) + 1
    values = np.arange(n) * step
    edges = (np.arange(n + 1) - 0.5) * step  # bin edges; edges[0] = -step/2
    edges[0] = -np.inf  # bin 0 absorbs the clip atom (all mass at or below 0)
    cdf = np.array([normal_cdf((e - mu) / sigma) if np.isfinite(e) else 0.0 for e in edges])
    probs = np.diff(cdf)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return values, probs


def gaussian_censored_market_probs(
    mu_h: float,
    sigma_h: float,
    mu_a: float,
    sigma_a: float,
    *,
    supports_draw: bool,
    spread_home: float | None,
    over_under: float | None,
) -> dict[str, Any]:
    """Exact markets for two independent clipped-normal scores (Normal archetypes).

    Builds each team's clipped-normal grid and convolves to get the exact margin
    and total distributions, then sums the tails — the closed-form analogue of the
    Monte-Carlo ``max(0, N(mu, sigma))`` sampler. Predicted scores use the exact
    censored mean. Mirrors the market formulas in ``_build_team_score_result``.
    """
    step = max(min(sigma_h, sigma_a) / 50.0, 1e-3)
    max_val = max(mu_h + 8.0 * sigma_h, mu_a + 8.0 * sigma_a, 10.0 * step)
    v_h, p_h = censored_normal_grid(mu_h, sigma_h, step, max_val)
    v_a, p_a = censored_normal_grid(mu_a, sigma_a, step, max_val)
    n = v_h.size

    # Margin M = home - away: correlation of the two pmfs. Index k maps to margin
    # value (k - (n-1)) * step.
    margin_probs = np.convolve(p_h, p_a[::-1])
    margin_vals = (np.arange(margin_probs.size) - (n - 1)) * step

    # Total T = home + away: straight convolution. Index k maps to value k * step.
    total_probs = np.convolve(p_h, p_a)
    total_vals = np.arange(total_probs.size) * step

    home_win = float(margin_probs[margin_vals > 0].sum())
    away_win = float(margin_probs[margin_vals < 0].sum())
    draw = float(margin_probs[margin_vals == 0].sum())

    if not supports_draw and draw > 0.0:
        decisive = home_win + away_win
        home_share = (home_win / decisive) if decisive > 0 else 0.5
        home_win += draw * home_share
        away_win += draw * (1.0 - home_share)
        draw = 0.0

    out: dict[str, Any] = {
        "home_win": home_win,
        "away_win": away_win,
        "draw": draw,
        "home_mean": censored_normal_mean(mu_h, sigma_h),
        "away_mean": censored_normal_mean(mu_a, sigma_a),
        "_dists": {
            "home": (v_h, p_h),
            "away": (v_a, p_a),
            "total": (total_vals, total_probs),
            "spread": (margin_vals, margin_probs),
        },
    }

    if spread_home is not None:
        threshold = -spread_home
        out["home_cover"] = float(margin_probs[margin_vals > threshold].sum())
        out["away_cover"] = float(margin_probs[margin_vals < threshold].sum())

    if over_under is not None:
        out["over"] = float(total_probs[total_vals > over_under].sum())
        out["under"] = float(total_probs[total_vals < over_under].sum())

    return out


def normal_grid(
    mu: float, sigma: float, *, num: int = 2001, n_sigma: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """A fine discretized ``N(mu, sigma)`` over ``mu ± n_sigma*sigma`` for building
    distribution rows of an uncensored normal (values may be negative)."""
    if sigma <= 0:
        return np.array([mu]), np.array([1.0])
    values = np.linspace(mu - n_sigma * sigma, mu + n_sigma * sigma, num)
    z = (values - mu) / sigma
    probs = np.exp(-0.5 * z * z)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return values, probs


def gaussian_market_probs(
    mu_h: float,
    sigma_h: float,
    mu_a: float,
    sigma_a: float,
    *,
    supports_draw: bool,
    spread_home: float | None,
    over_under: float | None,
) -> dict[str, Any]:
    """Exact markets for two independent *uncensored* normal scores (golf, where
    scores are far from any 0 clip). Margin and total are themselves normal, so the
    markets are closed-form normal-CDF evaluations — no grid needed. Mirrors the
    market formulas in ``_build_team_score_result``.
    """
    sig_m = math.hypot(sigma_h, sigma_a)  # sd of both margin and total (independent)
    mu_m = mu_h - mu_a
    mu_t = mu_h + mu_a

    if sig_m > 0:
        home_win = normal_cdf(mu_m / sig_m)
        away_win = normal_cdf(-mu_m / sig_m)
    else:
        home_win = float(mu_m > 0)
        away_win = float(mu_m < 0)
    draw = max(0.0, 1.0 - home_win - away_win)  # ~0 for continuous scores

    if not supports_draw and draw > 0.0:
        decisive = home_win + away_win
        home_share = (home_win / decisive) if decisive > 0 else 0.5
        home_win += draw * home_share
        away_win += draw * (1.0 - home_share)
        draw = 0.0

    out: dict[str, Any] = {
        "home_win": home_win,
        "away_win": away_win,
        "draw": draw,
        "home_mean": mu_h,
        "away_mean": mu_a,
        "_dists": {
            "home": normal_grid(mu_h, sigma_h),
            "away": normal_grid(mu_a, sigma_a),
            "total": normal_grid(mu_t, sig_m),
            "spread": normal_grid(mu_m, sig_m),
        },
    }

    if spread_home is not None and sig_m > 0:
        threshold = -spread_home
        out["home_cover"] = normal_cdf((mu_m - threshold) / sig_m)
        out["away_cover"] = normal_cdf((threshold - mu_m) / sig_m)

    if over_under is not None and sig_m > 0:
        out["over"] = normal_cdf((mu_t - over_under) / sig_m)
        out["under"] = normal_cdf((over_under - mu_t) / sig_m)

    return out


def negative_binomial_pmf(mean: float, k: float, max_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values, probs)`` for the negative-binomial count distribution.

    Uses the same parametrization as ``NegBinomPropBackend``: ``p = k / (k + mean)``,
    ``X`` = number of failures before ``k`` successes, ``E[X] = mean``. The pmf is
    evaluated over ``0..max_count`` via lgamma and renormalized (truncation mass is
    negligible when ``max_count`` covers the tail).
    """
    p = k / (k + mean)
    values = np.arange(max_count + 1)
    log_pmf = (
        _lgamma_arr(values + k)
        - _lgamma_arr(k)
        - _lgamma_arr(values + 1)
        + k * math.log(p)
        + values * math.log1p(-p)
    )
    probs = np.exp(log_pmf)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return values, probs


def _lgamma_arr(arr: np.ndarray | float) -> np.ndarray:
    """Vectorized lgamma (numpy has no ufunc for it; use math.lgamma elementwise)."""
    a = np.asarray(arr, dtype=float)
    flat = a.reshape(-1)
    out = np.array([math.lgamma(x) for x in flat], dtype=float)
    return out.reshape(a.shape)
