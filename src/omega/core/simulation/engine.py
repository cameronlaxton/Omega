"""
Simulation Engine Module

Runs Monte Carlo simulations for team markets (spreads, totals, ML) and player props,
dispatching to sport-archetype-specific models.

Architecture: Engine is input-driven only. No network calls. Callers must supply
home_context, away_context (and for player props: game_context, player_context).

Sport archetypes:
    basketball        - ORtg/DRtg/pace possession model (Normal)
    american_football - Points/drives efficiency model (Normal)
    baseball          - Run environment model (Poisson), pitcher-aware
    hockey            - Goal/shot model with goalie (Poisson), regulation draw
    soccer            - Goal model (Poisson), 3-way result
    tennis            - Point-level probability, best-of-N sets (Bernoulli)
    golf              - Field probability, strokes-gained model (Normal)
    fighting          - Win probability + method-of-victory (Bernoulli)
    esports           - Map win probability, best-of-N (Bernoulli)
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from omega.core.config.leagues import get_league_config
from omega.core.simulation.archetypes import (
    get_archetype,
    get_archetype_name,
)
from omega.core.simulation.backends import (
    GameSimulationBackend,
    GameSimulationInput,
    PropSimulationInput,
    enforce_game_backend_contract,
    register_game_backend,
    register_prop_backend,
)

# ---------------------------------------------------------------------------
# Distribution samplers
# ---------------------------------------------------------------------------


def select_distribution(
    metric_key: str,
    league: str,
    mean: float | None = None,
    override: str | None = None,
) -> str:
    """
    Selects appropriate distribution (Normal vs Poisson) based on metric and league.

    Args:
        metric_key: stat key (e.g. "pts", "blk", "kills")
        league: league code (e.g. "NBA", "MLB")
        mean: optional expected mean; used to route low-mean basketball count stats
            (blk/stl/3pm/oreb/dreb/to) to Poisson where Normal would understate
            right-tail mass.
        override: optional caller override ("normal" or "poisson"). When supplied
            and valid, it short-circuits all routing logic.
    """
    if override in {"normal", "poisson"}:
        return override

    league = league.upper()
    metric_key = metric_key.lower()

    if metric_key in {"run_rate", "goal_rate"} or league in {"MLB", "NHL"}:
        return "poisson"

    if league in {"NFL", "NCAAF"} and metric_key == "score":
        return "poisson"

    discrete_stats = {
        "goals",
        "td",
        "touchdowns",
        "receptions",
        "rec",
        "sog",
        "aces",
        "double_faults",
        "kills",
        "deaths",
        "assists_esport",
        "hrs",
        "stolen_bases",
        "saves",
    }
    if metric_key in discrete_stats:
        return "poisson"

    # Low-mean basketball count stats: Normal grossly understates right-tail mass
    # at low lambda. A blk prop with line=1.5 and mean=0.6 under Normal yields
    # under_prob ~ 0.95; under Poisson it correctly lands near 0.88.
    low_count_basketball = {
        "blk",
        "stl",
        "3pm",
        "oreb",
        "dreb",
        "to",
        "blocks",
        "steals",
        "turnovers",
        "offensive_rebounds",
    }
    if metric_key in low_count_basketball and (mean is None or mean < 3.0):
        return "poisson"

    return "normal"


def _poisson_sample(lam: float, size: int, rng: np.random.Generator | random.Random | None = None) -> list[float]:
    """Generate Poisson samples."""
    lam = max(0.01, lam)
    if rng is not None:
        if isinstance(rng, np.random.Generator):
            return rng.poisson(lam=lam, size=size).tolist()
        else:
            samples = []
            for _ in range(size):
                L = pow(2.718281828459045, -lam)
                k, p = 0, 1.0
                while p > L:
                    k += 1
                    p *= rng.random()
                samples.append(k - 1)
            return samples

    if np is not None:
        return np.random.poisson(lam=lam, size=size).tolist()
    samples = []
    for _ in range(size):
        L = pow(2.718281828459045, -lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= random.random()
        samples.append(k - 1)
    return samples


def _normal_sample(mu: float, sigma: float, size: int, rng: np.random.Generator | random.Random | None = None) -> list[float]:
    """Generate Normal samples."""
    sigma = max(0.1, sigma)
    if rng is not None:
        if isinstance(rng, np.random.Generator):
            return rng.normal(mu, sigma, size).tolist()
        else:
            return [rng.gauss(mu, sigma) for _ in range(size)]

    if np is not None:
        return np.random.normal(mu, sigma, size).tolist()
    return [random.gauss(mu, sigma) for _ in range(size)]


def _expected_against_allowed_rate(
    team_off: float,
    opponent_allowed: float,
    league_avg_allowed: float,
    pace_factor: float = 1.0,
) -> float:
    """Expected score when defense is an allowed-rate metric; lower is better."""
    if opponent_allowed <= 0 or league_avg_allowed <= 0:
        return team_off * pace_factor
    return team_off * (opponent_allowed / league_avg_allowed) * pace_factor


def _bernoulli_sample(p: float, size: int, rng: np.random.Generator | random.Random | None = None) -> list[int]:
    """Generate Bernoulli samples (0 or 1)."""
    p = max(0.001, min(0.999, p))
    if rng is not None:
        if isinstance(rng, np.random.Generator):
            return rng.binomial(1, p, size).tolist()
        else:
            return [1 if rng.random() < p else 0 for _ in range(size)]

    if np is not None:
        return np.random.binomial(1, p, size).tolist()
    return [1 if random.random() < p else 0 for _ in range(size)]


def _correct_score_distribution(
    home_scores: list,
    away_scores: list,
    max_goals: int = 5,
) -> dict[str, float]:
    """Empirical correct-score distribution as {"home-away": pct}.

    Scorelines with either side above ``max_goals`` are bucketed under
    "other" so the map stays bounded. Percentages sum to ~100.
    """
    n = len(home_scores)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for h, a in zip(home_scores, away_scores):
        hi, ai = int(h), int(a)
        key = f"{hi}-{ai}" if hi <= max_goals and ai <= max_goals else "other"
        counts[key] = counts.get(key, 0) + 1
    return {k: round(v / n * 100, 1) for k, v in sorted(counts.items())}


def _percentile(sorted_values: list[float], q: float) -> float:
    """Return a deterministic nearest-rank percentile from sorted samples."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return float(sorted_values[idx])


def _context_hash(*contexts: dict | None) -> str:
    payload = [ctx or {} for ctx in contexts]
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _distribution_row(
    *,
    target: str,
    market: str,
    stat_key: str | None,
    distribution_type: str,
    distribution_params: dict[str, Any],
    samples: list[float],
    n_iterations: int,
    seed: int | None,
    context_hash: str | None,
    component_version: str = "fast_sim_v1",
) -> dict[str, Any]:
    sorted_vals = sorted(float(v) for v in samples)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n if n else 0.0
    variance = sum((v - mean) ** 2 for v in sorted_vals) / n if n else 0.0
    return {
        "target": target,
        "market": market,
        "stat_key": stat_key,
        "distribution_type": distribution_type,
        "distribution_params": distribution_params,
        "params_schema_version": 1,
        "sample_mean": mean,
        "sample_std": variance**0.5,
        "p10": _percentile(sorted_vals, 0.10),
        "p50": _percentile(sorted_vals, 0.50),
        "p90": _percentile(sorted_vals, 0.90),
        "n_iterations": n_iterations,
        "seed": seed,
        "context_hash": context_hash,
        "component_version": component_version,
    }


# ---------------------------------------------------------------------------
# Skip / missing-requirements helpers
# ---------------------------------------------------------------------------


def _skip_result(
    home_team: str,
    away_team: str,
    league: str,
    skip_reason: str,
    missing_requirements: list[str] | None = None,
) -> dict:
    """Build a standard skip response."""
    return {
        "success": False,
        "skipped": True,
        "skip_reason": skip_reason,
        "missing_requirements": missing_requirements or [],
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
        "context_source": "missing",
        "baseline_used": False,
    }


def _validate_required_keys(
    context: dict | None, side: str, required_keys: tuple, league: str
) -> list[str]:
    """Return list of missing requirement strings for a context dict."""
    if context is None:
        # Enumerate all required keys so callers know exactly what to supply.
        return [f"{side}_context.{k}" for k in required_keys]
    missing = []
    for key in required_keys:
        val = context.get(key)
        if val is None:
            missing.append(f"{side}_context.{key}")
    return missing


# ---------------------------------------------------------------------------
# Simulation result builder
# ---------------------------------------------------------------------------


def _build_team_score_result(
    home_team: str,
    away_team: str,
    league: str,
    n_iterations: int,
    home_scores: list[float],
    away_scores: list[float],
    home_context: dict | None = None,
    away_context: dict | None = None,
    archetype_name: str | None = None,
    spread_home: float | None = None,
    over_under: float | None = None,
    context_source: str = "provided",
    baseline_used: bool = False,
    seed: int | None = None,
    backend_name: str = "fast_score",
    component_version: str = "fast_score_v1",
) -> dict:
    """Build a standardized result dict from team score simulations."""
    archetype = get_archetype(league)
    supports_draw = bool(archetype.supports_draw) if archetype is not None else True
    home_wins = sum(1 for h, a in zip(home_scores, away_scores) if h > a)
    away_wins = sum(1 for h, a in zip(home_scores, away_scores) if a > h)
    draws = n_iterations - home_wins - away_wins
    reported_draws = draws
    if not supports_draw and draws:
        decided_home, decided_away = _allocate_ties(home_wins, away_wins, draws)
        home_wins += decided_home
        away_wins += decided_away
        reported_draws = 0

    home_mean = sum(home_scores) / n_iterations
    away_mean = sum(away_scores) / n_iterations

    result = {
        "success": True,
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
        "archetype": archetype_name,
        "iterations": n_iterations,
        "home_win_prob": round(home_wins / n_iterations * 100, 1),
        "away_win_prob": round(away_wins / n_iterations * 100, 1),
        "draw_prob": round(reported_draws / n_iterations * 100, 1),
        "predicted_home_score": round(home_mean, 1),
        "predicted_away_score": round(away_mean, 1),
        "predicted_spread": round(home_mean - away_mean, 1),
        "predicted_total": round(home_mean + away_mean, 1),
        "missing_requirements": [],
        "home_context": home_context or {},
        "away_context": away_context or {},
        "context_source": context_source,
        "baseline_used": baseline_used,
        "simulation_backend": backend_name,
        "component_version": component_version,
    }

    # Exotic 3-way derived markets (soccer, hockey regulation). Gated on
    # supports_draw so non-draw sports never carry these fields. All derived
    # from the same score arrays — no extra simulation.
    if supports_draw:
        both_score = sum(1 for h, a in zip(home_scores, away_scores) if h > 0 and a > 0)
        # Double chance: two of the three 3-way outcomes.
        result["double_chance_home_draw_prob"] = round((home_wins + draws) / n_iterations * 100, 1)
        result["double_chance_home_away_prob"] = round(
            (home_wins + away_wins) / n_iterations * 100, 1
        )
        result["double_chance_away_draw_prob"] = round((away_wins + draws) / n_iterations * 100, 1)
        # Draw-no-bet: draws void, probabilities conditional on a decisive result.
        decisive = home_wins + away_wins
        if decisive > 0:
            result["dnb_home_prob"] = round(home_wins / decisive * 100, 1)
            result["dnb_away_prob"] = round(away_wins / decisive * 100, 1)
        else:
            result["dnb_home_prob"] = 0.0
            result["dnb_away_prob"] = 0.0
        # Both teams to score.
        result["btts_yes_prob"] = round(both_score / n_iterations * 100, 1)
        result["btts_no_prob"] = round((n_iterations - both_score) / n_iterations * 100, 1)
        # Correct-score distribution over common scorelines ("H-A" → pct).
        result["correct_score_probs"] = _correct_score_distribution(home_scores, away_scores)

    if spread_home is not None:
        # spread_home convention: negative = home favored (e.g., -1.5 → home must win by 2+).
        # Home covers when margin > -spread_home; away covers when margin < -spread_home.
        threshold = -spread_home
        home_covers = sum(1 for h, a in zip(home_scores, away_scores) if h - a > threshold)
        away_covers = sum(1 for h, a in zip(home_scores, away_scores) if h - a < threshold)
        result["home_cover_prob"] = round(home_covers / n_iterations * 100, 1)
        result["away_cover_prob"] = round(away_covers / n_iterations * 100, 1)

    ctx_hash = _context_hash(home_context, away_context)
    totals = [h + a for h, a in zip(home_scores, away_scores)]
    spreads = [h - a for h, a in zip(home_scores, away_scores)]
    if over_under is not None:
        over_hits = sum(1 for total in totals if total > over_under)
        under_hits = sum(1 for total in totals if total < over_under)
        result["over_prob"] = round(over_hits / n_iterations * 100, 1)
        result["under_prob"] = round(under_hits / n_iterations * 100, 1)
    result["simulation_distributions"] = [
        _distribution_row(
            target="home_score",
            market="game_score",
            stat_key=None,
            distribution_type="empirical",
            distribution_params={"source": "monte_carlo_scores"},
            samples=home_scores,
            n_iterations=n_iterations,
            seed=seed,
            context_hash=ctx_hash,
            component_version=component_version,
        ),
        _distribution_row(
            target="away_score",
            market="game_score",
            stat_key=None,
            distribution_type="empirical",
            distribution_params={"source": "monte_carlo_scores"},
            samples=away_scores,
            n_iterations=n_iterations,
            seed=seed,
            context_hash=ctx_hash,
            component_version=component_version,
        ),
        _distribution_row(
            target="total",
            market="game_total",
            stat_key=None,
            distribution_type="empirical",
            distribution_params={"source": "monte_carlo_scores"},
            samples=totals,
            n_iterations=n_iterations,
            seed=seed,
            context_hash=ctx_hash,
            component_version=component_version,
        ),
        _distribution_row(
            target="spread",
            market="game_spread",
            stat_key=None,
            distribution_type="empirical",
            distribution_params={"source": "monte_carlo_scores"},
            samples=spreads,
            n_iterations=n_iterations,
            seed=seed,
            context_hash=ctx_hash,
            component_version=component_version,
        ),
    ]

    return result


def _allocate_ties(home_wins: int, away_wins: int, draws: int) -> tuple[int, int]:
    """Resolve impossible draw samples for sports that always produce a winner."""
    decided_home = 0
    decided_away = 0
    total_decided = home_wins + away_wins
    home_share = (home_wins / total_decided) if total_decided else 0.5
    for idx in range(draws):
        if (idx + 0.5) / draws <= home_share:
            decided_home += 1
        else:
            decided_away += 1
    return decided_home, decided_away


# ---------------------------------------------------------------------------
# Legacy standalone functions (preserved for backward compatibility)
# ---------------------------------------------------------------------------


def run_game_simulation(
    projection: dict,
    n_iter: int = 10000,
    seed: int | None = None,
    league: str | None = None,
) -> dict:
    """
    Simulates a game between two teams, returning win probabilities and score distributions.
    Vectorized using NumPy when available, otherwise falls back to isolated Random.
    """
    teams = list(projection.get("off_rating", {}).keys())
    if len(teams) != 2:
        raise ValueError("Projection requires exactly two teams.")

    resolved_league = league if league is not None else projection.get("league", "NFL")
    variance_scalar = projection.get("variance_scalar", 1.0)
    team_a, team_b = teams
    off_a = projection["off_rating"][team_a]
    off_b = projection["off_rating"][team_b]
    dist = select_distribution("score", resolved_league)

    results: dict[str, Any] = {
        "team_a_wins": 0,
        "team_b_wins": 0,
        "a_scores": [],
        "b_scores": [],
    }

    if np is not None:
        rng = np.random.default_rng(seed)
        if dist == "poisson":
            a_scores = rng.poisson(lam=max(0.01, off_a * variance_scalar), size=n_iter)
            b_scores = rng.poisson(lam=max(0.01, off_b * variance_scalar), size=n_iter)
        else:
            sigma = max(1.5, 5.0 * variance_scalar)
            a_scores = rng.normal(off_a, sigma, n_iter)
            b_scores = rng.normal(off_b, sigma, n_iter)

        # Clip to non-negative scores
        a_scores = np.maximum(0.0, a_scores)
        b_scores = np.maximum(0.0, b_scores)

        results["a_scores"] = a_scores.tolist()
        results["b_scores"] = b_scores.tolist()
        results["team_a_wins"] = int(np.sum(a_scores > b_scores))
        results["team_b_wins"] = int(np.sum(b_scores > a_scores))
    else:
        rng = random.Random(seed)
        for _ in range(n_iter):
            if dist == "poisson":
                score_a = _poisson_sample(off_a * variance_scalar, 1, rng=rng)[0]
                score_b = _poisson_sample(off_b * variance_scalar, 1, rng=rng)[0]
            else:
                sigma = max(1.5, 5.0 * variance_scalar)
                score_a = _normal_sample(off_a, sigma, 1, rng=rng)[0]
                score_b = _normal_sample(off_b, sigma, 1, rng=rng)[0]

            score_a = max(0.0, score_a)
            score_b = max(0.0, score_b)
            results["a_scores"].append(score_a)
            results["b_scores"].append(score_b)

            if score_a > score_b:
                results["team_a_wins"] += 1
            elif score_b > score_a:
                results["team_b_wins"] += 1

    results["true_prob_a"] = results["team_a_wins"] / n_iter
    results["true_prob_b"] = results["team_b_wins"] / n_iter
    return results


def simulate_totals(
    mean: float,
    variance: float,
    market_total: float,
    dist: str,
    n_iter: int = 10000,
) -> dict:
    """Simulates totals (over/under) with explicit distribution selection."""
    sigma = max(0.1, variance**0.5)
    samples = (
        _normal_sample(mean, sigma, n_iter) if dist == "normal" else _poisson_sample(mean, n_iter)
    )

    over_hits = sum(1 for x in samples if x > market_total)
    under_hits = sum(1 for x in samples if x < market_total)
    push_hits = sum(1 for x in samples if abs(x - market_total) < 0.5)

    return {
        "over_prob": over_hits / n_iter,
        "under_prob": under_hits / n_iter,
        "push_prob": push_hits / n_iter,
        "mean": sum(samples) / n_iter,
        "std": sigma,
    }


def simulate_totals_auto(
    mean: float,
    variance: float,
    market_total: float,
    metric_key: str,
    league: str,
    n_iter: int = 10000,
    seed: int | None = None,
) -> dict:
    """Automatically selects distribution and simulates totals."""
    if seed is not None:
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)
    dist = select_distribution(metric_key, league, mean=mean)
    return simulate_totals(mean, variance, market_total, dist, n_iter)


def run_player_simulation(
    player_proj: dict,
    n_iter: int = 10000,
    seed: int | None = None,
) -> dict:
    """Simulates a single player stat vs a market line."""
    if np is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = random.Random(seed)

    league = player_proj.get("league", "NBA").upper()
    stat_key = player_proj.get("stat_key", "pts")
    mean = player_proj.get("mean", 0.0)
    variance = player_proj.get("variance", 1.0)
    market_line = player_proj.get("market_line", mean)
    distribution_override = player_proj.get("distribution")
    dud_prob = player_proj.get("dud_prob", 0.0)

    dist = select_distribution(stat_key, league, mean=mean, override=distribution_override)
    sigma = max(0.1, variance**0.5)

    if np is not None and isinstance(rng, np.random.Generator):
        if dud_prob > 0.0:
            dud_mask = rng.binomial(1, dud_prob, size=n_iter)
            if dist == "poisson":
                base_stats = rng.poisson(max(0.01, mean), size=n_iter)
            else:
                base_stats = rng.normal(mean, sigma, size=n_iter)
            samples = np.where(dud_mask == 1, 0.0, base_stats).tolist()
        else:
            if dist == "poisson":
                samples = rng.poisson(max(0.01, mean), size=n_iter).tolist()
            else:
                samples = rng.normal(mean, sigma, size=n_iter).tolist()
    else:
        if dist == "poisson":
            samples = _poisson_sample(mean, n_iter, rng=rng)
        else:
            samples = _normal_sample(mean, sigma, n_iter, rng=rng)

        if dud_prob > 0.0:
            def _rand():
                if rng is not None:
                    return rng.random()
                return random.random()
            samples = [0.0 if (_rand() < dud_prob) else x for x in samples]

    over_hits = sum(1 for x in samples if x > market_line)
    under_hits = sum(1 for x in samples if x < market_line)
    push_hits = sum(1 for x in samples if abs(x - market_line) < 0.5)

    sample_mean = sum(samples) / n_iter
    sample_variance = sum((x - sample_mean) ** 2 for x in samples) / n_iter
    sample_std = sample_variance**0.5
    sorted_samples = sorted(float(x) for x in samples)

    return {
        "over_prob": over_hits / n_iter,
        "under_prob": under_hits / n_iter,
        "push_prob": push_hits / n_iter,
        "mean": sample_mean,
        "std": sample_std,
        "p10": _percentile(sorted_samples, 0.10),
        "p50": _percentile(sorted_samples, 0.50),
        "p90": _percentile(sorted_samples, 0.90),
        "distribution_type": dist,
        "distribution_params": (
            {"lambda": float(mean)}
            if dist == "poisson"
            else {"mu": float(mean), "sigma": float(sigma)}
        ),
        "samples": samples[:100] if len(samples) > 100 else samples,
    }


# ---------------------------------------------------------------------------
# League-average archetype defaults (for calibration-eligible fallback)
# ---------------------------------------------------------------------------


def _archetype_league_defaults(league: str) -> dict:
    """Return league-average context so game analyses produce calibration-eligible
    predictions even when the caller omits home_context / away_context.

    These are intentionally coarse — accuracy degrades to a coin-flip baseline,
    but the resulting trace is persistable and contributes a calibration pair
    once the outcome is known. The caller should supply real context whenever
    possible to improve accuracy.
    """
    from omega.core.config.leagues import get_league_config
    config = get_league_config(league)
    archetype = get_archetype(league)
    if archetype is None:
        return {}
    name = archetype.name
    if name == "basketball":
        pace = config.get("avg_pace", 100.0)
        # off_rating = points scored per 100 possessions; league avg ~110
        avg_off = config.get("avg_total", 224.0) / 2.0
        return {"off_rating": avg_off, "def_rating": avg_off, "pace": pace}
    if name == "american_football":
        avg = config.get("avg_total", 45.0) / 2.0
        return {"off_rating": avg, "def_rating": avg}
    if name == "baseball":
        avg = config.get("avg_total", 8.5) / 2.0
        return {"off_rating": avg, "def_rating": avg}
    if name == "hockey":
        avg = config.get("avg_total", 5.5) / 2.0
        return {"off_rating": avg, "def_rating": avg}
    if name == "soccer":
        avg = config.get("avg_total", 2.5) / 2.0
        return {"off_rating": avg, "def_rating": avg}
    if name == "tennis":
        return {"serve_win_pct": 0.62, "return_win_pct": 0.38}
    if name == "golf":
        return {"strokes_gained_total": 0.0}
    if name == "fighting":
        return {"win_pct": 0.5, "finish_rate": 0.5}
    if name == "esports":
        return {"map_win_rate": 0.5, "recent_form": 0.5}
    return {}


# ---------------------------------------------------------------------------
# Archetype-specific simulation models
# ---------------------------------------------------------------------------


def _sim_basketball(
    home_ctx: dict, away_ctx: dict, league: str, n_iter: int, config: dict, rng: np.random.Generator | random.Random | None = None
) -> tuple:
    """Basketball: ORtg/DRtg/pace possession model (Normal distribution)."""
    home_off = home_ctx.get("off_rating", 110.0)
    home_def = home_ctx.get("def_rating", 110.0)
    home_pace = home_ctx.get("pace", config.get("avg_pace", 100.0))

    away_off = away_ctx.get("off_rating", 110.0)
    away_def = away_ctx.get("def_rating", 110.0)
    away_pace = away_ctx.get("pace", config.get("avg_pace", 100.0))

    game_pace = (home_pace + away_pace) / 2.0
    league_avg_pace = config.get("avg_pace", 100.0)
    std = config.get("std", 12.0)

    # ORtg/DRtg score model. def_rating is points allowed per 100 possessions;
    # lower opposing defense reduces expected scoring.
    league_avg_rating = config.get("avg_total", 220.0) / 2.0
    pace_factor = game_pace / league_avg_pace
    home_expected = _expected_against_allowed_rate(
        home_off, away_def, league_avg_rating, pace_factor
    )
    away_expected = _expected_against_allowed_rate(
        away_off, home_def, league_avg_rating, pace_factor
    )

    # Home court advantage
    hca = config.get("home_advantage", 3.0)
    home_expected += hca / 2.0
    away_expected -= hca / 2.0

    home_scores = _normal_sample(home_expected, std, n_iter, rng=rng)
    away_scores = _normal_sample(away_expected, std, n_iter, rng=rng)
    home_scores = [max(0, s) for s in home_scores]
    away_scores = [max(0, s) for s in away_scores]
    return home_scores, away_scores


def _sim_american_football(
    home_ctx: dict, away_ctx: dict, league: str, n_iter: int, config: dict, rng: np.random.Generator | random.Random | None = None
) -> tuple:
    """American Football: (PPG + opp PAPG) / 2 with Normal distribution."""
    home_off = home_ctx.get("off_rating", config.get("avg_total", 45.0) / 2)
    home_def = home_ctx.get("def_rating", config.get("avg_total", 45.0) / 2)
    away_off = away_ctx.get("off_rating", config.get("avg_total", 45.0) / 2)
    away_def = away_ctx.get("def_rating", config.get("avg_total", 45.0) / 2)

    # off_rating = points_per_game, def_rating = points_allowed_per_game
    home_expected = (home_off + away_def) / 2.0
    away_expected = (away_off + home_def) / 2.0

    hca = config.get("home_advantage", 2.5)
    home_expected += hca / 2.0
    away_expected -= hca / 2.0

    std = config.get("std", 10.0)
    home_scores = _normal_sample(home_expected, std, n_iter, rng=rng)
    away_scores = _normal_sample(away_expected, std, n_iter, rng=rng)
    home_scores = [max(0, s) for s in home_scores]
    away_scores = [max(0, s) for s in away_scores]
    return home_scores, away_scores


def _sim_baseball(home_ctx: dict, away_ctx: dict, league: str, n_iter: int, config: dict, rng: np.random.Generator | random.Random | None = None) -> tuple:
    """Baseball: Poisson run environment model.

    off_rating = runs scored per game, def_rating = runs allowed per game.
    Expected runs = (team_off * opp_runs_allowed / league_avg) adjusted for park factor.
    """
    league_avg_rpg = config.get("avg_total", 8.5) / 2.0  # ~4.25

    home_off = home_ctx.get("off_rating", league_avg_rpg)
    home_def = home_ctx.get("def_rating", league_avg_rpg)
    away_off = away_ctx.get("off_rating", league_avg_rpg)
    away_def = away_ctx.get("def_rating", league_avg_rpg)

    park_factor = home_ctx.get("park_factor", 1.0)

    # Pitcher adjustments: if starter ERA is available, blend with team rate
    home_starter_era = home_ctx.get("starter_era")
    away_starter_era = away_ctx.get("starter_era")

    # Expected runs for each team. def_rating is runs allowed per game; lower
    # opposing defense reduces expected scoring.
    home_lambda = _expected_against_allowed_rate(home_off, away_def, league_avg_rpg, park_factor)
    away_lambda = _expected_against_allowed_rate(away_off, home_def, league_avg_rpg, park_factor)

    # Pitcher ERA adjustment: blend with 40% weight toward starter quality
    if away_starter_era is not None and away_starter_era > 0:
        pitcher_factor = away_starter_era / (league_avg_rpg * 9 / 9)  # ERA relative to league
        home_lambda = home_lambda * 0.6 + (home_lambda * pitcher_factor) * 0.4

    if home_starter_era is not None and home_starter_era > 0:
        pitcher_factor = home_starter_era / (league_avg_rpg * 9 / 9)
        away_lambda = away_lambda * 0.6 + (away_lambda * pitcher_factor) * 0.4

    hca = config.get("home_advantage", 0.3)
    home_lambda += hca / 2.0
    away_lambda -= hca / 2.0

    home_lambda = max(0.5, home_lambda)
    away_lambda = max(0.5, away_lambda)

    home_scores = _poisson_sample(home_lambda, n_iter, rng=rng)
    away_scores = _poisson_sample(away_lambda, n_iter, rng=rng)
    return home_scores, away_scores


def _sim_hockey(home_ctx: dict, away_ctx: dict, league: str, n_iter: int, config: dict, rng: np.random.Generator | random.Random | None = None) -> tuple:
    """Hockey: Poisson goal model with goalie/shot-rate adjustments.

    off_rating = goals per game, def_rating = goals allowed per game.
    Goalie save percentage adjusts expected goals against.
    """
    league_avg_gpg = config.get("avg_total", 6.0) / 2.0  # ~3.0

    home_off = home_ctx.get("off_rating", league_avg_gpg)
    home_def = home_ctx.get("def_rating", league_avg_gpg)
    away_off = away_ctx.get("off_rating", league_avg_gpg)
    away_def = away_ctx.get("def_rating", league_avg_gpg)

    home_lambda = _expected_against_allowed_rate(home_off, away_def, league_avg_gpg)
    away_lambda = _expected_against_allowed_rate(away_off, home_def, league_avg_gpg)

    # Goalie save pct adjustment: if away goalie is elite, reduce home goals
    away_goalie_sv = away_ctx.get("goalie_sv_pct")
    home_goalie_sv = home_ctx.get("goalie_sv_pct")
    league_avg_sv = 0.905

    if away_goalie_sv and away_goalie_sv > 0:
        sv_factor = (1 - away_goalie_sv) / (1 - league_avg_sv)
        home_lambda *= sv_factor

    if home_goalie_sv and home_goalie_sv > 0:
        sv_factor = (1 - home_goalie_sv) / (1 - league_avg_sv)
        away_lambda *= sv_factor

    # Special teams adjustment
    home_pp = home_ctx.get("pp_pct", 0.20)
    away_pp = away_ctx.get("pp_pct", 0.20)

    # ~3.5 power plays per game average; adjust goal expectation
    pp_goals_home = 3.5 * (home_pp - 0.20)
    pp_goals_away = 3.5 * (away_pp - 0.20)
    home_lambda += pp_goals_home * 0.5
    away_lambda += pp_goals_away * 0.5

    hca = config.get("home_advantage", 0.2)
    home_lambda += hca / 2.0
    away_lambda -= hca / 2.0

    home_lambda = max(0.3, home_lambda)
    away_lambda = max(0.3, away_lambda)

    home_scores = _poisson_sample(home_lambda, n_iter, rng=rng)
    away_scores = _poisson_sample(away_lambda, n_iter, rng=rng)
    return home_scores, away_scores


# Dixon-Coles low-score correction defaults. ``rho`` is the dependence
# parameter on the {0-0, 1-0, 0-1, 1-1} cells; negative values shift mass toward
# 0-0/1-1 (more draws) and away from 1-0/0-1, matching observed soccer scorelines.
# Correction is opt-in per league via the ``dixon_coles`` config flag so existing
# independent-Poisson traces remain reproducible on the legacy path.
_SOCCER_DC_RHO_DEFAULT = -0.13
_SOCCER_DIXON_COLES_DEFAULT = False
_SOCCER_DC_MAX_GOALS = 15


def _dixon_coles_scores(
    home_lambda: float,
    away_lambda: float,
    rho: float,
    n_iter: int,
    max_goals: int = _SOCCER_DC_MAX_GOALS,
    rng: np.random.Generator | random.Random | None = None,
) -> tuple[list[int], list[int]]:
    """Sample correlated soccer scorelines via a Dixon-Coles adjusted joint pmf.

    Builds the independent-Poisson joint over a truncated ``max_goals`` grid,
    applies the Dixon-Coles tau correction to the four low-score cells,
    renormalizes, and inverse-CDF samples with a single vectorized uniform
    draw. Using one ``np.random.random`` call after the engine's global seed is
    set keeps reruns with the same seed bit-identical (frozen-artifact
    invariant).
    """
    import math

    ks = range(max_goals + 1)
    h_pmf = np.array(
        [math.exp(-home_lambda) * home_lambda**k / math.factorial(k) for k in ks]
    )
    a_pmf = np.array(
        [math.exp(-away_lambda) * away_lambda**k / math.factorial(k) for k in ks]
    )
    joint = np.outer(h_pmf, a_pmf)  # joint[i, j] = P(home=i) * P(away=j)

    # Dixon-Coles tau correction (x = home goals, y = away goals).
    joint[0, 0] *= 1.0 - home_lambda * away_lambda * rho
    joint[0, 1] *= 1.0 + home_lambda * rho
    joint[1, 0] *= 1.0 + away_lambda * rho
    joint[1, 1] *= 1.0 - rho
    joint = np.clip(joint, 0.0, None)

    total = float(joint.sum())
    if total <= 0.0:
        # Degenerate correction (extreme rho) — fall back to independent draws.
        return (
            [int(s) for s in _poisson_sample(home_lambda, n_iter, rng=rng)],
            [int(s) for s in _poisson_sample(away_lambda, n_iter, rng=rng)],
        )

    cdf = np.cumsum((joint / total).ravel())
    if rng is not None and isinstance(rng, np.random.Generator):
        u = rng.random(n_iter)
    else:
        u = np.random.random(n_iter)
    idx = np.clip(np.searchsorted(cdf, u, side="right"), 0, cdf.size - 1)
    width = max_goals + 1
    home_scores = (idx // width).astype(int).tolist()
    away_scores = (idx % width).astype(int).tolist()
    return home_scores, away_scores


def _sim_soccer(
    home_ctx: dict,
    away_ctx: dict,
    league: str,
    n_iter: int,
    config: dict,
    rng: np.random.Generator | random.Random | None = None,
) -> tuple:
    """Soccer: Poisson goal model with xG integration.

    off_rating = goals per game (or xG), def_rating = goals conceded per game (or xGA).

    When the league config sets ``dixon_coles: True``, low-score correlation is
    modelled via a Dixon-Coles tau correction (see :func:`_dixon_coles_scores`);
    otherwise home and away goals are sampled independently.
    """
    league_avg_gpg = config.get("avg_total", 2.5) / 2.0  # ~1.25

    home_off = home_ctx.get("off_rating", league_avg_gpg)
    home_def = home_ctx.get("def_rating", league_avg_gpg)
    away_off = away_ctx.get("off_rating", league_avg_gpg)
    away_def = away_ctx.get("def_rating", league_avg_gpg)

    # Prefer xG if available
    home_xg = home_ctx.get("xg_for", home_off)
    away_xg = away_ctx.get("xg_for", away_off)
    home_xga = home_ctx.get("xg_against", home_def)
    away_xga = away_ctx.get("xg_against", away_def)

    home_lambda = _expected_against_allowed_rate(home_xg, away_xga, league_avg_gpg)
    away_lambda = _expected_against_allowed_rate(away_xg, home_xga, league_avg_gpg)

    hca = config.get("home_advantage", 0.3)
    home_lambda += hca / 2.0
    away_lambda -= hca / 2.0

    home_lambda = max(0.2, home_lambda)
    away_lambda = max(0.2, away_lambda)

    use_dc = config.get("dixon_coles", _SOCCER_DIXON_COLES_DEFAULT)
    if use_dc and np is not None:
        rho = config.get("rho", _SOCCER_DC_RHO_DEFAULT)
        return _dixon_coles_scores(home_lambda, away_lambda, rho, n_iter, rng=rng)

    home_scores = _poisson_sample(home_lambda, n_iter, rng=rng)
    away_scores = _poisson_sample(away_lambda, n_iter, rng=rng)
    return home_scores, away_scores


def _sim_tennis(home_ctx: dict, away_ctx: dict, league: str, n_iter: int, config: dict, rng: np.random.Generator | random.Random | None = None) -> tuple:
    """Tennis: Point-level serve/return probability → simulate sets.

    home = Player A (listed first / higher seed), away = Player B.
    serve_win_pct: probability of winning a point on own serve.
    return_win_pct: probability of winning a point on opponent's serve.
    """
    # Player A serve/return win rates
    a_serve = home_ctx.get("serve_win_pct", 0.64)
    a_return = home_ctx.get("return_win_pct", 0.36)
    # Player B
    b_serve = away_ctx.get("serve_win_pct", 0.62)
    b_return = away_ctx.get("return_win_pct", 0.34)

    # Combine: A's probability of winning point when A serves
    # = average of (A serve%) and (1 - B return%)
    p_a_serve = (a_serve + (1 - b_return)) / 2.0
    p_b_serve = (b_serve + (1 - a_return)) / 2.0

    best_of = config.get("best_of", 3)
    sets_to_win = (best_of // 2) + 1

    a_match_wins = 0
    b_match_wins = 0
    a_total_sets = []
    b_total_sets = []

    for _ in range(n_iter):
        a_sets, b_sets = 0, 0
        total_games = 0
        while a_sets < sets_to_win and b_sets < sets_to_win:
            # Simulate a set
            a_games, b_games = _simulate_tennis_set(p_a_serve, p_b_serve, rng=rng)
            total_games += a_games + b_games
            if a_games > b_games:
                a_sets += 1
            else:
                b_sets += 1

        a_total_sets.append(a_sets)
        b_total_sets.append(b_sets)
        if a_sets > b_sets:
            a_match_wins += 1
        else:
            b_match_wins += 1

    # Convert to "score" format: sets won
    return (
        [float(s) for s in a_total_sets],
        [float(s) for s in b_total_sets],
    )


def _simulate_tennis_set(p_a_serve: float, p_b_serve: float, rng: np.random.Generator | random.Random | None = None) -> tuple:
    """Simulate a single tennis set. Returns (a_games, b_games)."""
    a_games, b_games = 0, 0
    # Alternate serve: A serves first
    server_is_a = True

    def _rand():
        if rng is not None:
            return rng.random()
        return random.random()

    while True:
        # Check for set win (6-x with 2+ lead, or tiebreak at 6-6)
        if a_games >= 6 and a_games - b_games >= 2:
            return a_games, b_games
        if b_games >= 6 and b_games - a_games >= 2:
            return a_games, b_games
        if a_games == 6 and b_games == 6:
            # Tiebreak
            if _rand() < (p_a_serve + (1 - p_b_serve)) / 2.0:
                return 7, 6
            else:
                return 6, 7

        # Simulate a game: server wins with p_serve probability
        # A game is ~4 points; serve win % maps to game win % via deuce model
        p_serve = p_a_serve if server_is_a else p_b_serve
        game_win_prob = _tennis_game_win_prob(p_serve)

        if _rand() < game_win_prob:
            # Server wins the game
            if server_is_a:
                a_games += 1
            else:
                b_games += 1
        else:
            # Returner breaks
            if server_is_a:
                b_games += 1
            else:
                a_games += 1

        server_is_a = not server_is_a


def _tennis_game_win_prob(p: float) -> float:
    """Probability that server wins a game given point-win probability p.

    Uses the exact formula accounting for deuce:
    P(win game) = p^4 * (15 - 4p - (10p^2)/(1 - 2p(1-p))) ... simplified via
    standard game-tree calculation.
    """
    p = max(0.01, min(0.99, p))
    q = 1 - p
    # Prob of reaching deuce (3-3 in points) = C(6,3) * p^3 * q^3 = 20 * p^3 * q^3
    # Prob server wins from deuce = p^2 / (p^2 + q^2)
    p_deuce_win = (p * p) / (p * p + q * q)
    p_reach_deuce = 20 * (p**3) * (q**3)

    # Prob server wins before deuce (4-0, 4-1, 4-2)
    p_win_0 = p**4
    p_win_1 = 4 * (p**4) * q
    p_win_2 = 10 * (p**4) * (q**2)

    return p_win_0 + p_win_1 + p_win_2 + p_reach_deuce * p_deuce_win


def _sim_golf(home_ctx: dict, away_ctx: dict, league: str, n_iter: int, config: dict, rng: np.random.Generator | random.Random | None = None) -> tuple:
    """Golf: Strokes-gained field probability model.

    For head-to-head matchup betting, we simulate 4-round tournament scores
    for two golfers using their strokes-gained total (SG:Total) as the
    primary input.  Lower score wins.

    home_ctx = Golfer A, away_ctx = Golfer B.
    off_rating = strokes_gained_total (positive = better than field average).
    """
    n_rounds = config.get("rounds", 4)
    round_std = config.get("round_std", 3.0)

    # SG:Total is strokes better than field average per round
    sg_a = home_ctx.get("strokes_gained_total", home_ctx.get("off_rating", 0.0))
    sg_b = away_ctx.get("strokes_gained_total", away_ctx.get("off_rating", 0.0))

    # Par 72 baseline; SG adjusts expected score
    par = 72.0
    a_per_round = par - sg_a
    b_per_round = par - sg_b

    a_totals = []
    b_totals = []

    if np is not None and isinstance(rng, np.random.Generator):
        a_draws = rng.normal(a_per_round, round_std, size=(n_iter, n_rounds))
        b_draws = rng.normal(b_per_round, round_std, size=(n_iter, n_rounds))
        a_totals = np.sum(a_draws, axis=1).tolist()
        b_totals = np.sum(b_draws, axis=1).tolist()
    else:
        for _ in range(n_iter):
            a_score = sum(_normal_sample(a_per_round, round_std, 1, rng=rng)[0] for _ in range(n_rounds))
            b_score = sum(_normal_sample(b_per_round, round_std, 1, rng=rng)[0] for _ in range(n_rounds))
            a_totals.append(a_score)
            b_totals.append(b_score)

    # In golf, lower is better. Invert for the standard result builder:
    # "home_win" = golfer A wins = golfer A has lower total
    # We store actual scores so the result builder counts correctly
    # But we need to flip the comparison: A wins when A < B
    # So we negate scores for the result builder
    return ([-s for s in a_totals], [-s for s in b_totals])


def _sim_fighting(home_ctx: dict, away_ctx: dict, league: str, n_iter: int, config: dict, rng: np.random.Generator | random.Random | None = None) -> tuple:
    """Fighting: Win probability with method-of-victory modeling.

    off_rating = win percentage (0-1), finish_rate = rate of finishes.
    Returns scores as 1 (win) or 0 (loss) for each iteration, plus
    method-of-victory data in a side channel (stored in the context dicts).

    home = Fighter A, away = Fighter B.
    """
    a_win_pct = home_ctx.get("win_pct", 0.5)
    b_win_pct = away_ctx.get("win_pct", 0.5)
    a_ko = home_ctx.get("ko_tko_rate", 0.3)
    b_ko = away_ctx.get("ko_tko_rate", 0.3)
    a_sub = home_ctx.get("submission_rate", 0.15)
    b_sub = away_ctx.get("submission_rate", 0.15)

    # Implied win probability from both records (normalize)
    total = a_win_pct + b_win_pct
    if total > 0:
        p_a = a_win_pct / total
    else:
        p_a = 0.5

    # Elo/rating override if available
    a_elo = home_ctx.get("elo_rating")
    b_elo = away_ctx.get("elo_rating")
    if a_elo and b_elo:
        p_a = 1.0 / (1.0 + 10 ** ((b_elo - a_elo) / 400.0))

    a_scores = []
    b_scores = []
    method_counts = {"ko_tko": 0, "submission": 0, "decision": 0, "draw": 0}

    def _rand():
        if rng is not None:
            return rng.random()
        return random.random()

    for _ in range(n_iter):
        if _rand() < p_a:
            a_scores.append(1.0)
            b_scores.append(0.0)
            # Method of victory for fighter A
            r = _rand()
            if r < a_ko:
                method_counts["ko_tko"] += 1
            elif r < a_ko + a_sub:
                method_counts["submission"] += 1
            else:
                method_counts["decision"] += 1
        else:
            a_scores.append(0.0)
            b_scores.append(1.0)
            r = _rand()
            if r < b_ko:
                method_counts["ko_tko"] += 1
            elif r < b_ko + b_sub:
                method_counts["submission"] += 1
            else:
                method_counts["decision"] += 1

    # Draw probability in boxing is ~2-3%, negligible in MMA
    is_boxing = league.upper() == "BOXING"
    draw_rate = 0.025 if is_boxing else 0.005
    # Retroactively convert some decisions to draws
    for i in range(len(a_scores)):
        if _rand() < draw_rate:
            a_scores[i] = 0.5
            b_scores[i] = 0.5
            method_counts["draw"] += 1
            method_counts["decision"] -= 1

    return a_scores, b_scores


def _sim_esports(home_ctx: dict, away_ctx: dict, league: str, n_iter: int, config: dict, rng: np.random.Generator | random.Random | None = None) -> tuple:
    """Esports: Map win probability with best-of-N simulation.

    map_win_rate: team's overall map win rate (0-1).
    recent_form: recent performance modifier.
    """
    a_map_wr = home_ctx.get("map_win_rate", 0.5)
    b_map_wr = away_ctx.get("map_win_rate", 0.5)

    # Derive head-to-head map win probability
    total = a_map_wr + b_map_wr
    if total > 0:
        p_a_map = a_map_wr / total
    else:
        p_a_map = 0.5

    # Elo override if available
    a_elo = home_ctx.get("elo_rating")
    b_elo = away_ctx.get("elo_rating")
    if a_elo and b_elo:
        p_a_map = 1.0 / (1.0 + 10 ** ((b_elo - a_elo) / 400.0))

    best_of = config.get("best_of", 3)
    maps_to_win = (best_of // 2) + 1

    a_total_maps = []
    b_total_maps = []

    def _rand():
        if rng is not None:
            return rng.random()
        return random.random()

    for _ in range(n_iter):
        a_maps, b_maps = 0, 0
        while a_maps < maps_to_win and b_maps < maps_to_win:
            if _rand() < p_a_map:
                a_maps += 1
            else:
                b_maps += 1
        a_total_maps.append(float(a_maps))
        b_total_maps.append(float(b_maps))

    return a_total_maps, b_total_maps


# ---------------------------------------------------------------------------
# Archetype dispatch table
# ---------------------------------------------------------------------------

_ARCHETYPE_SIMULATORS = {
    "basketball": _sim_basketball,
    "american_football": _sim_american_football,
    "baseball": _sim_baseball,
    "hockey": _sim_hockey,
    "soccer": _sim_soccer,
    "tennis": _sim_tennis,
    "golf": _sim_golf,
    "fighting": _sim_fighting,
    "esports": _sim_esports,
}


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------


class FastScoreSimulationBackend:
    """Current normal/Poisson archetype simulator behind the backend contract."""

    backend_name = "fast_score"
    component_version = "fast_score_v1"
    evidence_mode = "plane_adjustment"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        if np is not None:
            rng = np.random.default_rng(request.seed)
        else:
            rng = random.Random(request.seed)

        league = request.league.upper()
        archetype = get_archetype(league)
        archetype_name = get_archetype_name(league)
        config = get_league_config(league)

        if archetype is None or archetype_name is None:
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason=(
                    f"No simulation model for league '{league}'. Add it to "
                    "LEAGUE_TO_ARCHETYPE in sport_archetypes.py."
                ),
                missing_requirements=["league_model"],
            )

        home_context = request.home_context
        away_context = request.away_context
        baseline_used = False
        context_source = "provided"
        if home_context is None or away_context is None:
            if not request.allow_baseline:
                missing = []
                if home_context is None:
                    missing.extend(f"home_context.{k}" for k in archetype.required_team_keys)
                if away_context is None:
                    missing.extend(f"away_context.{k}" for k in archetype.required_team_keys)
                return _skip_result(
                    request.home_team,
                    request.away_team,
                    league,
                    skip_reason=(
                        "Missing home_context or away_context; league-average "
                        "baseline requires allow_baseline=True"
                    ),
                    missing_requirements=missing,
                )
            defaults = _archetype_league_defaults(league)
            if defaults:
                baseline_used = True
                context_source = "league_default"
                if home_context is None:
                    home_context = dict(defaults)
                if away_context is None:
                    away_context = dict(defaults)

        required = archetype.required_team_keys
        home_missing = _validate_required_keys(home_context, "home", required, league)
        away_missing = _validate_required_keys(away_context, "away", required, league)
        all_missing = home_missing + away_missing
        if all_missing:
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason=(
                    f"Missing required inputs for {archetype.display_name}: "
                    f"{', '.join(all_missing)}"
                ),
                missing_requirements=all_missing,
            )

        simulator = _ARCHETYPE_SIMULATORS.get(archetype_name)
        if simulator is None:
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason=f"Simulator not implemented for archetype '{archetype_name}'",
                missing_requirements=["archetype_simulator"],
            )

        assert home_context is not None
        assert away_context is not None
        home_scores, away_scores = simulator(
            home_context,
            away_context,
            league,
            request.n_iterations,
            config,
            rng=rng,
        )

        return _build_team_score_result(
            request.home_team,
            request.away_team,
            league,
            request.n_iterations,
            home_scores,
            away_scores,
            home_context=home_context,
            away_context=away_context,
            archetype_name=archetype_name,
            spread_home=request.spread_home,
            over_under=request.over_under,
            context_source=context_source,
            baseline_used=baseline_used,
            seed=request.seed,
            backend_name=self.backend_name,
            component_version=self.component_version,
        )


def run_markov_game_simulation(
    request: GameSimulationInput,
    *,
    backend_name: str,
    component_version: str,
    context_source: str = "provided",
    baseline_used: bool = False,
) -> dict[str, Any]:
    """Shared Markov team-score simulation body.

    Owns the possession-level Markov run used by both the generic
    ``markov_state`` backend and sport-tuned variants (WNBA, etc.). Keeping the
    body here — rather than duplicating it per backend — preserves a single
    calibration path (CLAUDE.md hard rule against parallel pipelines). Callers
    pass their own ``backend_name`` / ``component_version`` for provenance and
    set ``context_source`` / ``baseline_used`` when they injected league
    defaults into the contexts before calling.
    """
    if request.seed is not None:
        random.seed(request.seed)

    league = request.league.upper()
    archetype = get_archetype(league)
    archetype_name = get_archetype_name(league)
    if archetype is None or archetype_name is None:
        return _skip_result(
            request.home_team,
            request.away_team,
            league,
            skip_reason=f"No Markov simulation model for league '{league}'.",
            missing_requirements=["league_model"],
        )
    if archetype.result_type != "team_score":
        return _skip_result(
            request.home_team,
            request.away_team,
            league,
            skip_reason=f"Markov backend only supports team_score archetypes, got {archetype.result_type}.",
            missing_requirements=["team_score_archetype"],
        )

    if request.home_context is None or request.away_context is None:
        return _skip_result(
            request.home_team,
            request.away_team,
            league,
            skip_reason="Missing home_context or away_context for Markov backend",
            missing_requirements=["home_context", "away_context"],
        )

    home_missing = _validate_required_keys(
        request.home_context,
        "home",
        archetype.required_team_keys,
        league,
    )
    away_missing = _validate_required_keys(
        request.away_context,
        "away",
        archetype.required_team_keys,
        league,
    )
    all_missing = home_missing + away_missing
    if all_missing:
        return _skip_result(
            request.home_team,
            request.away_team,
            league,
            skip_reason=(
                f"Missing required inputs for Markov {archetype.display_name}: "
                f"{', '.join(all_missing)}"
            ),
            missing_requirements=all_missing,
        )

    from omega.core.simulation.markov_engine import MarkovSimulator

    simulator = MarkovSimulator(
        league=league,
        players=[],
        home_context=request.home_context,
        away_context=request.away_context,
        transition_modifiers=request.transition_modifiers,
    )
    n_possessions = simulator._base_n_possessions
    home_scores: list[float] = []
    away_scores: list[float] = []
    for _ in range(request.n_iterations):
        state = simulator.simulate_game(n_possessions)
        home_scores.append(state.home_score)
        away_scores.append(state.away_score)

    result = _build_team_score_result(
        request.home_team,
        request.away_team,
        league,
        request.n_iterations,
        home_scores,
        away_scores,
        home_context=request.home_context,
        away_context=request.away_context,
        archetype_name=archetype_name,
        spread_home=request.spread_home,
        over_under=request.over_under,
        context_source=context_source,
        baseline_used=baseline_used,
        seed=request.seed,
        backend_name=backend_name,
        component_version=component_version,
    )
    for row in result["simulation_distributions"]:
        params = dict(row.get("distribution_params") or {})
        params.update(
            {
                "source": "markov_terminal_scores",
                "base_possessions": n_possessions,
                "transition_matrix_ids": simulator.transition_matrix_ids,
                "transition_modifiers": request.transition_modifiers or {},
            }
        )
        row["distribution_type"] = "empirical_markov"
        row["distribution_params"] = params
    return result


class MarkovGameSimulationBackend:
    """Markov state-transition backend implementing the game simulation contract."""

    backend_name = "markov_state"
    component_version = "markov_state_v1"
    evidence_mode = "markov_transition"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        return run_markov_game_simulation(
            request,
            backend_name=self.backend_name,
            component_version=self.component_version,
        )


class OmegaSimulationEngine:
    """
    High-level simulation engine dispatching to sport-archetype models.
    Input-driven only — no network calls.
    """

    def __init__(self, game_backend: GameSimulationBackend | None = None):
        self._game_backend = game_backend or FastScoreSimulationBackend()

    def run_fast_game_simulation(
        self,
        home_team: str,
        away_team: str,
        league: str = "NBA",
        n_iterations: int = 100,
        home_context: dict | None = None,
        away_context: dict | None = None,
        seed: int | None = None,
        spread_home: float | None = None,
        over_under: float | None = None,
        allow_baseline: bool = False,
        transition_modifiers: dict | None = None,
        prior_payload: dict | None = None,
        backend: GameSimulationBackend | None = None,
    ) -> dict:
        """
        Run a fast game simulation using team stats dispatched by sport archetype.

        Args:
            home_team: Home team / Player A name
            away_team: Away team / Player B name
            league: League code (NBA, NFL, EPL, UFC, ATP, CS2, PGA, ...)
            n_iterations: Number of simulation iterations
            home_context: Pre-fetched home team / player A context dict
            away_context: Pre-fetched away team / player B context dict
            seed: Optional RNG seed for reproducible results
            transition_modifiers: Scalar modifiers for the Markov backend (ignored
                by fast_score). Produced by evidence_to_modifier.signals_to_transition_modifiers().
            backend: Optional per-call game backend. When supplied it overrides the
                engine's default backend, so a single shared engine can run any
                registered backend without re-instantiation or name-based dispatch.

        Returns:
            Dict with score distributions, win probabilities, and missing_requirements
        """
        active = backend or self._game_backend
        request = GameSimulationInput(
            home_team=home_team,
            away_team=away_team,
            league=league,
            n_iterations=n_iterations,
            home_context=home_context,
            away_context=away_context,
            seed=seed,
            spread_home=spread_home,
            over_under=over_under,
            allow_baseline=allow_baseline,
            transition_modifiers=transition_modifiers,
            prior_payload=prior_payload,
        )
        return enforce_game_backend_contract(active.run(request))

    def run_game_simulation(
        self,
        home_team: str,
        away_team: str,
        league: str = "NBA",
        n_iterations: int = 1000,
        home_context: dict | None = None,
        away_context: dict | None = None,
        home_players: list[dict] | None = None,
        away_players: list[dict] | None = None,
    ) -> dict:
        """
        Run a full game simulation using Markov engine + player projections.

        Architecture fix: callers must supply all context. No network calls.
        """
        if home_context is None or away_context is None:
            return _skip_result(
                home_team,
                away_team,
                league,
                skip_reason="Missing home_context or away_context (caller must supply)",
                missing_requirements=["home_context", "away_context"],
            )

        try:
            import importlib

            MarkovSimulator = importlib.import_module(
                "omega.core.simulation.markov_engine"
            ).MarkovSimulator
        except ModuleNotFoundError:
            return _skip_result(
                home_team,
                away_team,
                league,
                skip_reason=(
                    "Markov simulator module is not available in this checkout; "
                    "use run_fast_game_simulation() or the canonical analyze_game() path"
                ),
                missing_requirements=["omega.core.simulation.markov_engine"],
            )

        home_players = home_players or []
        away_players = away_players or []

        all_players = []
        for p in home_players:
            player = dict(p)
            player["team_side"] = "home"
            all_players.append(player)
        for p in away_players:
            player = dict(p)
            player["team_side"] = "away"
            all_players.append(player)

        simulator = MarkovSimulator(
            league=league,
            players=all_players,
            home_context=home_context,
            away_context=away_context,
        )

        home_scores = []
        away_scores = []
        all_player_stats: dict[str, dict[str, list]] = {}

        n_possessions = simulator._base_n_possessions

        for _ in range(n_iterations):
            game_state = simulator.simulate_game(n_possessions)
            home_scores.append(game_state.home_score)
            away_scores.append(game_state.away_score)

            for player_name, stats in game_state.player_stats.items():
                if player_name not in all_player_stats:
                    all_player_stats[player_name] = {}
                for stat_key, value in stats.items():
                    if stat_key not in all_player_stats[player_name]:
                        all_player_stats[player_name][stat_key] = []
                    all_player_stats[player_name][stat_key].append(value)

        player_projections: dict[str, dict[str, dict[str, float]]] = {}
        for player_name, stats in all_player_stats.items():
            player_projections[player_name] = {}
            for stat_key, values in stats.items():
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                player_projections[player_name][stat_key] = {
                    "mean": sum(values) / n if n > 0 else 0,
                    "std": (sum((v - sum(values) / n) ** 2 for v in values) / n) ** 0.5
                    if n > 1
                    else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "p10": sorted_vals[int(n * 0.1)] if n > 10 else min(values) if values else 0,
                    "p25": sorted_vals[int(n * 0.25)] if n > 4 else min(values) if values else 0,
                    "p50": sorted_vals[int(n * 0.5)] if n > 2 else sum(values) / n if values else 0,
                    "p75": sorted_vals[int(n * 0.75)] if n > 4 else max(values) if values else 0,
                    "p90": sorted_vals[int(n * 0.9)] if n > 10 else max(values) if values else 0,
                }

        distribution_result = _build_team_score_result(
            home_team,
            away_team,
            league,
            n_iterations,
            home_scores,
            away_scores,
            home_context=home_context,
            away_context=away_context,
            archetype_name=get_archetype_name(league),
            context_source="provided",
            baseline_used=False,
            backend_name="markov_state",
            component_version="markov_state_v1",
        )
        for row in distribution_result["simulation_distributions"]:
            params = dict(row.get("distribution_params") or {})
            params.update(
                {
                    "source": "markov_terminal_scores",
                    "base_possessions": n_possessions,
                    "transition_matrix_ids": simulator.transition_matrix_ids,
                    "transition_modifiers": {},
                }
            )
            row["distribution_type"] = "empirical_markov"
            row["distribution_params"] = params

        return {
            "success": True,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "archetype": get_archetype_name(league),
            "iterations": n_iterations,
            "home_win_prob": distribution_result["home_win_prob"],
            "away_win_prob": distribution_result["away_win_prob"],
            "draw_prob": distribution_result["draw_prob"],
            "predicted_home_score": distribution_result["predicted_home_score"],
            "predicted_away_score": distribution_result["predicted_away_score"],
            "predicted_spread": distribution_result["predicted_spread"],
            "predicted_total": distribution_result["predicted_total"],
            "missing_requirements": [],
            "home_context": home_context,
            "away_context": away_context,
            "player_projections": player_projections,
            "home_scores_sample": [round(s, 1) for s in home_scores[:20]],
            "away_scores_sample": [round(s, 1) for s in away_scores[:20]],
            "context_source": "provided",
            "baseline_used": False,
            "simulation_backend": "markov_state",
            "component_version": "markov_state_v1",
            "simulation_distributions": distribution_result["simulation_distributions"],
        }

    def run_player_prop_simulation(
        self,
        player_name: str,
        team: str,
        opponent: str,
        league: str = "NBA",
        prop_type: str = "pts",
        line: float = 20.0,
        n_iterations: int = 500,
        game_context: dict | None = None,
        player_context: dict | None = None,
    ) -> dict:
        """
        Run player prop simulation focused on a specific player's stats.
        Caller must supply game_context and player_context; engine does not fetch.
        """
        archetype = get_archetype(league)
        if archetype is None:
            return {
                "success": False,
                "skip_reason": f"No model for league '{league}'",
                "missing_requirements": ["league_model"],
                "player": player_name,
                "prop_type": prop_type,
                "line": line,
            }

        if prop_type not in archetype.prop_stat_keys:
            return {
                "success": False,
                "skip_reason": f"Prop type '{prop_type}' not supported for {archetype.display_name}. Supported: {', '.join(archetype.prop_stat_keys)}",
                "missing_requirements": [],
                "player": player_name,
                "prop_type": prop_type,
                "line": line,
            }

        if game_context is None or player_context is None:
            missing = []
            if game_context is None:
                missing.append("game_context")
            if player_context is None:
                missing.append("player_context")
            return {
                "success": False,
                "skip_reason": "Missing game_context or player_context (caller must supply; agent-only path for live data)",
                "missing_requirements": missing,
                "player": player_name,
                "prop_type": prop_type,
                "line": line,
            }

        try:
            import importlib

            MarkovSimulator = importlib.import_module(
                "omega.core.simulation.markov_engine"
            ).MarkovSimulator
        except ModuleNotFoundError:
            return {
                "success": False,
                "skip_reason": (
                    "Markov simulator module is not available in this checkout; "
                    "use run_player_simulation() or the canonical analyze_player_prop() path"
                ),
                "missing_requirements": ["omega.core.simulation.markov_engine"],
                "player": player_name,
                "prop_type": prop_type,
                "line": line,
            }

        game_ctx = game_context
        pc = player_context
        if not isinstance(pc, dict) and hasattr(pc, "to_dict"):
            pc = pc.to_dict()
        elif not isinstance(pc, dict):
            pc = {}

        home_context = game_ctx.get("home_context", {})
        away_context = game_ctx.get("away_context", {})
        home_players = game_ctx.get("home_players", [])
        away_players = game_ctx.get("away_players", [])

        all_players = []
        player_found = False

        for p in home_players:
            player = dict(p)
            player["team_side"] = "home"
            if p.get("name", "").lower() == player_name.lower():
                player_found = True
                player.update(pc)
            all_players.append(player)

        for p in away_players:
            player = dict(p)
            player["team_side"] = "away"
            if p.get("name", "").lower() == player_name.lower():
                player_found = True
                player.update(pc)
            all_players.append(player)

        if not player_found:
            entry = dict(pc)
            entry["name"] = player_name
            entry["team_side"] = "home"
            all_players.append(entry)

        simulator = MarkovSimulator(
            league=league,
            players=all_players,
            home_context=home_context,
            away_context=away_context,
        )

        stat_values = []
        n_possessions = simulator._base_n_possessions

        for _ in range(n_iterations):
            game_state = simulator.simulate_game(n_possessions)
            stat_val = game_state.get_player_stat(player_name, prop_type)
            stat_values.append(stat_val)

        if not stat_values:
            return {
                "success": False,
                "error": "No stats generated",
                "missing_requirements": [],
                "player": player_name,
                "prop_type": prop_type,
                "line": line,
            }

        sorted_vals = sorted(stat_values)
        n = len(sorted_vals)
        mean_val = sum(stat_values) / n
        std_val = (sum((v - mean_val) ** 2 for v in stat_values) / n) ** 0.5

        over_count = sum(1 for v in stat_values if v > line)
        under_count = sum(1 for v in stat_values if v < line)
        push_count = sum(1 for v in stat_values if abs(v - line) < 0.5)

        return {
            "success": True,
            "player": player_name,
            "prop_type": prop_type,
            "line": line,
            "projected_value": round(mean_val, 1),
            "std": round(std_val, 1),
            "hit_probability": round(over_count / n * 100, 1),
            "over_prob": round(over_count / n * 100, 1),
            "under_prob": round(under_count / n * 100, 1),
            "push_prob": round(push_count / n * 100, 1),
            "p10": round(sorted_vals[int(n * 0.1)], 1),
            "p25": round(sorted_vals[int(n * 0.25)], 1),
            "p50": round(sorted_vals[int(n * 0.5)], 1),
            "p75": round(sorted_vals[int(n * 0.75)], 1),
            "p90": round(sorted_vals[int(n * 0.9)], 1),
            "min": round(min(stat_values), 1),
            "max": round(max(stat_values), 1),
            "iterations": n_iterations,
            "missing_requirements": [],
        }


# ---------------------------------------------------------------------------
# Prop-simulation backend: distribution router
# ---------------------------------------------------------------------------


class PropDistributionRouterBackend:
    """Default prop backend wrapping the existing select_distribution + sampler
    logic in run_player_simulation(). Behavior is bit-identical to today's
    direct callers; this is purely a registry-compatible adapter so new prop
    backends (Negative Binomial, tennis serve) can be dispatched uniformly.
    """

    backend_name = "prop_distribution_router"
    component_version = "prop_distribution_router_v1"

    def run(self, request: PropSimulationInput) -> dict[str, Any]:
        variance = (
            request.projection_std ** 2
            if request.projection_std is not None
            else 1.0
        )
        player_proj = {
            "league": request.league,
            "stat_key": request.stat_type,
            "mean": request.projection_mean,
            "variance": variance,
            "market_line": request.line,
        }
        # Forward caller-supplied distribution override and dud probability when
        # present so registry dispatch stays bit-identical to the direct
        # run_player_simulation path in service.analyze_player_prop.
        prior = request.prior_payload or {}
        if "distribution" in prior:
            player_proj["distribution"] = prior["distribution"]
        if "dud_prob" in prior:
            player_proj["dud_prob"] = prior["dud_prob"]
        return run_player_simulation(
            player_proj, n_iter=request.n_iter, seed=request.seed
        )


# ---------------------------------------------------------------------------
# Backend registration (import-time wiring)
# ---------------------------------------------------------------------------
#
# Registering here — rather than in backends.py — keeps the registry module
# free of the concrete simulator implementations and avoids an import cycle.
# service.py imports this module, so registration is in place before dispatch.

register_game_backend("fast_score", FastScoreSimulationBackend())
register_game_backend("markov_state", MarkovGameSimulationBackend())
register_prop_backend("prop_distribution_router", PropDistributionRouterBackend())

# Phase 7 sport backends. Imported here (after run_markov_game_simulation is
# defined) so registration happens whenever the engine module loads. Each new
# sport module does a lazy import of the shared run helper inside run(), so this
# import does not create a cycle.
from omega.core.simulation.markov_wnba import (  # noqa: E402
    MarkovWNBAGameSimulationBackend,
)

register_game_backend("markov_state_wnba", MarkovWNBAGameSimulationBackend())
