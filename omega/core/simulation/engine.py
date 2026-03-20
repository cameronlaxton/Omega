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
import math
import random
from typing import Dict, List, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None

from omega.core.config.leagues import get_league_config
from omega.core.simulation.archetypes import (
    get_archetype,
    get_archetype_name,
    get_required_inputs,
    SportArchetype,
)


# ---------------------------------------------------------------------------
# Distribution samplers
# ---------------------------------------------------------------------------

def select_distribution(metric_key: str, league: str) -> str:
    """
    Selects appropriate distribution (Normal vs Poisson) based on metric and league.
    """
    league = league.upper()
    metric_key = metric_key.lower()

    if metric_key in {"run_rate", "goal_rate"} or league in {"MLB", "NHL"}:
        return "poisson"

    if league in {"NFL", "NCAAF"} and metric_key == "score":
        return "poisson"

    discrete_stats = {"goals", "td", "touchdowns", "receptions", "rec", "sog",
                      "aces", "double_faults", "kills", "deaths", "assists_esport",
                      "hrs", "stolen_bases", "saves"}
    if metric_key in discrete_stats:
        return "poisson"

    return "normal"


def _poisson_sample(lam: float, size: int) -> List[float]:
    """Generate Poisson samples."""
    lam = max(0.01, lam)
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


def _normal_sample(mu: float, sigma: float, size: int) -> List[float]:
    """Generate Normal samples."""
    sigma = max(0.1, sigma)
    if np is not None:
        return np.random.normal(mu, sigma, size).tolist()
    return [random.gauss(mu, sigma) for _ in range(size)]


def _bernoulli_sample(p: float, size: int) -> List[int]:
    """Generate Bernoulli samples (0 or 1)."""
    p = max(0.001, min(0.999, p))
    if np is not None:
        return np.random.binomial(1, p, size).tolist()
    return [1 if random.random() < p else 0 for _ in range(size)]


# ---------------------------------------------------------------------------
# Skip / missing-requirements helpers
# ---------------------------------------------------------------------------

def _skip_result(
    home_team: str,
    away_team: str,
    league: str,
    skip_reason: str,
    missing_requirements: Optional[List[str]] = None,
) -> Dict:
    """Build a standard skip response."""
    return {
        "success": False,
        "skipped": True,
        "skip_reason": skip_reason,
        "missing_requirements": missing_requirements or [],
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
    }


def _validate_required_keys(
    context: Optional[Dict], side: str, required_keys: tuple, league: str
) -> List[str]:
    """Return list of missing requirement strings for a context dict."""
    if context is None:
        return [f"{side}_context"]
    missing = []
    for key in required_keys:
        val = context.get(key)
        if val is None or (isinstance(val, (int, float)) and val == 0):
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
    home_scores: List[float],
    away_scores: List[float],
    home_context: Optional[Dict] = None,
    away_context: Optional[Dict] = None,
    archetype_name: Optional[str] = None,
) -> Dict:
    """Build a standardized result dict from team score simulations."""
    home_wins = sum(1 for h, a in zip(home_scores, away_scores) if h > a)
    away_wins = sum(1 for h, a in zip(home_scores, away_scores) if a > h)
    draws = n_iterations - home_wins - away_wins

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
        "draw_prob": round(draws / n_iterations * 100, 1),
        "predicted_home_score": round(home_mean, 1),
        "predicted_away_score": round(away_mean, 1),
        "predicted_spread": round(home_mean - away_mean, 1),
        "predicted_total": round(home_mean + away_mean, 1),
        "missing_requirements": [],
        "home_context": home_context or {},
        "away_context": away_context or {},
    }
    return result


# ---------------------------------------------------------------------------
# Legacy standalone functions (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def run_game_simulation(
    projection: Dict,
    n_iter: int = 10000,
    seed: Optional[int] = None,
    league: Optional[str] = None,
) -> Dict:
    """
    Simulates a game between two teams, returning win probabilities and score distributions.
    """
    if seed is not None:
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)

    teams = list(projection.get("off_rating", {}).keys())
    if len(teams) != 2:
        raise ValueError("Projection requires exactly two teams.")

    resolved_league = league if league is not None else projection.get("league", "NFL")
    variance_scalar = projection.get("variance_scalar", 1.0)
    team_a, team_b = teams
    off_a = projection["off_rating"][team_a]
    off_b = projection["off_rating"][team_b]
    dist = select_distribution("score", resolved_league)

    results = {"team_a_wins": 0, "team_b_wins": 0, "a_scores": [], "b_scores": []}

    for _ in range(n_iter):
        if dist == "poisson":
            score_a = _poisson_sample(off_a * variance_scalar, 1)[0]
            score_b = _poisson_sample(off_b * variance_scalar, 1)[0]
        else:
            sigma = max(1.5, 5 * variance_scalar)
            score_a = _normal_sample(off_a, sigma, 1)[0]
            score_b = _normal_sample(off_b, sigma, 1)[0]

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
) -> Dict:
    """Simulates totals (over/under) with explicit distribution selection."""
    sigma = max(0.1, variance ** 0.5)
    samples = _normal_sample(mean, sigma, n_iter) if dist == "normal" else _poisson_sample(mean, n_iter)

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
    seed: Optional[int] = None,
) -> Dict:
    """Automatically selects distribution and simulates totals."""
    if seed is not None:
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)
    dist = select_distribution(metric_key, league)
    return simulate_totals(mean, variance, market_total, dist, n_iter)


def run_player_simulation(
    player_proj: Dict,
    n_iter: int = 10000,
    seed: Optional[int] = None,
) -> Dict:
    """Simulates a single player stat vs a market line."""
    if seed is not None:
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)

    league = player_proj.get("league", "NBA").upper()
    stat_key = player_proj.get("stat_key", "pts")
    mean = player_proj.get("mean", 0.0)
    variance = player_proj.get("variance", 1.0)
    market_line = player_proj.get("market_line", mean)

    dist = select_distribution(stat_key, league)
    sigma = max(0.1, variance ** 0.5)

    if dist == "poisson":
        samples = _poisson_sample(mean, n_iter)
    else:
        samples = _normal_sample(mean, sigma, n_iter)

    over_hits = sum(1 for x in samples if x > market_line)
    under_hits = sum(1 for x in samples if x < market_line)
    push_hits = sum(1 for x in samples if abs(x - market_line) < 0.5)

    sample_mean = sum(samples) / n_iter
    sample_variance = sum((x - sample_mean) ** 2 for x in samples) / n_iter
    sample_std = sample_variance ** 0.5

    return {
        "over_prob": over_hits / n_iter,
        "under_prob": under_hits / n_iter,
        "push_prob": push_hits / n_iter,
        "mean": sample_mean,
        "std": sample_std,
        "samples": samples[:100] if len(samples) > 100 else samples,
    }


# ---------------------------------------------------------------------------
# Archetype-specific simulation models
# ---------------------------------------------------------------------------

def _sim_basketball(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
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

    # ORtg-based expected score: (team_off * (league_avg / opp_def)) * (pace / 100)
    home_expected = (home_off * (league_avg_pace / away_def)) * (game_pace / league_avg_pace)
    away_expected = (away_off * (league_avg_pace / home_def)) * (game_pace / league_avg_pace)

    # Home court advantage
    hca = config.get("home_advantage", 3.0)
    home_expected += hca / 2.0
    away_expected -= hca / 2.0

    home_scores = _normal_sample(home_expected, std, n_iter)
    away_scores = _normal_sample(away_expected, std, n_iter)
    home_scores = [max(0, s) for s in home_scores]
    away_scores = [max(0, s) for s in away_scores]
    return home_scores, away_scores


def _sim_american_football(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
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
    home_scores = _normal_sample(home_expected, std, n_iter)
    away_scores = _normal_sample(away_expected, std, n_iter)
    home_scores = [max(0, s) for s in home_scores]
    away_scores = [max(0, s) for s in away_scores]
    return home_scores, away_scores


def _sim_baseball(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
) -> tuple:
    """Baseball: Poisson run environment model.

    off_rating = runs scored per game, def_rating = runs allowed per game.
    Expected runs = (team_off * league_avg / opp_def) adjusted for park factor.
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

    # Expected runs for each team
    home_lambda = (home_off * (league_avg_rpg / away_def)) * park_factor
    away_lambda = (away_off * (league_avg_rpg / home_def)) * park_factor

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

    home_scores = _poisson_sample(home_lambda, n_iter)
    away_scores = _poisson_sample(away_lambda, n_iter)
    return home_scores, away_scores


def _sim_hockey(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
) -> tuple:
    """Hockey: Poisson goal model with goalie/shot-rate adjustments.

    off_rating = goals per game, def_rating = goals allowed per game.
    Goalie save percentage adjusts expected goals against.
    """
    league_avg_gpg = config.get("avg_total", 6.0) / 2.0  # ~3.0

    home_off = home_ctx.get("off_rating", league_avg_gpg)
    home_def = home_ctx.get("def_rating", league_avg_gpg)
    away_off = away_ctx.get("off_rating", league_avg_gpg)
    away_def = away_ctx.get("def_rating", league_avg_gpg)

    home_lambda = home_off * (league_avg_gpg / away_def)
    away_lambda = away_off * (league_avg_gpg / home_def)

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
    away_pk = away_ctx.get("pk_pct", 0.80)
    away_pp = away_ctx.get("pp_pct", 0.20)
    home_pk = home_ctx.get("pk_pct", 0.80)

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

    home_scores = _poisson_sample(home_lambda, n_iter)
    away_scores = _poisson_sample(away_lambda, n_iter)
    return home_scores, away_scores


def _sim_soccer(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
) -> tuple:
    """Soccer: Poisson goal model with xG integration.

    off_rating = goals per game (or xG), def_rating = goals conceded per game (or xGA).
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

    home_lambda = (home_xg * (league_avg_gpg / away_xga)) if away_xga > 0 else home_xg
    away_lambda = (away_xg * (league_avg_gpg / home_xga)) if home_xga > 0 else away_xg

    hca = config.get("home_advantage", 0.3)
    home_lambda += hca / 2.0
    away_lambda -= hca / 2.0

    home_lambda = max(0.2, home_lambda)
    away_lambda = max(0.2, away_lambda)

    home_scores = _poisson_sample(home_lambda, n_iter)
    away_scores = _poisson_sample(away_lambda, n_iter)
    return home_scores, away_scores


def _sim_tennis(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
) -> tuple:
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
            a_games, b_games = _simulate_tennis_set(p_a_serve, p_b_serve)
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


def _simulate_tennis_set(p_a_serve: float, p_b_serve: float) -> tuple:
    """Simulate a single tennis set. Returns (a_games, b_games)."""
    a_games, b_games = 0, 0
    # Alternate serve: A serves first
    server_is_a = True

    while True:
        # Check for set win (6-x with 2+ lead, or tiebreak at 6-6)
        if a_games >= 6 and a_games - b_games >= 2:
            return a_games, b_games
        if b_games >= 6 and b_games - a_games >= 2:
            return a_games, b_games
        if a_games == 6 and b_games == 6:
            # Tiebreak
            if random.random() < (p_a_serve + (1 - p_b_serve)) / 2.0:
                return 7, 6
            else:
                return 6, 7

        # Simulate a game: server wins with p_serve probability
        # A game is ~4 points; serve win % maps to game win % via deuce model
        p_serve = p_a_serve if server_is_a else p_b_serve
        game_win_prob = _tennis_game_win_prob(p_serve)

        if random.random() < game_win_prob:
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
    p_reach_deuce = 20 * (p ** 3) * (q ** 3)

    # Prob server wins before deuce (4-0, 4-1, 4-2)
    p_win_0 = p ** 4
    p_win_1 = 4 * (p ** 4) * q
    p_win_2 = 10 * (p ** 4) * (q ** 2)

    return p_win_0 + p_win_1 + p_win_2 + p_reach_deuce * p_deuce_win


def _sim_golf(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
) -> tuple:
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
    for _ in range(n_iter):
        a_score = sum(_normal_sample(a_per_round, round_std, 1)[0] for _ in range(n_rounds))
        b_score = sum(_normal_sample(b_per_round, round_std, 1)[0] for _ in range(n_rounds))
        a_totals.append(a_score)
        b_totals.append(b_score)

    # In golf, lower is better. Invert for the standard result builder:
    # "home_win" = golfer A wins = golfer A has lower total
    # We store actual scores so the result builder counts correctly
    # But we need to flip the comparison: A wins when A < B
    # So we negate scores for the result builder
    return ([-s for s in a_totals], [-s for s in b_totals])


def _sim_fighting(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
) -> tuple:
    """Fighting: Win probability with method-of-victory modeling.

    off_rating = win percentage (0-1), finish_rate = rate of finishes.
    Returns scores as 1 (win) or 0 (loss) for each iteration, plus
    method-of-victory data in a side channel (stored in the context dicts).

    home = Fighter A, away = Fighter B.
    """
    a_win_pct = home_ctx.get("win_pct", 0.5)
    b_win_pct = away_ctx.get("win_pct", 0.5)
    a_finish = home_ctx.get("finish_rate", 0.5)
    b_finish = away_ctx.get("finish_rate", 0.5)
    a_ko = home_ctx.get("ko_tko_rate", 0.3)
    b_ko = away_ctx.get("ko_tko_rate", 0.3)
    a_sub = home_ctx.get("submission_rate", 0.15)
    b_sub = away_ctx.get("submission_rate", 0.15)

    rounds_scheduled = home_ctx.get("rounds_scheduled",
                                    config.get("default_rounds", 3))

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

    for _ in range(n_iter):
        if random.random() < p_a:
            a_scores.append(1.0)
            b_scores.append(0.0)
            # Method of victory for fighter A
            r = random.random()
            if r < a_ko:
                method_counts["ko_tko"] += 1
            elif r < a_ko + a_sub:
                method_counts["submission"] += 1
            else:
                method_counts["decision"] += 1
        else:
            a_scores.append(0.0)
            b_scores.append(1.0)
            r = random.random()
            if r < b_ko:
                method_counts["ko_tko"] += 1
            elif r < b_ko + b_sub:
                method_counts["submission"] += 1
            else:
                method_counts["decision"] += 1

    # Draw probability in boxing is ~2-3%, negligible in MMA
    league_cfg = config
    is_boxing = (league.upper() == "BOXING")
    draw_rate = 0.025 if is_boxing else 0.005
    # Retroactively convert some decisions to draws
    for i in range(len(a_scores)):
        if random.random() < draw_rate:
            a_scores[i] = 0.5
            b_scores[i] = 0.5
            method_counts["draw"] += 1
            method_counts["decision"] -= 1

    return a_scores, b_scores


def _sim_esports(
    home_ctx: Dict, away_ctx: Dict, league: str, n_iter: int, config: Dict
) -> tuple:
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

    for _ in range(n_iter):
        a_maps, b_maps = 0, 0
        while a_maps < maps_to_win and b_maps < maps_to_win:
            if random.random() < p_a_map:
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

class OmegaSimulationEngine:
    """
    High-level simulation engine dispatching to sport-archetype models.
    Input-driven only — no network calls.
    """

    def __init__(self):
        pass

    def run_fast_game_simulation(
        self,
        home_team: str,
        away_team: str,
        league: str = "NBA",
        n_iterations: int = 100,
        home_context: Optional[Dict] = None,
        away_context: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> Dict:
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

        Returns:
            Dict with score distributions, win probabilities, and missing_requirements
        """
        if seed is not None:
            random.seed(seed)
            if np is not None:
                np.random.seed(seed)

        league = league.upper()
        archetype = get_archetype(league)
        archetype_name = get_archetype_name(league)
        config = get_league_config(league)

        # Unknown sport → skip with helpful message
        if archetype is None or archetype_name is None:
            return _skip_result(
                home_team, away_team, league,
                skip_reason=f"No simulation model for league '{league}'. Add it to LEAGUE_TO_ARCHETYPE in sport_archetypes.py.",
                missing_requirements=["league_model"],
            )

        # Validate required context
        required = archetype.required_team_keys
        home_missing = _validate_required_keys(home_context, "home", required, league)
        away_missing = _validate_required_keys(away_context, "away", required, league)
        all_missing = home_missing + away_missing

        if all_missing:
            return _skip_result(
                home_team, away_team, league,
                skip_reason=f"Missing required inputs for {archetype.display_name}: {', '.join(all_missing)}",
                missing_requirements=all_missing,
            )

        # Dispatch to archetype simulator
        simulator = _ARCHETYPE_SIMULATORS.get(archetype_name)
        if simulator is None:
            return _skip_result(
                home_team, away_team, league,
                skip_reason=f"Simulator not implemented for archetype '{archetype_name}'",
                missing_requirements=["archetype_simulator"],
            )

        home_scores, away_scores = simulator(
            home_context, away_context, league, n_iterations, config
        )

        return _build_team_score_result(
            home_team, away_team, league, n_iterations,
            home_scores, away_scores,
            home_context=home_context,
            away_context=away_context,
            archetype_name=archetype_name,
        )

    def run_game_simulation(
        self,
        home_team: str,
        away_team: str,
        league: str = "NBA",
        n_iterations: int = 1000,
        home_context: Optional[Dict] = None,
        away_context: Optional[Dict] = None,
        home_players: Optional[List[Dict]] = None,
        away_players: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run a full game simulation using Markov engine + player projections.

        Architecture fix: callers must supply all context. No network calls.
        """
        from omega.core.simulation.markov_engine import MarkovSimulator  # noqa: future module

        if home_context is None or away_context is None:
            return _skip_result(
                home_team, away_team, league,
                skip_reason="Missing home_context or away_context (caller must supply)",
                missing_requirements=["home_context", "away_context"],
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
        all_player_stats: Dict[str, Dict[str, List]] = {}

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

        home_mean = sum(home_scores) / len(home_scores)
        away_mean = sum(away_scores) / len(away_scores)

        home_wins = sum(1 for h, a in zip(home_scores, away_scores) if h > a)
        away_wins = sum(1 for h, a in zip(home_scores, away_scores) if a > h)
        draws = n_iterations - home_wins - away_wins

        player_projections = {}
        for player_name, stats in all_player_stats.items():
            player_projections[player_name] = {}
            for stat_key, values in stats.items():
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                player_projections[player_name][stat_key] = {
                    "mean": sum(values) / n if n > 0 else 0,
                    "std": (sum((v - sum(values) / n) ** 2 for v in values) / n) ** 0.5 if n > 1 else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "p10": sorted_vals[int(n * 0.1)] if n > 10 else min(values) if values else 0,
                    "p25": sorted_vals[int(n * 0.25)] if n > 4 else min(values) if values else 0,
                    "p50": sorted_vals[int(n * 0.5)] if n > 2 else sum(values) / n if values else 0,
                    "p75": sorted_vals[int(n * 0.75)] if n > 4 else max(values) if values else 0,
                    "p90": sorted_vals[int(n * 0.9)] if n > 10 else max(values) if values else 0,
                }

        return {
            "success": True,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "archetype": get_archetype_name(league),
            "iterations": n_iterations,
            "home_win_prob": round(home_wins / n_iterations * 100, 1),
            "away_win_prob": round(away_wins / n_iterations * 100, 1),
            "draw_prob": round(draws / n_iterations * 100, 1),
            "predicted_home_score": round(home_mean, 1),
            "predicted_away_score": round(away_mean, 1),
            "predicted_spread": round(home_mean - away_mean, 1),
            "predicted_total": round(home_mean + away_mean, 1),
            "missing_requirements": [],
            "home_context": home_context,
            "away_context": away_context,
            "player_projections": player_projections,
            "home_scores_sample": [round(s, 1) for s in home_scores[:20]],
            "away_scores_sample": [round(s, 1) for s in away_scores[:20]],
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
        game_context: Optional[Dict] = None,
        player_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Run player prop simulation focused on a specific player's stats.
        Caller must supply game_context and player_context; engine does not fetch.
        """
        from omega.core.simulation.markov_engine import MarkovSimulator  # noqa: future module

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
