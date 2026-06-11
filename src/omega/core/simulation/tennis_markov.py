"""Tennis IID-Markov game backend with pressure-state nodes (Phase 7 M3).

Closed-form/exact probability chains over serve points:

* game   — exact finite-state recursion over point scores with the deuce loop
           solved algebraically; equals the Newton (1962) hold polynomial when
           all pressure deltas are zero;
* tiebreak — exact recursion with the real serve rotation (A serves point 1,
           then two points each); the 6-6 deadlock reduces to the alternating
           two-point-block closed form;
* set    — exact recursion over game scores with serve alternation and a 6-6
           tiebreak;
* match  — exact recursion over set scores honoring best-of-3/best-of-5.

Pressure states (design decision 7) apply additive SPW% deltas at exactly the
named nodes, per player, from ``request.prior_payload["pressure_coefficients"]``
(``{"home": {state: delta}, "away": {...}}``; a flat ``{state: delta}`` dict
applies to both sides). Missing states default to 0.0 — flat IID is the
documented rollback. Within-game mapping:

* ``break_point_against``   — receiver game points (x-40 / Ad-out);
* ``set_point_serving``     — server game points in a serving-for-set game;
* ``match_point_serving``   — same, when the set win clinches the match;
* ``serving_for_set``       — every point of a serving-for-set game;
* ``serving_for_match``     — every point of such a game in a clinch set;
* ``tiebreak``              — every point the player serves in a tiebreak.

Documented simplifications: per-match point probabilities blend each side's
SPW% with the opponent's RPW% (``p = (spw + 1 - opp_rpw) / 2``, matching the
legacy fast-score blend); set-win probabilities average the two first-server
cases and sets are treated independently at match level apart from the clinch
deltas (first-server effects on a full set are O(1e-3) and vanish at match
level, far inside the 0.5% MC acceptance tolerance).

Market conventions: ``request.spread_home`` is the SET handicap (archetype
``set_spread``); ``request.over_under`` is the GAMES total (``total_games``,
tiebreak counted as one game). Headline win probabilities are closed-form
(``distribution_type="markov_closed_form"`` rows) so calibration consumes
exact values; score/total distribution rows come from a seeded game-level
Monte Carlo of the same chains. No serve/return defaults exist: missing
``serve_win_pct``/``return_win_pct`` fails closed (the legacy ``_sim_tennis``
0.64/0.62 defaults stay quarantined on the fast_score path).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from omega.core.simulation.backends import GameSimulationInput

try:  # pragma: no cover - exercised implicitly wherever numpy exists
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

_REQUIRED_KEYS = ("serve_win_pct", "return_win_pct")
_P_FLOOR, _P_CEIL = 0.01, 0.99

_STATE_BP = "break_point_against"
_STATE_SP = "set_point_serving"
_STATE_MP = "match_point_serving"
_STATE_TB = "tiebreak"
_STATE_SFS = "serving_for_set"
_STATE_SFM = "serving_for_match"


def _clamp(p: float) -> float:
    return max(_P_FLOOR, min(_P_CEIL, p))


# ---------------------------------------------------------------------------
# Game level
# ---------------------------------------------------------------------------


def p_hold_closed_form(p: float) -> float:
    """Newton (1962) closed-form service-hold probability (no pressure)."""
    q = 1.0 - p
    direct = p**4 * (1.0 + 4.0 * q + 10.0 * q * q)
    deuce_reach = 20.0 * p**3 * q**3
    deuce_win = p * p / (1.0 - 2.0 * p * q)
    return direct + deuce_reach * deuce_win


@lru_cache(maxsize=None)
def p_hold_chain(p: float, bp_delta: float = 0.0, gp_delta: float = 0.0) -> float:
    """Exact hold probability with per-state deltas at game/break points.

    Equals :func:`p_hold_closed_form` when both deltas are zero. The deuce
    cycle (deuce -> Ad -> deuce) is solved algebraically so the recursion is
    finite: with ``D = P(hold from deuce)``, Ad-in plays at the game-point
    probability and Ad-out at the break-point probability, giving
    ``D = p*p_gp / (1 - p*(1-p_gp) - (1-p)*p_bp)``.
    """
    p = _clamp(p)
    p_gp = _clamp(p + gp_delta)  # server game point (40-x, Ad-in)
    p_bp = _clamp(p + bp_delta)  # break point against (x-40, Ad-out)

    denom = 1.0 - p * (1.0 - p_gp) - (1.0 - p) * p_bp
    deuce = (p * p_gp / denom) if denom > 1e-12 else 1.0

    def f(s: int, r: int) -> float:
        if s == 3 and r == 3:
            return deuce
        if s == 3:  # game point for server (r < 3)
            return p_gp + (1.0 - p_gp) * f(3, r + 1)
        if r == 3:  # break point (s < 3); receiver point ends the game
            return p_bp * f(s + 1, 3)
        return p * f(s + 1, r) + (1.0 - p) * f(s, r + 1)

    return f(0, 0)


# ---------------------------------------------------------------------------
# Tiebreak level
# ---------------------------------------------------------------------------


def p_tiebreak(pa: float, pb: float, tb_delta_a: float = 0.0, tb_delta_b: float = 0.0) -> float:
    """P(player A wins a tiebreak); exact with the real serve rotation."""
    pa_eff = _clamp(pa + tb_delta_a)  # A wins a point on A's serve
    pb_eff = _clamp(pb + tb_delta_b)  # B wins a point on B's serve

    block_a = pa_eff * (1.0 - pb_eff)
    block_b = (1.0 - pa_eff) * pb_eff
    deadlock = block_a / (block_a + block_b) if (block_a + block_b) > 1e-12 else 0.5

    cache: dict[tuple[int, int], float] = {}

    def f(a: int, b: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        if a == b and a >= 6:
            return deadlock
        key = (a, b)
        if key not in cache:
            n = a + b + 1  # 1-indexed point number; A serves when n % 4 in {0, 1}
            a_serving = n % 4 in (0, 1)
            p_point = pa_eff if a_serving else 1.0 - pb_eff
            cache[key] = p_point * f(a + 1, b) + (1.0 - p_point) * f(a, b + 1)
        return cache[key]

    return f(0, 0)


# ---------------------------------------------------------------------------
# Set / match level
# ---------------------------------------------------------------------------


def _coeff(coeffs: dict[str, float] | None, state: str) -> float:
    if not coeffs:
        return 0.0
    return float(coeffs.get(state, 0.0))


def _hold_for(
    p_base: float,
    coeffs: dict[str, float] | None,
    *,
    serving_for_set: bool,
    clinch: bool,
) -> float:
    """Hold probability for one service game in its set/match context."""
    if serving_for_set:
        flat = _coeff(coeffs, _STATE_SFM if clinch else _STATE_SFS)
        gp = _coeff(coeffs, _STATE_MP if clinch else _STATE_SP)
    else:
        flat, gp = 0.0, 0.0
    return p_hold_chain(
        round(_clamp(p_base + flat), 12),
        round(_coeff(coeffs, _STATE_BP), 12),
        round(gp, 12),
    )


def p_set(
    pa: float,
    pb: float,
    coeffs_a: dict[str, float] | None,
    coeffs_b: dict[str, float] | None,
    *,
    a_clinch: bool = False,
    b_clinch: bool = False,
    first_server_a: bool = True,
) -> float:
    """P(player A wins the set); exact recursion over game scores."""
    tb = p_tiebreak(pa, pb, _coeff(coeffs_a, _STATE_TB), _coeff(coeffs_b, _STATE_TB))
    cache: dict[tuple[int, int, bool], float] = {}

    def f(ga: int, gb: int, a_serving: bool) -> float:
        if ga >= 6 and ga - gb >= 2:
            return 1.0
        if gb >= 6 and gb - ga >= 2:
            return 0.0
        if ga == 7:
            return 1.0
        if gb == 7:
            return 0.0
        if ga == 6 and gb == 6:
            return tb
        key = (ga, gb, a_serving)
        if key not in cache:
            if a_serving:
                sfs = (ga == 5 and gb <= 4) or (ga == 6 and gb == 5)
                hold = _hold_for(pa, coeffs_a, serving_for_set=sfs, clinch=a_clinch)
                cache[key] = hold * f(ga + 1, gb, False) + (1.0 - hold) * f(ga, gb + 1, False)
            else:
                sfs = (gb == 5 and ga <= 4) or (gb == 6 and ga == 5)
                hold = _hold_for(pb, coeffs_b, serving_for_set=sfs, clinch=b_clinch)
                cache[key] = hold * f(ga, gb + 1, True) + (1.0 - hold) * f(ga + 1, gb, True)
        return cache[key]

    return f(0, 0, first_server_a)


def _p_set_avg(pa, pb, coeffs_a, coeffs_b, *, a_clinch=False, b_clinch=False) -> float:
    """Set-win probability averaged over the two first-server cases."""
    return 0.5 * (
        p_set(pa, pb, coeffs_a, coeffs_b, a_clinch=a_clinch, b_clinch=b_clinch,
              first_server_a=True)
        + p_set(pa, pb, coeffs_a, coeffs_b, a_clinch=a_clinch, b_clinch=b_clinch,
                first_server_a=False)
    )


def match_set_score_distribution(
    pa: float,
    pb: float,
    coeffs_a: dict[str, float] | None,
    coeffs_b: dict[str, float] | None,
    best_of: int,
) -> dict[tuple[int, int], float]:
    """Probability of each terminal set score, e.g. {(2,0): .., (1,2): ..}."""
    sets_to_win = best_of // 2 + 1
    terminals: dict[tuple[int, int], float] = {}

    set_cache: dict[tuple[bool, bool], float] = {}

    def set_win(a_clinch: bool, b_clinch: bool) -> float:
        key = (a_clinch, b_clinch)
        if key not in set_cache:
            set_cache[key] = _p_set_avg(
                pa, pb, coeffs_a, coeffs_b, a_clinch=a_clinch, b_clinch=b_clinch
            )
        return set_cache[key]

    def walk(a: int, b: int, prob: float) -> None:
        if prob <= 0.0:
            return
        if a == sets_to_win or b == sets_to_win:
            terminals[(a, b)] = terminals.get((a, b), 0.0) + prob
            return
        pw = set_win(a == sets_to_win - 1, b == sets_to_win - 1)
        walk(a + 1, b, prob * pw)
        walk(a, b + 1, prob * (1.0 - pw))

    walk(0, 0, 1.0)
    return terminals


# ---------------------------------------------------------------------------
# Seeded Monte Carlo (game level, same chains) for distribution rows
# ---------------------------------------------------------------------------


def _simulate_match(rng, pa, pb, coeffs_a, coeffs_b, sets_to_win):
    """One match at game level. Returns (sets_a, sets_b, games, set1_games, set1_a_won)."""
    tb_a = p_tiebreak(pa, pb, _coeff(coeffs_a, _STATE_TB), _coeff(coeffs_b, _STATE_TB))
    sets_a = sets_b = total_games = 0
    set1_games = 0
    set1_a_won = False
    a_serves = True  # rotation carries across sets

    set_index = 0
    while sets_a < sets_to_win and sets_b < sets_to_win:
        a_clinch = sets_a == sets_to_win - 1
        b_clinch = sets_b == sets_to_win - 1
        ga = gb = 0
        while True:
            if ga == 6 and gb == 6:
                a_won_game = rng.random() < tb_a
                total_games += 1
                ga, gb = (7, 6) if a_won_game else (6, 7)
                a_serves = not a_serves
                break
            if a_serves:
                sfs = (ga == 5 and gb <= 4) or (ga == 6 and gb == 5)
                hold = _hold_for(pa, coeffs_a, serving_for_set=sfs, clinch=a_clinch)
                if rng.random() < hold:
                    ga += 1
                else:
                    gb += 1
            else:
                sfs = (gb == 5 and ga <= 4) or (gb == 6 and ga == 5)
                hold = _hold_for(pb, coeffs_b, serving_for_set=sfs, clinch=b_clinch)
                if rng.random() < hold:
                    gb += 1
                else:
                    ga += 1
            total_games += 1
            a_serves = not a_serves
            if (ga >= 6 and ga - gb >= 2) or (gb >= 6 and gb - ga >= 2):
                break
        if set_index == 0:
            set1_games = ga + gb
            set1_a_won = ga > gb
        if ga > gb:
            sets_a += 1
        else:
            sets_b += 1
        set_index += 1
    return sets_a, sets_b, total_games, set1_games, set1_a_won


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def _resolve_best_of(league_config: dict, prior: dict[str, Any]) -> int:
    fmt = prior.get("match_format")
    if isinstance(fmt, str) and fmt.startswith("best_of_"):
        fmt = fmt.removeprefix("best_of_")
    if fmt is not None:
        best_of = int(fmt)
    else:
        best_of = int(league_config.get("best_of", 3))
    if best_of not in (3, 5):
        raise ValueError(f"match_format must resolve to best-of 3 or 5, got {best_of}")
    return best_of


def _side_coeffs(prior: dict[str, Any], side: str) -> dict[str, float] | None:
    pc = prior.get("pressure_coefficients")
    if not isinstance(pc, dict):
        return None
    if "home" in pc or "away" in pc:
        side_pc = pc.get(side)
        return dict(side_pc) if isinstance(side_pc, dict) else None
    return dict(pc)  # flat dict applies to both players


class TennisMarkovBackend:
    """Closed-form serve-point Markov backend with pressure-state nodes."""

    backend_name = "tennis_markov_iid"
    component_version = "tennis_markov_iid_v1"
    evidence_mode = "plane_adjustment"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        from omega.core.config.leagues import get_league_config
        from omega.core.simulation.engine import (
            _build_team_score_result,
            _context_hash,
            _distribution_row,
            _skip_result,
            _validate_required_keys,
        )

        league = request.league.upper()
        if np is None:
            return _skip_result(
                request.home_team, request.away_team, league,
                skip_reason="numpy is required for the tennis Markov backend",
                missing_requirements=["numpy"],
            )

        missing = _validate_required_keys(request.home_context, "home", _REQUIRED_KEYS, league)
        missing += _validate_required_keys(request.away_context, "away", _REQUIRED_KEYS, league)
        if missing:
            return _skip_result(
                request.home_team, request.away_team, league,
                skip_reason=(
                    "Missing tennis serve/return inputs (no defaults; supply "
                    "priors_tennis rates): " + ", ".join(missing)
                ),
                missing_requirements=missing,
            )

        prior = request.prior_payload or {}
        try:
            best_of = _resolve_best_of(get_league_config(league), prior)
        except (TypeError, ValueError) as exc:
            return _skip_result(
                request.home_team, request.away_team, league,
                skip_reason=f"Invalid match_format: {exc}",
                missing_requirements=["match_format"],
            )
        sets_to_win = best_of // 2 + 1

        home_ctx = request.home_context or {}
        away_ctx = request.away_context or {}
        # Bradley-Terry-style blend of own SPW% vs opponent RPW% (legacy blend).
        pa = _clamp((float(home_ctx["serve_win_pct"]) + (1.0 - float(away_ctx["return_win_pct"]))) / 2.0)
        pb = _clamp((float(away_ctx["serve_win_pct"]) + (1.0 - float(home_ctx["return_win_pct"]))) / 2.0)
        coeffs_a = _side_coeffs(prior, "home")
        coeffs_b = _side_coeffs(prior, "away")

        # Closed-form probabilities.
        terminals = match_set_score_distribution(pa, pb, coeffs_a, coeffs_b, best_of)
        p_match_a = sum(p for (a, b), p in terminals.items() if a > b)
        p_set1_a = _p_set_avg(pa, pb, coeffs_a, coeffs_b)  # set 1 is never a clinch set
        p_a_wins_a_set = 1.0 - terminals.get((0, sets_to_win), 0.0)

        # Seeded MC of the same chains for score/total distribution rows.
        rng = np.random.default_rng(request.seed)
        n = request.n_iterations
        sets_a_samples: list[float] = []
        sets_b_samples: list[float] = []
        games_samples: list[float] = []
        set1_games_samples: list[float] = []
        set1_wins = 0
        for _ in range(n):
            sa, sb, games, s1g, s1w = _simulate_match(
                rng, pa, pb, coeffs_a, coeffs_b, sets_to_win
            )
            sets_a_samples.append(float(sa))
            sets_b_samples.append(float(sb))
            games_samples.append(float(games))
            set1_games_samples.append(float(s1g))
            set1_wins += int(s1w)

        result = _build_team_score_result(
            request.home_team,
            request.away_team,
            league,
            n,
            sets_a_samples,
            sets_b_samples,
            home_context=request.home_context,
            away_context=request.away_context,
            archetype_name="tennis",
            spread_home=request.spread_home,  # SET handicap by convention
            over_under=None,  # games total handled below from games samples
            context_source="provided",
            baseline_used=False,
            seed=request.seed,
            backend_name=self.backend_name,
            component_version=self.component_version,
        )

        # Headline probabilities are closed-form (calibration reads exact values).
        result["home_win_prob"] = round(p_match_a * 100.0, 1)
        result["away_win_prob"] = round((1.0 - p_match_a) * 100.0, 1)
        result["draw_prob"] = 0.0
        games_mean = sum(games_samples) / n
        result["predicted_total"] = round(games_mean, 1)
        result["match_format"] = f"best_of_{best_of}"
        result["p_serve_point_home"] = round(pa, 4)
        result["p_serve_point_away"] = round(pb, 4)
        if prior.get("pressure_coefficient_source") is not None:
            result["pressure_coefficient_source"] = prior["pressure_coefficient_source"]

        if request.over_under is not None:
            over = sum(1 for g in games_samples if g > request.over_under)
            under = sum(1 for g in games_samples if g < request.over_under)
            result["over_prob"] = round(over / n * 100.0, 1)
            result["under_prob"] = round(under / n * 100.0, 1)

        ctx_hash = _context_hash(request.home_context, request.away_context)

        def _closed_form_row(target: str, prob: float) -> dict[str, Any]:
            return _distribution_row(
                target=target,
                market=target,
                stat_key=None,
                distribution_type="markov_closed_form",
                distribution_params={
                    "closed_form": True,
                    "best_of": best_of,
                    "p_serve_point_home": round(pa, 6),
                    "p_serve_point_away": round(pb, 6),
                },
                samples=[prob],
                n_iterations=n,
                seed=request.seed,
                context_hash=ctx_hash,
                component_version=self.component_version,
            )

        def _empirical_row(target: str, samples: list[float]) -> dict[str, Any]:
            return _distribution_row(
                target=target,
                market=target,
                stat_key=None,
                distribution_type="empirical",
                distribution_params={"source": "tennis_markov_mc"},
                samples=samples,
                n_iterations=n,
                seed=request.seed,
                context_hash=ctx_hash,
                component_version=self.component_version,
            )

        result["simulation_distributions"].extend(
            [
                _closed_form_row("match_winner", p_match_a),
                _closed_form_row("set_winner_set_1", p_set1_a),
                _closed_form_row("player_a_wins_a_set", p_a_wins_a_set),
                _empirical_row("total_games_match", games_samples),
                _empirical_row("total_games_set_1", set1_games_samples),
            ]
        )
        return result
