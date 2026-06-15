"""Soccer bivariate-Poisson game backend with Dixon-Coles low-score correction.

Phase 7 Milestone 2 backend (design: docs/phase7/MULTI_SPORT_EXPANSION.md Part 4).
Goal lambdas derive from team xG (preferred) or off/def goal rates, mirroring the
legacy ``_sim_soccer`` derivation so club-league traces on the fast_score path
stay comparable. Scorelines are sampled from the joint Dixon-Coles-adjusted pmf
via the engine's vectorized ``_dixon_coles_scores`` helper.

The Dixon-Coles ``rho`` is a **dynamic prior**: it must arrive in
``request.prior_payload["rho"]``, fit per competition profile (``fifa_intl_v1``,
``epl_v1``, ...) by ``omega-fit-dixon-coles`` and injected by the gatherer from
the ``priors_dixon_coles`` table. The static ``rho`` keys in ``leagues.py`` are
legacy fast_score-path inputs and are never read here. When the prior is absent
the backend fails closed with a skip result (``missing_requirements=
["rho_prior"]``) — no default rho, per locked design decision 5.
"""

from __future__ import annotations

import math
from typing import Any

from omega.core.simulation.backends import GameSimulationInput

try:  # pragma: no cover - exercised implicitly everywhere numpy exists
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

# Defensive clamp mirroring _sim_soccer: a league falling through to the generic
# _DEFAULT_CONFIG (avg_total=100) would shrink lambdas to near zero.
_SOCCER_AVG_TOTAL_CEILING = 10.0
_SOCCER_ARCHETYPE_AVG_TOTAL = 2.5

# First-half goal share for the independent-thinning approximation: each match
# goal lands in the first half with this probability (empirically ~45% of
# goals are scored before the break across top competitions; second halves run
# longer and tire defenses). 1H totals are Binomial(total_goals, share) — an
# approximation that ignores in-match state, documented per design Part 4.
_FIRST_HALF_GOAL_SHARE = 0.45


# Fail-closed contract: when no production Dixon-Coles profile exists the
# gatherer leaves ``rho`` out of prior_payload and this backend returns a
# skip-result (status="skipped", missing_requirements=["rho_prior"]). There is
# deliberately no exception type — skipping a single competition must never
# abort a batch, and the skip-result already carries the fail-closed signal.


def _valid_dixon_coles_rho(home_lambda: float, away_lambda: float, rho: float) -> bool:
    if not math.isfinite(rho):
        return False
    lower = max(-1.0 / home_lambda, -1.0 / away_lambda)
    upper = min(1.0 / (home_lambda * away_lambda), 1.0)
    return lower <= rho <= upper


def _missing_soccer_inputs(context: dict[str, Any] | None, side: str) -> list[str]:
    """Missing-requirement strings for one side's attack/defense inputs.

    Either xG keys (``xg_for``/``xg_against``) or goal-rate keys
    (``off_rating``/``def_rating``) satisfy the requirement.
    """
    if context is None:
        return [
            f"{side}_context.off_rating|xg_for",
            f"{side}_context.def_rating|xg_against",
        ]
    missing = []
    if context.get("xg_for") is None and context.get("off_rating") is None:
        missing.append(f"{side}_context.off_rating|xg_for")
    if context.get("xg_against") is None and context.get("def_rating") is None:
        missing.append(f"{side}_context.def_rating|xg_against")
    return missing


class SoccerPoissonBackend:
    """Bivariate Poisson + Dixon-Coles backend; fail-closed on the rho prior."""

    backend_name = "soccer_bivariate_poisson_dc"
    component_version = "soccer_bvp_dc_v1"
    evidence_mode = "plane_adjustment"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        # Lazy imports avoid an import cycle: engine.py imports this module at
        # the bottom to register the backend (markov_wnba reference pattern).
        from omega.core.config.leagues import get_league_config
        from omega.core.simulation.engine import (
            _build_team_score_result,
            _context_hash,
            _distribution_row,
            _dixon_coles_scores,
            _expected_against_allowed_rate,
            _skip_result,
        )

        league = request.league.upper()

        if np is None:
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason="numpy is required for the bivariate-Poisson backend",
                missing_requirements=["numpy"],
            )

        missing = _missing_soccer_inputs(request.home_context, "home")
        missing += _missing_soccer_inputs(request.away_context, "away")
        if missing:
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason=(
                    "Missing soccer attack/defense inputs: " + ", ".join(missing)
                ),
                missing_requirements=missing,
            )

        prior = request.prior_payload or {}
        try:
            rho = float(prior["rho"])
        except (KeyError, TypeError, ValueError):
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason=(
                    "Dixon-Coles rho prior missing from prior_payload; fit and "
                    "promote a competition profile via omega-fit-dixon-coles"
                ),
                missing_requirements=["rho_prior"],
            )

        config = get_league_config(league)
        avg_total = config.get("avg_total", _SOCCER_ARCHETYPE_AVG_TOTAL)
        if avg_total > _SOCCER_AVG_TOTAL_CEILING:
            avg_total = _SOCCER_ARCHETYPE_AVG_TOTAL
        league_avg_gpg = avg_total / 2.0

        home_ctx = request.home_context or {}
        away_ctx = request.away_context or {}
        home_xg = home_ctx.get("xg_for", home_ctx.get("off_rating", league_avg_gpg))
        away_xg = away_ctx.get("xg_for", away_ctx.get("off_rating", league_avg_gpg))
        home_xga = home_ctx.get("xg_against", home_ctx.get("def_rating", league_avg_gpg))
        away_xga = away_ctx.get("xg_against", away_ctx.get("def_rating", league_avg_gpg))

        home_lambda = _expected_against_allowed_rate(home_xg, away_xga, league_avg_gpg)
        away_lambda = _expected_against_allowed_rate(away_xg, home_xga, league_avg_gpg)

        hca = config.get("home_advantage", 0.0)
        home_lambda += hca / 2.0
        away_lambda -= hca / 2.0

        # Poisson(0.5) floor: lower floors hallucinate structurally unrealistic
        # 0-0 rates (see _sim_soccer).
        home_lambda = max(0.5, home_lambda)
        away_lambda = max(0.5, away_lambda)
        
        if request.dispersion is not None and request.dispersion.variance_multiplier != 1.0:
            mult = request.dispersion.variance_multiplier
            home_lambda *= mult
            away_lambda *= mult
            request.dispersion.applied_to.extend(["home_lambda", "away_lambda"])

        if not _valid_dixon_coles_rho(home_lambda, away_lambda, rho):
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason=(
                    "Dixon-Coles rho prior is outside admissible bounds for "
                    f"home_lambda={home_lambda:.4f}, away_lambda={away_lambda:.4f}"
                ),
                missing_requirements=["valid_rho_prior"],
            )

        if request.exact:
            return self._run_exact(
                request,
                league,
                home_lambda,
                away_lambda,
                rho,
                prior,
            )

        rng = np.random.default_rng(request.seed)
        home_scores, away_scores = _dixon_coles_scores(
            home_lambda, away_lambda, rho, request.n_iterations, rng=rng
        )

        result = _build_team_score_result(
            request.home_team,
            request.away_team,
            league,
            request.n_iterations,
            home_scores,
            away_scores,
            home_context=request.home_context,
            away_context=request.away_context,
            archetype_name="soccer",
            spread_home=request.spread_home,
            over_under=request.over_under,
            context_source="provided",
            baseline_used=False,
            seed=request.seed,
            backend_name=self.backend_name,
            component_version=self.component_version,
        )

        # Provenance for trace/sidecar audit: which fitted profile priced this.
        result["dc_rho"] = rho
        if prior.get("rho_profile_id") is not None:
            result["rho_profile_id"] = prior["rho_profile_id"]
        if prior.get("rho_as_of_date") is not None:
            result["rho_as_of_date"] = prior["rho_as_of_date"]

        # Soccer-derivative distribution rows beyond the standard score/total/
        # spread set, consumed downstream by the soccer edge logic.
        ctx_hash = _context_hash(request.home_context, request.away_context)
        dc_params = {
            "source": "dixon_coles_joint",
            "home_lambda": round(home_lambda, 4),
            "away_lambda": round(away_lambda, 4),
            "rho": rho,
        }

        def _extra_row(target: str, market: str, samples: list[float]) -> dict[str, Any]:
            return _distribution_row(
                target=target,
                market=market,
                stat_key=None,
                distribution_type="empirical",
                distribution_params=dc_params,
                samples=samples,
                n_iterations=request.n_iterations,
                seed=request.seed,
                context_hash=ctx_hash,
                component_version=self.component_version,
            )

        totals = [float(h + a) for h, a in zip(home_scores, away_scores)]

        # First-half totals via independent thinning (drawn after the score
        # sample on the same seeded rng, so replays stay bit-identical).
        fh_totals = rng.binomial(
            [int(t) for t in totals], _FIRST_HALF_GOAL_SHARE
        ).tolist()

        result["simulation_distributions"].extend(
            [
                _extra_row("total_goals", "game_total", totals),
                _extra_row(
                    "home_clean_sheet",
                    "clean_sheet",
                    [1.0 if a == 0 else 0.0 for a in away_scores],
                ),
                _extra_row(
                    "away_clean_sheet",
                    "clean_sheet",
                    [1.0 if h == 0 else 0.0 for h in home_scores],
                ),
                _extra_row(
                    "both_teams_to_score",
                    "btts",
                    [1.0 if h > 0 and a > 0 else 0.0 for h, a in zip(home_scores, away_scores)],
                ),
                _extra_row(
                    "first_half_total", "first_half_total", [float(t) for t in fh_totals]
                ),
            ]
        )

        # Empirical pmfs for downstream derivative-market evaluation (Asian
        # handicap quarter-lines, 1H totals) by omega/core/edge/
        # soccer_derivatives.py. String keys keep the trace JSON-safe.
        def _counts(values: list) -> dict[str, int]:
            out: dict[str, int] = {}
            for v in values:
                key = str(int(v))
                out[key] = out.get(key, 0) + 1
            return out

        result["margin_counts"] = _counts(
            [h - a for h, a in zip(home_scores, away_scores)]
        )
        result["total_counts"] = _counts(totals)
        result["fh_total_counts"] = _counts(fh_totals)
        return result

    def _run_exact(
        self,
        request: GameSimulationInput,
        league: str,
        home_lambda: float,
        away_lambda: float,
        rho: float,
        prior: dict[str, Any],
    ) -> dict[str, Any]:
        """Exact (non-MC) evaluation of the Dixon-Coles joint.

        Sums every market over the same Dixon-Coles grid the MC path samples, and
        emits exact pmfs for the soccer-derivative rows/counts the edge layer
        consumes (margin/total/first-half) — zero sampling noise. Markets and rows
        mirror the MC ``run`` path; the equivalence is pinned by the exact-eval
        parity tests.
        """
        from omega.core.simulation import exact_eval
        from omega.core.simulation.engine import (
            _SOCCER_DC_MAX_GOALS,
            _analytic_distribution_row,
            _build_team_score_result_exact_grid,
            _context_hash,
            _normalized_score_grid,
        )

        max_goals = _SOCCER_DC_MAX_GOALS
        grid = _normalized_score_grid(home_lambda, away_lambda, rho, max_goals)
        exact_version = self.component_version.removesuffix("_v1") + "_exact_v1"

        result = _build_team_score_result_exact_grid(
            request.home_team,
            request.away_team,
            league,
            request.n_iterations,
            grid,
            home_lambda,
            away_lambda,
            rho,
            max_goals,
            home_context=request.home_context,
            away_context=request.away_context,
            archetype_name="soccer",
            spread_home=request.spread_home,
            over_under=request.over_under,
            context_source="provided",
            baseline_used=False,
            seed=request.seed,
            backend_name=self.backend_name + "_exact",
            component_version=exact_version,
        )

        # Provenance (mirrors the MC path).
        result["dc_rho"] = rho
        if prior.get("rho_profile_id") is not None:
            result["rho_profile_id"] = prior["rho_profile_id"]
        if prior.get("rho_as_of_date") is not None:
            result["rho_as_of_date"] = prior["rho_as_of_date"]

        # Exact soccer-derivative rows + pmfs (consumed by soccer_derivatives.py).
        ctx_hash = _context_hash(request.home_context, request.away_context)
        dc_params = {
            "source": "dixon_coles_joint_exact",
            "home_lambda": round(home_lambda, 4),
            "away_lambda": round(away_lambda, 4),
            "rho": rho,
        }
        g = grid.shape[0]
        home_marg = grid.sum(axis=1)
        away_marg = grid.sum(axis=0)
        hh, aa = np.indices((g, g))
        flat = grid.ravel()
        total_vals = np.arange(2 * (g - 1) + 1)
        total_probs = np.zeros(2 * (g - 1) + 1)
        np.add.at(total_probs, (hh + aa).ravel(), flat)

        p_home_cs = float(away_marg[0])  # home clean sheet ⇔ away scores 0
        p_away_cs = float(home_marg[0])
        p_btts = float(1.0 - home_marg[0] - away_marg[0] + grid[0, 0])

        fh_pmf = exact_eval.thinned_total_pmf(grid, _FIRST_HALF_GOAL_SHARE)
        fh_vals = np.array(sorted(int(k) for k in fh_pmf), dtype=float)
        fh_probs = np.array([fh_pmf[str(int(v))] for v in fh_vals], dtype=float)

        def _exrow(target: str, market: str, values, probs) -> dict[str, Any]:
            return _analytic_distribution_row(
                target=target,
                market=market,
                values=values,
                probs=probs,
                distribution_params=dc_params,
                n_iterations=request.n_iterations,
                seed=request.seed,
                context_hash=ctx_hash,
                component_version=exact_version,
            )

        result["simulation_distributions"].extend(
            [
                _exrow("total_goals", "game_total", total_vals, total_probs),
                _exrow("home_clean_sheet", "clean_sheet", np.array([0.0, 1.0]), np.array([1.0 - p_home_cs, p_home_cs])),
                _exrow("away_clean_sheet", "clean_sheet", np.array([0.0, 1.0]), np.array([1.0 - p_away_cs, p_away_cs])),
                _exrow("both_teams_to_score", "btts", np.array([0.0, 1.0]), np.array([1.0 - p_btts, p_btts])),
                _exrow("first_half_total", "first_half_total", fh_vals, fh_probs),
            ]
        )

        margin_counts, total_counts = exact_eval.margin_total_pmfs(grid)
        result["margin_counts"] = margin_counts
        result["total_counts"] = total_counts
        result["fh_total_counts"] = dict(fh_pmf)
        return result
