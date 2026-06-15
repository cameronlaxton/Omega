"""NFL Gamma-Poisson (Negative Binomial) team-score backend.

Phase 7 Milestone 4 (design: docs/phase7/MULTI_SPORT_EXPANSION.md Part 4). Team
scores are sampled from a Negative Binomial (Gamma-Poisson mixture) rather than
the fast_score Normal, capturing the heavy upper tail (40+ point outputs) that a
Normal under-prices. Score means derive from the same off/def-rating context the
fast_score american-football path uses, so traces stay comparable.

The discrete margin pmf (``margin_counts``) and total pmf (``total_counts``) are
emitted alongside the standard score/total/spread distribution rows so the NFL
edge consumer (``omega/core/edge/nfl_teasers.py`` via ``nfl_consumer.py``) can
evaluate Wong-teaser legs at the 3- and 7-point crossings. Teaser EV math is done
in the edge layer; the backend stays line-unaware.

Exact (non-MC) evaluation is not implemented for this backend yet — the ``exact``
flag falls back to Monte-Carlo, the documented behavior for backends without a
closed form. Independent-NB convolution is a candidate follow-up.
"""

from __future__ import annotations

from typing import Any

from omega.core.simulation.backends import GameSimulationInput

try:  # pragma: no cover - exercised implicitly everywhere numpy exists
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

# Game-level team-score over-dispersion. NB variance = mu + mu**2 / k. At an NFL
# team mean of ~22 this default reproduces ~Normal(sd=10) variance (k ≈ 22**2 /
# (100 - 22) ≈ 6.2) while adding the right-skew the Normal lacks. Override per
# request via prior_payload["team_score_nb_k"] or per league via config.
_DEFAULT_TEAM_SCORE_NB_K = 6.0

# NB requires a strictly positive mean; floor mirrors the Poisson floors used by
# the soccer/baseball backends so a degenerate context cannot produce mu <= 0.
_TEAM_SCORE_FLOOR = 1.0


def _missing_nfl_inputs(context: dict[str, Any] | None, side: str) -> list[str]:
    """Missing-requirement strings for one side's critical scoring inputs.

    The american_football archetype's critical team keys are off_rating and
    def_rating (points-for / points-against per game); both are required, matching
    the archetype contract rather than silently defaulting like the legacy path.
    """
    if context is None:
        return [f"{side}_context.off_rating", f"{side}_context.def_rating"]
    missing = []
    if context.get("off_rating") is None:
        missing.append(f"{side}_context.off_rating")
    if context.get("def_rating") is None:
        missing.append(f"{side}_context.def_rating")
    return missing


def _resolve_team_score_nb_k(prior: dict[str, Any], config: dict[str, Any]) -> float:
    raw = prior.get("team_score_nb_k", config.get("team_score_nb_k"))
    if raw is None:
        return _DEFAULT_TEAM_SCORE_NB_K
    k = float(raw)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("team_score_nb_k must be a finite positive number")
    return k


def _counts(values: Any) -> dict[str, int]:
    """Empirical pmf as {str(int(value)): count}, JSON-safe for the trace."""
    out: dict[str, int] = {}
    for v in values:
        key = str(int(v))
        out[key] = out.get(key, 0) + 1
    return out


class NflSimulationBackend:
    """Gamma-Poisson NFL team-score backend exposing the discrete margin pmf."""

    backend_name = "nfl_neg_binom"
    component_version = "nfl_nb_v1"
    evidence_mode = "plane_adjustment"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        # Lazy imports avoid an import cycle: engine.py imports this module at the
        # bottom to register the backend (markov_wnba reference pattern).
        from omega.core.config.leagues import get_league_config
        from omega.core.simulation.engine import (
            _american_football_score_params,
            _build_team_score_result,
            _context_hash,
            _distribution_row,
            _skip_result,
        )

        league = request.league.upper()

        if np is None:
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason="numpy is required for the negative-binomial backend",
                missing_requirements=["numpy"],
            )

        missing = _missing_nfl_inputs(request.home_context, "home")
        missing += _missing_nfl_inputs(request.away_context, "away")
        if missing:
            return _skip_result(
                request.home_team,
                request.away_team,
                league,
                skip_reason="Missing NFL scoring inputs: " + ", ".join(missing),
                missing_requirements=missing,
            )

        config = get_league_config(league)
        prior = request.prior_payload or {}
        k = _resolve_team_score_nb_k(prior, config)

        home_mu, _std, away_mu, _ = _american_football_score_params(
            request.home_context or {}, request.away_context or {}, league, config
        )
        home_mu = max(_TEAM_SCORE_FLOOR, home_mu)
        away_mu = max(_TEAM_SCORE_FLOOR, away_mu)

        rng = np.random.default_rng(request.seed)
        home_scores = rng.negative_binomial(k, k / (k + home_mu), size=request.n_iterations)
        away_scores = rng.negative_binomial(k, k / (k + away_mu), size=request.n_iterations)

        result = _build_team_score_result(
            request.home_team,
            request.away_team,
            league,
            request.n_iterations,
            home_scores,
            away_scores,
            home_context=request.home_context,
            away_context=request.away_context,
            archetype_name="american_football",
            spread_home=request.spread_home,
            over_under=request.over_under,
            context_source="provided",
            baseline_used=False,
            seed=request.seed,
            backend_name=self.backend_name,
            component_version=self.component_version,
        )

        # Provenance: the dispersion that priced this game.
        result["team_score_nb_k"] = k

        # Discrete margin + total distribution row (the heavy-tail payload the
        # Wong-teaser edge consumer evaluates the 3/7-point crossings against).
        ctx_hash = _context_hash(request.home_context, request.away_context)
        margins = [int(h) - int(a) for h, a in zip(home_scores, away_scores)]
        result["simulation_distributions"].append(
            _distribution_row(
                target="home_margin",
                market="game_margin",
                stat_key=None,
                distribution_type="empirical",
                distribution_params={"source": "negative_binomial_scores", "k": k},
                samples=[float(m) for m in margins],
                n_iterations=request.n_iterations,
                seed=request.seed,
                context_hash=ctx_hash,
                component_version=self.component_version,
            )
        )

        # Empirical pmfs for downstream teaser-leg evaluation (string keys keep
        # the trace JSON-safe), mirroring the soccer backend's margin/total pmfs.
        result["margin_counts"] = _counts(margins)
        result["total_counts"] = _counts(
            [int(h) + int(a) for h, a in zip(home_scores, away_scores)]
        )
        return result
