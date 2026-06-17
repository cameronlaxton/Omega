"""Tennis serve-derived prop backend (aces), Phase 7 M3.

Registered as ``tennis_prop_serve`` — already routed for ATP/WTA
``player_aces`` in ``DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT``; until this
registration the router fallback priced it as a generic distribution.

Model: aces are Bernoulli per service point, so per match

    service_games  ~ Normal(expected_service_games, 1.5)   (volume noise)
    service_points = service_games * E[points per service game]   (exact,
                     from the tennis_markov game-length recursion at SPW%)
    aces           ~ Binomial(service_points, ace_rate)

``ace_rate`` (per service point) comes from ``prior_payload`` (gatherer /
player_context); without it the rate is derived from the projection mean so
the backend still prices on router-equivalent inputs. ``serve_win_pct`` and
``match_format`` / ``expected_total_games`` refine point volume; absent those,
the league's ``avg_total_games`` config supplies it. The result dict matches
the ``run_player_simulation`` contract consumed by ``analyze_player_prop``.
"""

from __future__ import annotations

from typing import Any

from omega.core.simulation.backends import PropSimulationInput

try:  # pragma: no cover
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

_DEFAULT_SPW = 0.62  # volume estimate only — never a probability output
_SERVICE_GAMES_STD = 1.5
_MIN_SERVICE_GAMES, _MAX_SERVICE_GAMES = 6.0, 33.0


class TennisServePropBackend:
    """Binomial-per-service-point sampler for serve-derived props (aces)."""

    backend_name = "tennis_prop_serve"
    component_version = "tennis_prop_serve_v1"

    def run(self, request: PropSimulationInput) -> dict[str, Any]:
        if np is None:  # pragma: no cover - numpy is a hard runtime dep
            raise RuntimeError("numpy is required for tennis_prop_serve")

        from omega.core.config.leagues import get_league_config
        from omega.core.simulation.engine import _percentile
        from omega.core.simulation.tennis_markov import expected_game_points

        prior = request.prior_payload or {}
        config = get_league_config(request.league.upper())

        spw = float(prior.get("serve_win_pct", _DEFAULT_SPW))
        pts_per_game = expected_game_points(spw)

        total_games = prior.get("expected_total_games")
        if total_games is None:
            fmt = str(prior.get("match_format", ""))
            if fmt.endswith("5"):
                total_games = 38.0
            else:
                total_games = float(config.get("avg_total_games", 22.0))
        service_games_mean = float(total_games) / 2.0
        service_points_mean = service_games_mean * pts_per_game

        ace_rate = prior.get("ace_rate")
        if ace_rate is None:
            # Derive from the projection so router-style inputs still price.
            ace_rate = request.projection_mean / max(1.0, service_points_mean)
        ace_rate = max(0.0, min(0.5, float(ace_rate)))

        rng = np.random.default_rng(request.seed)
        n = request.n_iter
        service_games = np.clip(
            rng.normal(service_games_mean, _SERVICE_GAMES_STD, size=n),
            _MIN_SERVICE_GAMES,
            _MAX_SERVICE_GAMES,
        )
        service_points = np.maximum(1, np.rint(service_games * pts_per_game)).astype(int)
        samples_arr = rng.binomial(service_points, ace_rate)
        samples = [float(x) for x in samples_arr]

        line = request.line
        over = sum(1 for x in samples if x > line)
        under = sum(1 for x in samples if x < line)
        push = sum(1 for x in samples if abs(x - line) < 0.5)
        mean = sum(samples) / n
        std = (sum((x - mean) ** 2 for x in samples) / n) ** 0.5
        sorted_samples = sorted(samples)

        return {
            "over_prob": over / n,
            "under_prob": under / n,
            "push_prob": push / n,
            "mean": mean,
            "std": std,
            "p10": _percentile(sorted_samples, 0.10),
            "p50": _percentile(sorted_samples, 0.50),
            "p90": _percentile(sorted_samples, 0.90),
            "distribution_type": "binomial_serve_points",
            "distribution_params": {
                "ace_rate": round(float(ace_rate), 5),
                "service_points_mean": round(float(service_points_mean), 2),
                "points_per_service_game": round(float(pts_per_game), 3),
                "serve_win_pct": round(spw, 4),
            },
            "samples": samples[:100],
        }
