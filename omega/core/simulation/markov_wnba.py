"""WNBA Markov game-simulation backend.

Lowest-risk Phase 7 backend: it reuses the shared possession-level Markov body
(``run_markov_game_simulation``) with WNBA-tuned pace/efficiency defaults from
``omega.core.sport_baselines``. No new simulation math — same engine, different
constants. When the caller supplies full team context it is used verbatim; only
missing required keys are filled from the WNBA baselines, and the result is then
flagged ``context_source="league_default"`` for honest calibration provenance.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from omega.core.simulation.backends import GameSimulationInput
from omega.core.sport_baselines import basketball_context_defaults

_REQUIRED_BASKETBALL_KEYS = ("off_rating", "def_rating", "pace")


def _fill_context(
    context: dict[str, Any] | None, defaults: dict[str, float]
) -> tuple[dict[str, Any] | None, bool]:
    """Return (context, baseline_used) with missing required keys filled.

    ``baseline_used`` is True if any default was injected. If there are no
    defaults available the context is returned untouched so the engine fails
    closed exactly as it does today.
    """
    if not defaults:
        return context, False
    if context is None:
        return dict(defaults), True
    filled = dict(context)
    used = False
    for key in _REQUIRED_BASKETBALL_KEYS:
        if filled.get(key) is None:
            filled[key] = defaults[key]
            used = True
    return filled, used


class MarkovWNBAGameSimulationBackend:
    """Markov backend tuned to WNBA pace/efficiency baselines."""

    backend_name = "markov_state_wnba"
    component_version = "markov_wnba_v1"
    evidence_mode = "markov_transition"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        # Lazy import avoids an import cycle: engine.py imports this module at the
        # bottom to register the backend, after run_markov_game_simulation exists.
        from omega.core.simulation.engine import run_markov_game_simulation

        defaults = basketball_context_defaults("WNBA")
        home_ctx, home_used = _fill_context(request.home_context, defaults)
        away_ctx, away_used = _fill_context(request.away_context, defaults)
        baseline_used = home_used or away_used
        new_request = replace(request, home_context=home_ctx, away_context=away_ctx)
        context_source = "league_default" if baseline_used else "provided"
        return run_markov_game_simulation(
            new_request,
            backend_name=self.backend_name,
            component_version=self.component_version,
            context_source=context_source,
            baseline_used=baseline_used,
        )
