"""Simulation backend contracts for deterministic game models.

Every game simulator backend, including the current fast score model and a
future Markov state-transition model, must return the same contract: standard
probability/score fields plus V10 distribution rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

CONTEXT_SOURCES = frozenset({"provided", "league_default", "missing"})
REQUIRED_SUCCESS_FIELDS = frozenset(
    {
        "success",
        "home_team",
        "away_team",
        "league",
        "iterations",
        "home_win_prob",
        "away_win_prob",
        "draw_prob",
        "predicted_home_score",
        "predicted_away_score",
        "predicted_spread",
        "predicted_total",
        "context_source",
        "baseline_used",
        "simulation_distributions",
    }
)
REQUIRED_DISTRIBUTION_FIELDS = frozenset(
    {
        "target",
        "distribution_type",
        "distribution_params",
        "params_schema_version",
        "sample_mean",
        "sample_std",
        "p10",
        "p50",
        "p90",
        "n_iterations",
        "seed",
        "context_hash",
        "component_version",
    }
)


@dataclass(frozen=True)
class GameSimulationInput:
    """Canonical game-simulation input shared by all deterministic backends."""

    home_team: str
    away_team: str
    league: str
    n_iterations: int
    home_context: dict[str, Any] | None = None
    away_context: dict[str, Any] | None = None
    seed: int | None = None
    spread_home: float | None = None
    over_under: float | None = None
    allow_baseline: bool = False
    transition_modifiers: dict[str, float] | None = None


class GameSimulationBackend(Protocol):
    """Protocol implemented by fast score, Markov, or future game backends."""

    backend_name: str
    component_version: str

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        """Return a standard engine result dict satisfying V10 provenance."""


def validate_distribution_rows(rows: Any) -> None:
    """Raise ValueError if V10 distribution rows are absent or malformed."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("successful simulation backend result must include distribution rows")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"distribution row {idx} is not a dict")
        missing = sorted(REQUIRED_DISTRIBUTION_FIELDS - row.keys())
        if missing:
            raise ValueError(f"distribution row {idx} missing required fields: {missing}")
        params = row.get("distribution_params")
        if not isinstance(params, dict):
            raise ValueError(f"distribution row {idx} distribution_params must be a dict")


def enforce_game_backend_contract(result: dict[str, Any]) -> dict[str, Any]:
    """Validate and return one game backend result.

    Skipped/error results may omit distributions, but successful results must
    provide both standard win/score fields and queryable V10 distribution rows.
    """
    if not isinstance(result, dict):
        raise TypeError("simulation backend result must be a dict")
    if result.get("success") is not True:
        return result

    missing = sorted(REQUIRED_SUCCESS_FIELDS - result.keys())
    if missing:
        raise ValueError(f"successful simulation backend result missing fields: {missing}")
    context_source = result.get("context_source")
    if context_source not in CONTEXT_SOURCES:
        raise ValueError(f"invalid context_source={context_source!r}")
    validate_distribution_rows(result.get("simulation_distributions"))
    return result


# ---------------------------------------------------------------------------
# Game-simulation backend registry
# ---------------------------------------------------------------------------
#
# Phase 7 replaces the hardcoded {"fast_score", "markov_state"} switch in
# omega/core/contracts/service.py with a single dictionary lookup. New sport
# backends register at engine-module import time; dispatch is O(1).

GAME_BACKENDS: dict[str, GameSimulationBackend] = {}


def register_game_backend(name: str, backend: GameSimulationBackend) -> None:
    """Register a game-simulation backend under *name*.

    Raises ValueError on duplicate registration so import-time wiring mistakes
    fail loudly rather than silently shadowing an existing backend.
    """
    if name in GAME_BACKENDS:
        raise ValueError(f"game backend {name!r} already registered")
    GAME_BACKENDS[name] = backend


def resolve_game_backend(name: str) -> GameSimulationBackend | None:
    """Return the registered game backend for *name*, or None if unknown."""
    return GAME_BACKENDS.get(name)


# ---------------------------------------------------------------------------
# Prop-simulation backend contract + registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropSimulationInput:
    """Canonical prop-simulation input shared by all deterministic prop backends.

    ``prior_payload`` carries sport-specific priors (NB dispersion ``k``, tennis
    SPW%, soccer xG, etc.) without forcing a schema change on existing callers.
    """

    player_name: str
    league: str
    stat_type: str
    line: float
    projection_mean: float
    n_iter: int
    seed: int | None = None
    projection_std: float | None = None
    prior_payload: dict[str, Any] | None = None


class PropSimulationBackend(Protocol):
    """Protocol implemented by the distribution router, Negative Binomial, or
    future prop backends. Parallel to ``GameSimulationBackend``."""

    backend_name: str
    component_version: str

    def run(self, request: PropSimulationInput) -> dict[str, Any]:
        """Return a prop simulation result dict (over/under probs, percentiles)."""


PROP_BACKENDS: dict[str, PropSimulationBackend] = {}


def register_prop_backend(name: str, backend: PropSimulationBackend) -> None:
    """Register a prop-simulation backend under *name*.

    Raises ValueError on duplicate registration, mirroring
    ``register_game_backend``.
    """
    if name in PROP_BACKENDS:
        raise ValueError(f"prop backend {name!r} already registered")
    PROP_BACKENDS[name] = backend


def resolve_prop_backend(name: str) -> PropSimulationBackend | None:
    """Return the registered prop backend for *name*, or None if unknown."""
    return PROP_BACKENDS.get(name)


# Stat-targets that require a non-default prop backend. Anything not listed
# falls back to the distribution router, which covers all existing NBA/MLB
# props with bit-identical behavior.
DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT: dict[tuple[str, str], str] = {
    # NFL: yardage / longest-play markets are over-dispersed -> Negative Binomial.
    ("NFL", "rushing_yards"): "prop_neg_binom",
    ("NFL", "receiving_yards"): "prop_neg_binom",
    ("NFL", "passing_yards"): "prop_neg_binom",
    ("NFL", "longest_rush"): "prop_neg_binom",
    ("NFL", "longest_reception"): "prop_neg_binom",
    # NFL discrete count stats stay on the Poisson-capable router.
    ("NFL", "passing_tds"): "prop_distribution_router",
    ("NFL", "rushing_tds"): "prop_distribution_router",
    # Tennis serve-derived props handled by tennis-aware backend (Milestone 3).
    ("ATP", "player_aces"): "tennis_prop_serve",
    ("WTA", "player_aces"): "tennis_prop_serve",
}


def resolve_default_prop_backend(league: str, stat_type: str) -> str:
    """Return the default prop-backend name for a (league, stat_type) pair."""
    return DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT.get(
        (league, stat_type), "prop_distribution_router"
    )
