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
