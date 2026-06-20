"""Simulation backend contracts for deterministic game models.

Every game simulator backend, including the current fast score model and a
future Markov state-transition model, must return the same contract: standard
probability/score fields plus V10 distribution rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from omega.core.simulation.dispersion import DispersionPolicy

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
    # Game-level dynamic priors that are not team-scoped (soccer Dixon-Coles
    # ``rho``, tennis ``pressure_coefficients``, ...). Parallel to
    # ``PropSimulationInput.prior_payload``. Populated by ``service.analyze_game``
    # from ``GameAnalysisRequest.prior_payload``; backends that require a prior
    # fail closed when it is absent.
    prior_payload: dict[str, Any] | None = None
    # Exact-evaluation mode: when True, parametric backends evaluate market
    # probabilities by summing the closed-form outcome distribution instead of
    # Monte-Carlo sampling it. Removes MC sampling noise (and its optimizer's-curse
    # bias) from backtest/calibration decisions. Backends without a closed form
    # (path-dependent Markov, tennis set-chains) ignore the flag and stay on MC.
    exact: bool = False
    dispersion: DispersionPolicy | None = None
    # Structural soccer competition-strength index by side, e.g.
    # ``{"home": 1.08, "away": 0.95}`` (Issue #22 Feature 1). The soccer backend
    # multiplies a side's attack rate and divides its concede rate by its index
    # BEFORE lambda derivation. None (the default) leaves every backend's output
    # bit-identical; non-soccer backends ignore it.
    competition_strength_index: dict[str, float] | None = None


class GameSimulationBackend(Protocol):
    """Protocol implemented by fast score, Markov, or future game backends."""

    backend_name: str
    component_version: str
    # Evidence routing mode for this backend. "plane_adjustment" applies a
    # PlaneAdjustment to team context; "markov_transition" feeds transition
    # modifiers to the Markov sampler. Dispatch reads this attribute instead of
    # sniffing the backend name, so a new Markov-family backend need not be
    # named ``markov_state*`` to route evidence correctly.
    evidence_mode: str

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


def _require_backend_contract(name: str, backend: Any, required_attrs: tuple[str, ...]) -> None:
    """Validate a backend satisfies its Protocol's required surface.

    Protocols are structurally typed and not enforced at runtime, so a backend
    missing an attribute would only fail later at dispatch. This makes a wiring
    mistake fail loudly at registration (import) time instead, matching the
    fail-loud-on-duplicate philosophy.
    """
    missing = [attr for attr in required_attrs if not hasattr(backend, attr)]
    if missing:
        raise TypeError(f"backend {name!r} missing required attributes: {missing}")
    if not callable(getattr(backend, "run", None)):
        raise TypeError(f"backend {name!r} must define a callable run()")


def register_game_backend(name: str, backend: GameSimulationBackend) -> None:
    """Register a game-simulation backend under *name*.

    Raises TypeError if *backend* does not satisfy the ``GameSimulationBackend``
    surface, and ValueError on duplicate registration, so import-time wiring
    mistakes fail loudly rather than silently shadowing an existing backend or
    deferring an attribute error to dispatch time.
    """
    _require_backend_contract(
        name, backend, ("backend_name", "component_version", "evidence_mode")
    )
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
    # See ``GameSimulationInput.exact``: evaluate over/under/push and percentiles
    # from the closed-form CDF instead of sampling. Honored by parametric prop
    # backends (negative binomial); empirical/MC backends ignore it.
    exact: bool = False
    dispersion: DispersionPolicy | None = None


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

    Raises TypeError on an incomplete backend surface and ValueError on duplicate
    registration, mirroring ``register_game_backend``.
    """
    _require_backend_contract(name, backend, ("backend_name", "component_version"))
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

PROP_STAT_ALIASES_BY_LEAGUE: dict[tuple[str, str], str] = {
    # NFL request/odds surfaces historically use compact market keys, while the
    # nflverse dispersion fitter writes the upstream stat-column names.
    ("NFL", "pass_yds"): "passing_yards",
    ("NFL", "pass_yards"): "passing_yards",
    ("NFL", "passing_yds"): "passing_yards",
    ("NFL", "rush_yds"): "rushing_yards",
    ("NFL", "rush_yards"): "rushing_yards",
    ("NFL", "rushing_yds"): "rushing_yards",
    ("NFL", "rec_yds"): "receiving_yards",
    ("NFL", "rec_yards"): "receiving_yards",
    ("NFL", "receiving_yds"): "receiving_yards",
}


def canonical_prop_stat_type(league: str, stat_type: str) -> str:
    """Return the canonical stat key used by prop routing and prior tables."""
    league_uc = str(league or "").upper()
    stat = str(stat_type or "").strip().lower()
    return PROP_STAT_ALIASES_BY_LEAGUE.get((league_uc, stat), stat)


def resolve_default_prop_backend(league: str, stat_type: str) -> str:
    """Return the default prop-backend name for a (league, stat_type) pair."""
    canonical_stat = canonical_prop_stat_type(league, stat_type)
    return DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT.get(
        (str(league or "").upper(), canonical_stat), "prop_distribution_router"
    )
