"""
Strict Pydantic schemas for the OmegaSportsAgent service contract.

These models define the JSON interface between:
  - Backend <-> Frontend (GameAnalysisResponse)
  - Agent <-> Backend (GameAnalysisRequest)
  - External callers <-> API endpoints
"""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import BaseModel, Field, model_validator

from omega.core.contracts.evidence import EvidenceSignal

# -- Request Models ----------------------------------------------------------


class MarketQuote(BaseModel):
    """A single normalized market line from any sportsbook."""

    market_type: str = Field(
        description="e.g. moneyline, spread, total, moneyline_3way, puck_line, run_line, team_total, method_of_victory, set_spread, total_games, map_spread, outright_winner"
    )
    selection: str = Field(description="Human label, e.g. 'Home', 'Over 224.5', 'KO/TKO', 'Top 10'")
    price: float = Field(description="American odds")
    line: float | None = Field(default=None, description="Spread/total line value if applicable")
    segment: str = Field(
        default="full_game", description="full_game, 1h, 1q, 1p, regulation, first_5_innings, etc."
    )
    player: str | None = Field(default=None, description="Player name for player props")
    stat_key: str | None = Field(
        default=None, description="Prop stat key, e.g. pts, pass_yds, aces, kills"
    )
    bookmaker: str | None = Field(default=None, description="Source sportsbook")
    source: str | None = Field(
        default=None, description="Provider/source label, e.g. the-odds-api:betmgm"
    )
    event_id: str | None = Field(default=None, description="Provider event id")
    provider_market_key: str | None = Field(default=None, description="Provider-native market key")
    last_update: str | None = Field(
        default=None, description="Provider market last update timestamp"
    )
    snapshot_timestamp: str | None = Field(default=None, description="Provider snapshot timestamp")


class OddsInput(BaseModel):
    """User-supplied or agent-scraped odds for a single game.

    For backward compatibility the flat fields (spread_home, moneyline_home, etc.)
    are preserved.  New callers should prefer the ``markets`` list which can
    represent any market type across all sports.
    """

    # Legacy flat fields (2-way)
    spread_home: float | None = Field(default=None, description="Home spread line (e.g., -3.5)")
    spread_home_price: float | None = Field(
        default=-110, description="Juice on the home spread (American odds)"
    )
    spread_away_price: float | None = Field(
        default=-110, description="Juice on the away spread (American odds)"
    )
    moneyline_home: float | None = Field(default=None, description="Home moneyline (American odds)")
    moneyline_away: float | None = Field(default=None, description="Away moneyline (American odds)")
    over_under: float | None = Field(default=None, description="Total line (e.g., 224.5)")
    total_over_price: float | None = Field(default=-110, description="Over price (American odds)")
    total_under_price: float | None = Field(default=-110, description="Under price (American odds)")

    # 3-way moneyline (hockey regulation, soccer)
    moneyline_draw: float | None = Field(
        default=None, description="Draw moneyline (American odds) for 3-way markets"
    )

    # Exotic 3-way markets (soccer). All additive and optional — absent price
    # means no edge is built for that market.
    dc_home_draw: float | None = Field(
        default=None, description="Double chance Home-or-Draw (1X) price (American odds)"
    )
    dc_home_away: float | None = Field(
        default=None, description="Double chance Home-or-Away (12) price (American odds)"
    )
    dc_away_draw: float | None = Field(
        default=None, description="Double chance Away-or-Draw (X2) price (American odds)"
    )
    dnb_home: float | None = Field(
        default=None, description="Draw-no-bet Home price (American odds); draw voids"
    )
    dnb_away: float | None = Field(
        default=None, description="Draw-no-bet Away price (American odds); draw voids"
    )
    btts_yes: float | None = Field(
        default=None, description="Both teams to score - Yes price (American odds)"
    )
    btts_no: float | None = Field(
        default=None, description="Both teams to score - No price (American odds)"
    )
    correct_score: dict[str, float] | None = Field(
        default=None,
        description="Correct-score prices keyed by 'home-away' scoreline, e.g. {'1-0': 650}",
    )

    # Normalized market list (preferred path for all new integrations)
    markets: list[MarketQuote] | None = Field(
        default=None, description="Normalized list of all scraped markets"
    )


class GameAnalysisRequest(BaseModel):
    """Request to analyze a single game matchup.

    home_context / away_context carry team performance stats (archetype-specific).
    game_context carries situational signals used for calibration slice selection
    and any future game-level adjustments (is_playoff, rest_days, etc.).

    Required keys by archetype (see omega/core/simulation/archetypes.py for full lists):
      Basketball (NBA, NCAAB): off_rating, def_rating, pace
      American football (NFL): off_rating, def_rating
      Baseball (MLB): off_rating, def_rating
      Hockey (NHL): off_rating, def_rating
      Soccer (EPL, MLS, ...): off_rating, def_rating
      Tennis (ATP, WTA): serve_win_pct, return_win_pct
      Golf (PGA): strokes_gained_total
      Combat sports (UFC, Boxing): win_pct, finish_rate
      Esports (CS2): map_win_rate, recent_form
    """

    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    league: str = Field(
        description="League identifier: NBA, NFL, MLB, NHL, NCAAB, EPL, UFC, ATP, PGA, CS2, ..."
    )
    odds: OddsInput | None = Field(default=None, description="Market odds (if available)")
    n_iterations: int = Field(default=1000, ge=100, le=100000, description="Simulation iterations")
    allow_baseline: bool = Field(
        default=False,
        description=(
            "Explicitly allow league-average baseline contexts when home_context or "
            "away_context is absent. Baseline runs are audit-only and are not "
            "calibration eligible."
        ),
    )
    simulation_backend: str = Field(
        default="fast_score",
        description="Deterministic game simulator backend: 'fast_score' or 'markov_state'.",
    )
    home_context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Pre-fetched home team stats. Required keys depend on league archetype "
            "(e.g. off_rating, def_rating, pace for basketball). "
            "Omitting or passing None returns status='skipped' with missing_requirements listing exact keys."
        ),
    )
    away_context: dict[str, Any] | None = Field(
        default=None,
        description=("Pre-fetched away team stats. Same required keys as home_context."),
    )
    game_context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Situational game context for calibration slice selection. Open-ended dict; "
            "supply any applicable signals. "
            "Universal: is_playoff (bool), rest_days (int; 0=B2B), "
            "opponent_def_rank (int), blowout_risk (float 0-1), "
            "pace_adjustment_factor (float). "
            "MLB: park_factor (float), weather_wind_mph (float). "
            "NFL: is_dome (bool), weather_temp_f (float), week_of_season (int). "
            "Any additional matchup context (e.g. matchup_weakness, scheme_advantage) "
            "is preserved in context_labels for calibration fitting."
        ),
    )
    prior_payload: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Game-level dynamic priors for the simulation backend that are not "
            "team-scoped — e.g. Dixon-Coles 'rho' (soccer) or 'pressure_coefficients' "
            "(tennis). Supplied by the gatherer; flows verbatim into "
            "GameSimulationInput.prior_payload. A backend that requires a prior fails "
            "closed (status='skipped', missing_requirements) when it is absent."
        ),
    )
    seed: int | None = Field(default=None, description="RNG seed for reproducible simulations")
    evidence: list[EvidenceSignal] = Field(
        default_factory=list,
        description=(
            "Structured reasoning signals (game/matchup/situational/team_form). "
            "The engine applies known signal types deterministically before simulation "
            "and persists every signal for retrospective scoring; unknown types are "
            "ignored. Defaults to empty — supplying none preserves prior behavior."
        ),
    )

    @model_validator(mode="after")
    def _warn_missing_game_context_keys(self) -> GameAnalysisRequest:
        gc = self.game_context or {}
        missing = [k for k in ("is_playoff", "rest_days") if k not in gc]
        if missing:
            warnings.warn(
                f"GameAnalysisRequest missing game_context keys: {missing}. "
                "Context-slice calibration fitting will be unavailable for this trace.",
                stacklevel=3,
            )
        return self


class SlateAnalysisRequest(BaseModel):
    """Request to analyze all games for a league on a given date.
    Caller (e.g. agent) should supply games list; service does not fetch schedule."""

    league: str = Field(description="League identifier")
    date: str | None = Field(
        default=None, description="Date in YYYY-MM-DD format; defaults to today"
    )
    bankroll: float = Field(default=1000.0, gt=0, description="Bankroll for stake sizing")
    edge_threshold: float = Field(default=0.03, ge=0.0, le=0.5, description="Minimum edge to flag")
    games: list[dict[str, Any]] | None = Field(
        default=None, description="Pre-fetched games (home/away/odds); required for analysis"
    )


class PlayerPropRequest(BaseModel):
    """Request to analyze a single player prop."""

    player_name: str
    league: str
    prop_type: str = Field(description="Stat key, e.g. pts, pass_yds, aces, kills, goals")
    line: float = Field(description="The prop line, e.g. 22.5")
    odds_over: float | None = Field(default=None, description="American odds for Over")
    odds_under: float | None = Field(default=None, description="American odds for Under")
    player_context: dict[str, Any] | None = Field(
        default=None, description="Player statistical context"
    )
    game_context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Game-level context used to adjust player_context means before simulation. "
            "All keys optional; supply only what applies for the sport. "
            "Universal: is_playoff (bool), rest_days (int; 0=B2B), "
            "opponent_def_rank (int 1-30), blowout_risk (float 0-1), "
            "pace_adjustment_factor (float ratio vs league baseline). "
            "MLB: park_factor (float; >1=hitter-friendly), weather_wind_mph (float), "
            "pitcher_k_rate (float), is_starter_probable (bool). "
            "NFL: is_dome (bool), weather_temp_f (float), week_of_season (int)."
        ),
    )
    n_iterations: int = Field(default=5000, ge=100, le=100000)
    seed: int | None = Field(default=None, description="RNG seed for reproducible simulations")
    evidence: list[EvidenceSignal] = Field(
        default_factory=list,
        description=(
            "Structured reasoning signals (player_form/matchup/situational). "
            "The engine applies known signal types deterministically to player_context "
            "means before simulation and persists every signal for retrospective "
            "scoring; unknown types are ignored. Defaults to empty — supplying none "
            "preserves prior behavior."
        ),
    )

    @model_validator(mode="after")
    def _warn_missing_game_context_keys(self) -> PlayerPropRequest:
        gc = self.game_context or {}
        missing = [k for k in ("is_playoff", "rest_days") if k not in gc]
        if missing:
            warnings.warn(
                f"PlayerPropRequest missing game_context keys: {missing}. "
                "Context-slice calibration fitting will be unavailable for this trace.",
                stacklevel=3,
            )
        return self

    # Required because persisted prop traces are graded by
    # (game_date, home_team, away_team). Without these fields the outcome
    # resolver cannot safely attach an ESPN box-score result.
    home_team: str = Field(description="Home team of the game the prop is on", min_length=1)
    away_team: str = Field(description="Away team of the game the prop is on", min_length=1)
    game_date: str = Field(description="ISO date (YYYY-MM-DD) of the game", min_length=10)


# -- Response Sub-Models -----------------------------------------------------


class SimulationResult(BaseModel):
    """Core simulation output."""

    iterations: int
    home_win_prob: float = Field(description="Home/Player-A win probability (0-100)")
    away_win_prob: float = Field(description="Away/Player-B win probability (0-100)")
    draw_prob: float | None = Field(
        default=None, description="Draw probability (0-100), for 3-way markets"
    )
    predicted_spread: float = Field(description="Predicted spread (negative = home favored)")
    predicted_total: float = Field(description="Predicted combined score")
    predicted_home_score: float
    predicted_away_score: float
    context_source: str = Field(
        default="provided", description="'provided' or 'league_default'"
    )
    baseline_used: bool = Field(default=False)
    simulation_backend: str | None = None
    component_version: str | None = None


class EdgeDetail(BaseModel):
    """Edge analysis for one side of a matchup."""

    side: str = Field(description="'home', 'away', or 'draw'")
    team: str
    market: str = Field(default="moneyline", description="moneyline, spread, total, or draw")
    line: float | None = Field(default=None, description="Market line for spread/total edges")
    true_prob: float = Field(description="Raw model probability (0-1)")
    calibrated_prob: float = Field(description="Calibrated probability (0-1)")
    market_implied: float = Field(description="Market implied probability (0-1)")
    edge_pct: float = Field(description="Edge in percentage points")
    ev_pct: float = Field(description="Expected value percentage")
    market_odds: float = Field(description="American odds used for this side")
    confidence_tier: str = Field(description="A (high), B (medium), C (low), or Pass")
    recommended_units: float = Field(default=0.0, description="Kelly-sized stake in units")
    spread_coverage_prob: float | None = Field(
        default=None,
        description="P(covers spread) when market is run-line/puck-line (0-1); None for moneyline edges",
    )
    calibration_audit: CalibrationAudit | None = Field(
        default=None,
        description="Calibration provenance for this edge: which path/profile was used",
    )


class BetSlip(BaseModel):
    """A single actionable bet recommendation."""

    selection: str = Field(
        description="e.g., 'Lakers -3.5', 'Over 2.5 Goals', 'Fighter A by KO/TKO'"
    )
    odds: float = Field(description="American odds")
    edge_pct: float
    ev_pct: float
    confidence_tier: str
    recommended_units: float
    kelly_fraction: float


class CalibrationAudit(BaseModel):
    """Per-edge calibration provenance. Records which path was taken, not inferred."""

    raw_prob: float
    calibrated_prob: float
    league: str | None = None
    plane: str = Field(description="'game' or 'prop'")
    market: str = Field(description="'home', 'away', 'draw', 'over', 'under', 'cover'")
    method_resolved: str | None = None
    profile_id: str | None = None
    context_slice: str | None = None
    resolved_slice: str | None = None
    path: str = Field(
        description=(
            "'profile': learned profile applied; "
            "'base_profile_fallback': fell back from context_slice to base profile; "
            "'static_calibrated': no profile, static policy changed the value; "
            "'static_identity': no profile, within threshold, returned raw unchanged"
        )
    )


class AnalysisMetadata(BaseModel):
    """Metadata about how the analysis was produced."""

    engine_version: str = "2.0-dse"
    data_sources: list[str] = Field(default_factory=lambda: ["simulation"])
    archetype: str | None = Field(default=None, description="Sport archetype used for simulation")
    suppressed_markets: list[str] = Field(
        default_factory=list,
        description="Non-quant list of markets intentionally suppressed by service policy.",
    )


# -- Top-Level Response Models -----------------------------------------------


class GameAnalysisResponse(BaseModel):
    """Complete analysis for a single game. The primary JSON contract for the frontend."""

    matchup: str = Field(description="'Away @ Home' display string")
    league: str
    analyzed_at: str = Field(description="ISO 8601 timestamp")
    status: str = Field(description="'success', 'skipped', or 'error'")
    skip_reason: str | None = None
    missing_requirements: list[str] | None = Field(
        default=None,
        description="Machine-readable list of missing inputs the agent should fetch, e.g. ['home_context.off_rating', 'away_context.serve_win_pct']",
    )

    simulation: SimulationResult | None = None
    edges: list[EdgeDetail] = Field(default_factory=list)
    best_bet: BetSlip | None = None
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)
    context_source: str | None = None
    baseline_used: bool = False
    simulation_backend: str | None = None
    component_version: str | None = None
    simulation_distributions: list[dict[str, Any]] = Field(default_factory=list)


class SlateAnalysisResponse(BaseModel):
    """Analysis for a full slate of games."""

    league: str
    date: str
    total_games: int
    games_analyzed: int
    games_with_edge: int
    analyses: list[GameAnalysisResponse] = Field(default_factory=list)


class PlayerPropResponse(BaseModel):
    """Analysis for a single player prop."""

    player_name: str
    league: str
    prop_type: str
    line: float
    status: str = Field(description="'success', 'skipped', or 'error'")
    skip_reason: str | None = None
    missing_requirements: list[str] | None = None

    over_prob: float | None = Field(default=None, description="Probability of Over (0-1)")
    under_prob: float | None = Field(default=None, description="Probability of Under (0-1)")
    projection_mean: float | None = Field(default=None, description="Simulated distribution mean")
    projection_std: float | None = Field(default=None, description="Simulated distribution std dev")
    projection_p10: float | None = None
    projection_p50: float | None = None
    projection_p90: float | None = None
    distribution_type: str | None = Field(default=None, description="Resolved generator family")
    edge_over: float | None = Field(default=None, description="Edge on Over in pct points")
    edge_under: float | None = Field(default=None, description="Edge on Under in pct points")
    recommendation: str | None = Field(default=None, description="'over', 'under', or 'pass'")
    confidence_tier: str | None = None
    kelly_fraction: float | None = Field(
        default=None,
        description="Scaled Kelly fraction for the recommended prop side; null for pass or unsourced odds",
    )
    recommended_units: float | None = Field(
        default=None,
        description="Kelly-sized stake in units for the recommended prop side; null for pass or unsourced odds",
    )
    bet_side_odds: float | None = Field(
        default=None,
        description="American odds used for the recommended prop side; null for pass or unsourced odds",
    )
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Machine-readable annotations: 'odds_unsourced_over', "
            "'odds_unsourced_under', 'tier_capped_imputation', "
            "'insufficient_real_observations', 'imputed_keys_provided_without_sample_size', "
            "'distribution_override:<family>'."
        ),
    )
    imputed_fraction: float | None = Field(
        default=None,
        description="Fraction of observations marked as imputed (0..1); None when not reported.",
    )
    over_calibration_audit: CalibrationAudit | None = Field(
        default=None,
        description="Calibration provenance for the over probability",
    )
    under_calibration_audit: CalibrationAudit | None = Field(
        default=None,
        description="Calibration provenance for the under probability",
    )
    context_source: str | None = None
    baseline_used: bool = False
    simulation_distributions: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Structured error returned by the API."""

    error_code: str = Field(
        description="Machine-readable code: SIM_FAILED, DATA_MISSING, INVALID_INPUT"
    )
    message: str = Field(description="Human-readable error description")
    context: dict[str, Any] | None = None
    fallback_hint: str | None = Field(default=None, description="Suggestion for the caller")
    missing_requirements: list[str] | None = Field(
        default=None,
        description="Machine-readable missing inputs for agent self-healing",
    )
