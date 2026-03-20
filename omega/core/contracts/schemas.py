"""
Strict Pydantic schemas for the OmegaSportsAgent service contract.

These models define the JSON interface between:
  - Backend <-> Frontend (GameAnalysisResponse)
  - Agent <-> Backend (GameAnalysisRequest)
  - External callers <-> API endpoints
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# -- Request Models ----------------------------------------------------------


class MarketQuote(BaseModel):
    """A single normalized market line from any sportsbook."""

    market_type: str = Field(description="e.g. moneyline, spread, total, moneyline_3way, puck_line, run_line, team_total, method_of_victory, set_spread, total_games, map_spread, outright_winner")
    selection: str = Field(description="Human label, e.g. 'Home', 'Over 224.5', 'KO/TKO', 'Top 10'")
    price: float = Field(description="American odds")
    line: Optional[float] = Field(default=None, description="Spread/total line value if applicable")
    segment: str = Field(default="full_game", description="full_game, 1h, 1q, 1p, regulation, first_5_innings, etc.")
    player: Optional[str] = Field(default=None, description="Player name for player props")
    stat_key: Optional[str] = Field(default=None, description="Prop stat key, e.g. pts, pass_yds, aces, kills")
    bookmaker: Optional[str] = Field(default=None, description="Source sportsbook")


class OddsInput(BaseModel):
    """User-supplied or agent-scraped odds for a single game.

    For backward compatibility the flat fields (spread_home, moneyline_home, etc.)
    are preserved.  New callers should prefer the ``markets`` list which can
    represent any market type across all sports.
    """

    # Legacy flat fields (2-way)
    spread_home: Optional[float] = Field(default=None, description="Home spread line (e.g., -3.5)")
    spread_home_price: Optional[float] = Field(default=-110, description="Juice on the home spread (American odds)")
    moneyline_home: Optional[float] = Field(default=None, description="Home moneyline (American odds)")
    moneyline_away: Optional[float] = Field(default=None, description="Away moneyline (American odds)")
    over_under: Optional[float] = Field(default=None, description="Total line (e.g., 224.5)")

    # 3-way moneyline (hockey regulation, soccer)
    moneyline_draw: Optional[float] = Field(default=None, description="Draw moneyline (American odds) for 3-way markets")

    # Normalized market list (preferred path for all new integrations)
    markets: Optional[List[MarketQuote]] = Field(default=None, description="Normalized list of all scraped markets")


class GameAnalysisRequest(BaseModel):
    """Request to analyze a single game matchup.
    Caller (e.g. agent) should supply home_context/away_context when available;
    engine does not fetch live data."""

    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    league: str = Field(description="League identifier: NBA, NFL, MLB, NHL, NCAAB, EPL, UFC, ATP, PGA, CS2, ...")
    odds: Optional[OddsInput] = Field(default=None, description="Market odds (if available)")
    n_iterations: int = Field(default=1000, ge=100, le=100000, description="Simulation iterations")
    home_context: Optional[Dict[str, Any]] = Field(default=None, description="Pre-fetched home team/player A stats")
    away_context: Optional[Dict[str, Any]] = Field(default=None, description="Pre-fetched away team/player B stats")
    seed: Optional[int] = Field(default=None, description="RNG seed for reproducible simulations")


class SlateAnalysisRequest(BaseModel):
    """Request to analyze all games for a league on a given date.
    Caller (e.g. agent) should supply games list; service does not fetch schedule."""

    league: str = Field(description="League identifier")
    date: Optional[str] = Field(default=None, description="Date in YYYY-MM-DD format; defaults to today")
    bankroll: float = Field(default=1000.0, gt=0, description="Bankroll for stake sizing")
    edge_threshold: float = Field(default=0.03, ge=0.0, le=0.5, description="Minimum edge to flag")
    games: Optional[List[Dict[str, Any]]] = Field(default=None, description="Pre-fetched games (home/away/odds); required for analysis")


class PlayerPropRequest(BaseModel):
    """Request to analyze a single player prop."""

    player_name: str
    league: str
    prop_type: str = Field(description="Stat key, e.g. pts, pass_yds, aces, kills, goals")
    line: float = Field(description="The prop line, e.g. 22.5")
    odds_over: Optional[float] = Field(default=None, description="American odds for Over")
    odds_under: Optional[float] = Field(default=None, description="American odds for Under")
    player_context: Optional[Dict[str, Any]] = Field(default=None, description="Player statistical context")
    game_context: Optional[Dict[str, Any]] = Field(default=None, description="Game-level context (opponent, pace, etc.)")
    n_iterations: int = Field(default=5000, ge=100, le=100000)


# -- Response Sub-Models -----------------------------------------------------


class SimulationResult(BaseModel):
    """Core simulation output."""

    iterations: int
    home_win_prob: float = Field(description="Home/Player-A win probability (0-100)")
    away_win_prob: float = Field(description="Away/Player-B win probability (0-100)")
    draw_prob: Optional[float] = Field(default=None, description="Draw probability (0-100), for 3-way markets")
    predicted_spread: float = Field(description="Predicted spread (negative = home favored)")
    predicted_total: float = Field(description="Predicted combined score")
    predicted_home_score: float
    predicted_away_score: float


class EdgeDetail(BaseModel):
    """Edge analysis for one side of a matchup."""

    side: str = Field(description="'home', 'away', or 'draw'")
    team: str
    true_prob: float = Field(description="Raw model probability (0-1)")
    calibrated_prob: float = Field(description="Calibrated probability (0-1)")
    market_implied: float = Field(description="Market implied probability (0-1)")
    edge_pct: float = Field(description="Edge in percentage points")
    ev_pct: float = Field(description="Expected value percentage")
    market_odds: float = Field(description="American odds used for this side")
    confidence_tier: str = Field(description="A (high), B (medium), C (low), or Pass")


class BetSlip(BaseModel):
    """A single actionable bet recommendation."""

    selection: str = Field(description="e.g., 'Lakers -3.5', 'Over 2.5 Goals', 'Fighter A by KO/TKO'")
    odds: float = Field(description="American odds")
    edge_pct: float
    ev_pct: float
    confidence_tier: str
    recommended_units: float
    kelly_fraction: float


class AnalysisMetadata(BaseModel):
    """Metadata about how the analysis was produced."""

    engine_version: str = "2.0-dse"
    calibration_method: str = "combined"
    data_sources: List[str] = Field(default_factory=lambda: ["simulation"])
    archetype: Optional[str] = Field(default=None, description="Sport archetype used for simulation")


# -- Top-Level Response Models -----------------------------------------------


class GameAnalysisResponse(BaseModel):
    """Complete analysis for a single game. The primary JSON contract for the frontend."""

    matchup: str = Field(description="'Away @ Home' display string")
    league: str
    analyzed_at: str = Field(description="ISO 8601 timestamp")
    status: str = Field(description="'success', 'skipped', or 'error'")
    skip_reason: Optional[str] = None
    missing_requirements: Optional[List[str]] = Field(
        default=None,
        description="Machine-readable list of missing inputs the agent should fetch, e.g. ['home_context.off_rating', 'away_context.serve_win_pct']",
    )

    simulation: Optional[SimulationResult] = None
    edges: List[EdgeDetail] = Field(default_factory=list)
    best_bet: Optional[BetSlip] = None
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)


class SlateAnalysisResponse(BaseModel):
    """Analysis for a full slate of games."""

    league: str
    date: str
    total_games: int
    games_analyzed: int
    games_with_edge: int
    analyses: List[GameAnalysisResponse] = Field(default_factory=list)


class PlayerPropResponse(BaseModel):
    """Analysis for a single player prop."""

    player_name: str
    league: str
    prop_type: str
    line: float
    status: str = Field(description="'success', 'skipped', or 'error'")
    skip_reason: Optional[str] = None
    missing_requirements: Optional[List[str]] = None

    over_prob: Optional[float] = Field(default=None, description="Probability of Over (0-1)")
    under_prob: Optional[float] = Field(default=None, description="Probability of Under (0-1)")
    edge_over: Optional[float] = Field(default=None, description="Edge on Over in pct points")
    edge_under: Optional[float] = Field(default=None, description="Edge on Under in pct points")
    recommendation: Optional[str] = Field(default=None, description="'over', 'under', or 'pass'")
    confidence_tier: Optional[str] = None


class ErrorResponse(BaseModel):
    """Structured error returned by the API."""

    error_code: str = Field(description="Machine-readable code: SIM_FAILED, DATA_MISSING, INVALID_INPUT")
    message: str = Field(description="Human-readable error description")
    context: Optional[Dict[str, Any]] = None
    fallback_hint: Optional[str] = Field(default=None, description="Suggestion for the caller")
    missing_requirements: Optional[List[str]] = Field(
        default=None,
        description="Machine-readable missing inputs for agent self-healing",
    )

