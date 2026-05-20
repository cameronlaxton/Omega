"""
Strategy domain models.

A Strategy is a versioned, auditable unit of analysis logic that can be:
- registered (stored with metadata)
- backtested (replayed against historical data)
- promoted (moved from candidate → staging → production)
- rejected (demoted with documented reason)

Strategies are immutable once registered. New versions create new entries.
"""

from __future__ import annotations

from datetime import datetime, timezone
UTC = timezone.utc
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StrategyStatus(str, Enum):
    """Lifecycle status of a strategy."""
    CANDIDATE = "candidate"      # Just registered, untested
    BACKTESTING = "backtesting"  # Currently being backtested
    STAGING = "staging"          # Passed backtest, awaiting promotion
    PRODUCTION = "production"    # Live / promoted
    REJECTED = "rejected"        # Failed backtest or manually rejected
    ARCHIVED = "archived"        # Retired from production


class StrategyType(str, Enum):
    """What kind of analysis the strategy produces."""
    GAME_EDGE = "game_edge"           # Moneyline/spread/total edges
    PLAYER_PROP = "player_prop"       # Player prop over/under
    SLATE_FILTER = "slate_filter"     # Full-slate edge scanning
    CORRELATION = "correlation"       # Correlated multi-leg
    ANCHOR_PARLAY = "anchor_parlay"   # High-probability prop parlays
    CUSTOM = "custom"                 # User-defined


class BacktestResult(BaseModel):
    """Result of backtesting a strategy against historical data."""
    strategy_id: str
    strategy_version: int
    run_id: str = Field(description="Unique backtest run identifier")
    started_at: str = Field(description="ISO 8601")
    completed_at: str | None = None

    # Universe
    total_games: int = 0
    games_with_edge: int = 0
    total_bets_placed: int = 0

    # Performance
    win_count: int = 0
    loss_count: int = 0
    push_count: int = 0
    win_rate: float = 0.0
    roi_pct: float = Field(default=0.0, description="Return on investment %")
    units_won: float = 0.0
    units_lost: float = 0.0
    net_units: float = 0.0
    max_drawdown_units: float = 0.0

    # Calibration
    avg_edge_pct: float = 0.0
    avg_closing_line_value: float = Field(
        default=0.0,
        description="Average CLV — positive means beating the close",
    )
    brier_score: float | None = Field(
        default=None,
        description="Brier score of predicted probabilities vs outcomes",
    )

    # Breakdown
    results_by_league: dict[str, dict[str, Any]] = Field(default_factory=dict)
    results_by_market: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Phase 6b: Trace linkage
    artifact_schema_version: int = Field(default=1, description="FrozenArtifact schema version used")
    calibration_policy: str = Field(default="static_v1", description="Calibration policy applied")
    trace_ids: list[str] = Field(default_factory=list, description="Source trace IDs from artifacts")

    # Verdict
    passed: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)


class PromotionRecord(BaseModel):
    """Record of a promotion or rejection decision."""
    strategy_id: str
    strategy_version: int
    action: str = Field(description="'promote' or 'reject'")
    from_status: str
    to_status: str
    reason: str
    decided_by: str = Field(default="system", description="'system' or user ID")
    decided_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    backtest_run_id: str | None = None


class StrategyEntry(BaseModel):
    """A registered strategy with full metadata."""
    strategy_id: str = Field(description="Unique identifier (slug)")
    version: int = Field(default=1, ge=1)
    name: str = Field(description="Human-readable name")
    description: str = Field(default="")
    strategy_type: StrategyType = StrategyType.GAME_EDGE
    status: StrategyStatus = StrategyStatus.CANDIDATE

    # What this strategy does
    leagues: list[str] = Field(default_factory=list, description="Applicable leagues")
    markets: list[str] = Field(default_factory=list, description="Market types targeted")
    edge_threshold: float = Field(default=0.03, description="Minimum edge to trigger bet")
    confidence_tiers: list[str] = Field(
        default_factory=lambda: ["A", "B"],
        description="Required confidence tiers",
    )

    # Parameters (strategy-specific config)
    params: dict[str, Any] = Field(default_factory=dict)

    # Provenance
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    created_by: str = Field(default="system")

    # Backtest history
    backtest_results: list[BacktestResult] = Field(default_factory=list)
    promotion_history: list[PromotionRecord] = Field(default_factory=list)

    # Latest backtest summary (denormalized for fast access)
    latest_roi: float | None = None
    latest_win_rate: float | None = None
    latest_clv: float | None = None
