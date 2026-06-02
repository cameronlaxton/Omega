"""Typed contracts for the Omega MCP tool surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

MCP_SCHEMA_VERSION = 1


class TraceQueryRequest(BaseModel):
    """Filters accepted by omega_trace_query."""

    db_path: str | None = None
    league: str | None = None
    start: str | None = None
    end: str | None = None
    has_outcome: bool | None = None
    execution_mode: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)


class TraceAttachOutcomeRequest(BaseModel):
    """Outcome attachment is post-decision and must reference an existing trace."""

    trace_id: str
    home_score: int
    away_score: int
    source: str = "mcp"
    db_path: str | None = None


class CalibrationFitPreviewRequest(BaseModel):
    """Dry-run calibration fitting request."""

    db_path: str | None = None
    league: str | None = None
    plane: str = Field(default="game", pattern="^(game|prop)$")
    method: str = Field(default="isotonic", pattern="^(isotonic|shrinkage)$")
    limit: int = Field(default=1000, ge=1, le=10000)


class ReplayBundle(BaseModel):
    """Frozen evidence bundle for replay-plane audit only.

    Replay bundles are not quant benchmark inputs. They exist to audit routing,
    evidence selection, downgrade discipline, refusal discipline, and trace
    completeness using knowable-at-the-time facts.
    """

    schema_version: int = 1
    prompt: str
    facts: list[dict[str, Any]] = Field(default_factory=list)
    source_trace_id: str | None = None
    decision_date: str | None = None
    simulation_seed: int | None = None
    expected_outputs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_live_fetch_flags(self) -> ReplayBundle:
        if self.metadata.get("live_fetch_enabled") is True:
            raise ValueError("replay bundles must disable live fetching")
        for fact in self.facts:
            if fact.get("post_outcome") is True:
                raise ValueError("replay facts must exclude post-outcome information")
            if fact.get("live_fetch") is True:
                raise ValueError("replay facts must not request live fetching")
        return self


class ReplayToolRequest(BaseModel):
    """MCP wrapper around ReplayBundle."""

    bundle: ReplayBundle
    strict: bool = False


class EvidenceRetrieveRequest(BaseModel):
    """Explicit gather slots for evidence retrieval.

    The current MCP adapter does not perform live network retrieval. It returns
    a skipped response with the requested slots so callers can use approved
    evidence channels outside replay or Standard Text mode.
    """

    slots: list[dict[str, Any]] = Field(default_factory=list)


class FlatBetRequest(BaseModel):
    """Log a flat dollar wager into bet_ledger, tied to an existing trace.

    `side` is required (and must be a gradeable side) so the canonical
    selection_descriptor remains settleable by the outcome/regrade pipeline.
    Money fields are filled later at grade time; the bet lands pending.
    """

    trace_id: str = Field(description="FK to traces.trace_id; must already exist")
    market: str = Field(
        description="moneyline | spread | total | player_prop (or player_prop:<stat>)"
    )
    side: str = Field(description="home | away | draw | over | under")
    odds: float = Field(description="American odds, e.g. -110 or +135")
    line: float | None = Field(default=None, description="Point/total; None for moneyline")
    bookmaker: str = Field(default="betmgm", description="Sportsbook the price came from")
    stake_amount: float = Field(default=25.0, gt=0, description="Flat dollar stake")
    selection: str | None = Field(
        default=None, description="Human-readable label; auto-derived if omitted"
    )
    player_name: str | None = Field(default=None, description="Required for player_prop")
    prop_type: str | None = Field(default=None, description="Stat key for player_prop, e.g. pts")
    db_path: str | None = None

    @model_validator(mode="after")
    def _check_side(self) -> FlatBetRequest:
        if self.side.strip().lower() not in {"home", "away", "draw", "over", "under"}:
            raise ValueError("side must be one of: home, away, draw, over, under")
        return self


class PortfolioSummaryRequest(BaseModel):
    """Filters for the financial summary over bet_ledger."""

    league: str | None = None
    sport: str | None = None
    start: str | None = None
    end: str | None = None
    base_bankroll: float = Field(default=1000.0, gt=0, description="Starting bankroll")
    db_path: str | None = None


class GameContextRequest(BaseModel):
    """Resolve the situational context pack for one matchup."""

    league: str
    home_team: str
    away_team: str
    game_date: str = Field(description="Matchup date, YYYY-MM-DD")
    lookback_days: int = Field(default=5, ge=1, le=14, description="Rest-days search window")
