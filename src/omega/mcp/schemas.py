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


_VALID_LEAGUES = {"nba", "wnba", "mlb", "soccer", "props"}


class FetchOutcomesRequest(BaseModel):
    """Batch outcome gathering across leagues (wraps fetch_outcomes_all).

    Defaults to every league. To exclude soccer (commonly future-dated
    fixtures), pass ``leagues`` without ``"soccer"`` — there is no implicit
    exclusion. ``dry_run`` lists what would run without attaching outcomes.
    """

    leagues: list[str] | None = Field(
        default=None,
        description="Subset of: nba, wnba, mlb, soccer, props. None = all.",
    )
    since: str | None = Field(default=None, description="Start date YYYY-MM-DD")
    until: str | None = Field(default=None, description="End date YYYY-MM-DD")
    dry_run: bool = Field(default=False, description="Print commands without attaching")
    db_path: str | None = None

    @model_validator(mode="after")
    def _check_leagues(self) -> FetchOutcomesRequest:
        if self.leagues is not None:
            bad = [lg for lg in self.leagues if lg.lower() not in _VALID_LEAGUES]
            if bad:
                raise ValueError(
                    f"Unknown league(s): {', '.join(bad)}. "
                    f"Valid: {', '.join(sorted(_VALID_LEAGUES))}"
                )
            self.leagues = [lg.lower() for lg in self.leagues]
        return self


class SettleBetsRequest(BaseModel):
    """Settle pending bet_ledger rows with attached outcomes (wraps settle_bets).

    ``apply=False`` (default) is a dry run that scans and reports without
    writing. Mirrors the settle_bets CLI provenance gate.
    """

    apply: bool = Field(default=False, description="Write settled rows; default is dry-run")
    league: str | None = None
    sport: str | None = None
    provenance: str = Field(
        default="user_confirmed",
        pattern="^(user_confirmed|engine_auto|backfill|all)$",
        description="Ledger provenance to settle",
    )
    start: str | None = None
    end: str | None = None
    limit: int = Field(default=100000, ge=1)
    db_path: str | None = None


class TraceVoidPropRequest(BaseModel):
    """Record a DNP / no-action void for a player prop absent from the box score.

    Use when a player did not play (injury, scratch, ejection before qualifying)
    so the prop has no gradeable stat line. Records a ``void`` prop outcome so
    settlement returns VOID (stake returned, net 0) instead of leaving the bet
    pending forever or mis-grading it as a loss.
    """

    trace_id: str
    player_name: str
    stat_type: str = Field(description="Stat key, e.g. points, hits, strikeouts")
    side: str = Field(default="over", description="Recorded side; void result is side-agnostic")
    reason: str = Field(default="dnp", description="Why the prop is void, e.g. dnp, scratched")
    source: str = "mcp"
    db_path: str | None = None

    @model_validator(mode="after")
    def _check_side(self) -> TraceVoidPropRequest:
        if self.side.strip().lower() not in {"over", "under"}:
            raise ValueError("side must be 'over' or 'under'")
        self.side = self.side.strip().lower()
        return self


class GameContextRequest(BaseModel):
    """Resolve the situational context pack for one matchup."""

    league: str
    home_team: str
    away_team: str
    game_date: str = Field(description="Matchup date, YYYY-MM-DD")
    lookback_days: int = Field(default=5, ge=1, le=14, description="Rest-days search window")
