"""Pydantic response models for the read-only operator console (Milestone A).

These models shape the GET-only JSON API and feed the server-rendered HTML
templates. Two cross-cutting conventions:

* ``field_sources`` maps a logical field/section name to a *provenance label*
  (see :class:`Source`) naming the canonical DB table the numbers came from.
  Every numeric/betting value surfaced by the console carries one so the page
  can render "source: bet_ledger" next to a figure. This is the data-provenance
  contract from the Milestone-A spec — distinct from a bet row's own
  ``provenance`` column (how the row was created: engine_auto / user_confirmed).
* Session sidecars are session/process narrative only. Their numbers are never
  authoritative; ``Source.SIDECAR_PROCESS`` marks anything sourced from prose.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Source:
    """Provenance labels: the canonical DB origin of a displayed value.

    Anything not in this set that ends up labelled ``SIDECAR_PROCESS`` is, by
    construction, non-canonical session narrative — never a numeric authority.
    """

    DB_TRACE_PAYLOAD = "db_trace_payload"  # traces.full_trace JSON blob
    BET_LEDGER = "bet_ledger"  # bet_ledger table
    OUTCOMES = "outcomes"  # outcomes table (final scores / result)
    PROP_OUTCOMES = "prop_outcomes"  # prop_outcomes table
    EVIDENCE_SIGNALS = "evidence_signals"  # evidence_signals table
    SIMULATION_DISTRIBUTIONS = "simulation_distributions"  # simulation_distributions table
    CLOSING_LINES = "closing_lines"  # closing_lines table
    TRACE_QA_VERDICTS = "trace_qa_verdicts"  # trace_qa_verdicts audit table
    SIDECAR_PROCESS = "sidecar_process"  # session sidecar narrative; NON-canonical


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class Pagination(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int
    page_size: int
    total: int
    total_pages: int
    has_prev: bool
    has_next: bool
    # True when a bounded read scan (not a true SQL OFFSET) limited the candidate
    # set before in-memory filtering/paging. Surfaced honestly so the operator
    # knows older rows may exist beyond the scan window (Milestone-B sharpening).
    scan_capped: bool = False


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class TraceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    timestamp: str | None = None
    league: str | None = None
    sport: str | None = None
    kind: str | None = None
    matchup: str | None = None
    session_id: str | None = None
    execution_mode: str | None = None
    aggregate_quality: float | None = None
    confidence_tiers: list[str] = Field(default_factory=list)
    markets: list[str] = Field(default_factory=list)
    has_outcome: bool = False
    field_sources: dict[str, str] = Field(default_factory=dict)


class TraceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[TraceRow]
    pagination: Pagination
    filters: dict[str, Any] = Field(default_factory=dict)


class TraceDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    timestamp: str | None = None
    league: str | None = None
    sport: str | None = None
    kind: str | None = None
    matchup: str | None = None
    session_id: str | None = None
    execution_mode: str | None = None
    aggregate_quality: float | None = None
    # Canonical trace JSON blob (traces.full_trace).
    payload: dict[str, Any] = Field(default_factory=dict)
    recommendations: Any | None = None
    predictions: Any | None = None
    # Linked DB-backed children (each from its own table — see field_sources).
    evidence_signals: list[dict[str, Any]] = Field(default_factory=list)
    simulation_distributions: list[dict[str, Any]] = Field(default_factory=list)
    outcome: dict[str, Any] | None = None
    prop_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    bets: list[dict[str, Any]] = Field(default_factory=list)
    closing_lines: list[dict[str, Any]] = Field(default_factory=list)
    qa_verdict: dict[str, Any] | None = None
    field_sources: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bets
# ---------------------------------------------------------------------------


class BetRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ledger_id: str
    trace_id: str | None = None
    bet_date: str | None = None
    league: str | None = None
    sport: str | None = None
    matchup: str | None = None
    market: str | None = None
    bookmaker: str | None = None
    selection: str | None = None
    line: float | None = None
    odds: float | None = None
    stake_amount: float | None = None
    net_pnl: float | None = None
    status: str | None = None
    # bet_ledger.provenance column (engine_auto | backfill | user_confirmed | …):
    # how the row was created, NOT a data-source label.
    provenance: str | None = None
    field_sources: dict[str, str] = Field(default_factory=dict)


class BetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[BetRow]
    pagination: Pagination
    filters: dict[str, Any] = Field(default_factory=dict)


class BetDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ledger_id: str
    trace_id: str | None = None
    # Full decoded bet_ledger row (ledger fields + settlement/PnL + staking +
    # correlation group). All numbers here are bet_ledger-sourced.
    ledger: dict[str, Any] = Field(default_factory=dict)
    staking: dict[str, Any] = Field(default_factory=dict)
    correlation_group: str | None = None
    # Recommendation values from the LINKED trace's DB payload only (never
    # assumed to live in bet_ledger). None when no trace is linked/found.
    linked_trace_id: str | None = None
    linked_trace_recommendations: Any | None = None
    field_sources: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    file_name: str
    sidecar_valid: bool
    opened_at: str | None = None
    closed_at: str | None = None
    purpose: str | None = None
    league: str | None = None
    model_version: str | None = None
    quality_gate_status: str = "unknown"
    db_trace_count: int = 0
    event_count: int = 0


class SessionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[SessionSummary]
    pagination: Pagination


class AuditEventView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: str | None = None
    event_type: str | None = None
    step: str | None = None
    status: str | None = None
    notes: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    bugs: list[str] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None


class SessionDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    file_name: str
    sidecar_valid: bool
    sidecar_error: str | None = None
    # Sidecar metadata + process/narrative blocks (NON-canonical, sidecar_process).
    metadata: dict[str, Any] = Field(default_factory=dict)
    pipeline_status: dict[str, Any] = Field(default_factory=dict)
    next_required_action: str | None = None
    exec_stats: dict[str, Any] = Field(default_factory=dict)
    agent_notes: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    bugs: list[str] = Field(default_factory=list)
    audit_events: list[AuditEventView] = Field(default_factory=list)
    quality_gate_status: str = "unknown"
    # Canonical numbers come from these DB traces, never from the sidecar prose.
    db_traces: list[TraceRow] = Field(default_factory=list)
    field_sources: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    read_only: bool = True
    db_path: str
    db_source: str
    trace_count: int
    schema_version: int
    sessions_dir: str
    milestone: str = "phase8-A"
