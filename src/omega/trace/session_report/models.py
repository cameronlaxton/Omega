"""Typed contracts for derived Omega session reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReportKind = Literal["intake", "closing-lines", "portfolio", "all"]
ContextMode = Literal["persisted", "persisted+cited"]
ContextSourceType = Literal[
    "trace_evidence",
    "input_context",
    "sidecar_note",
    "closing_line",
    "outcome",
    "context_bundle",
    "missing",
]
ContextBucket = Literal["support", "concern", "missing"]


class ContextBullet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bucket: ContextBucket
    source_type: ContextSourceType
    text: str = Field(min_length=1)
    source_title: str | None = None
    source_url: str | None = None


class EngineView(BaseModel):
    """Persisted engine-owned values surfaced without recomputation."""

    model_probability: str | None = None
    edge: str | None = None
    units: str | None = None
    tier: str | None = None
    calibration_status: str | None = None


class LedgerView(BaseModel):
    status: str
    provenance: str | None = None
    close_attached: bool = False
    clv_available: bool = False
    outcome_attached: bool = False
    ledger_id: str | None = None


class TraceReportCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    session_id: str | None = None
    timestamp: str | None = None
    league: str | None = None
    market: str | None = None
    matchup: str | None = None
    selection: str | None = None
    book: str | None = None
    stake_status: str | None = None
    engine_view: EngineView = Field(default_factory=EngineView)
    ledger_view: LedgerView
    context: list[ContextBullet] = Field(default_factory=list)
    reasoning_narrative: str | None = Field(
        default=None, description="Detailed qualitative narrative reasoning for the prediction"
    )
    trace_quality_status: str
    sidecar_status: str
    evidence_status: str
    calibration_eligible: bool | None = None
    evidence_learning_eligible: bool | None = None


class CoverageRow(BaseModel):
    label: str
    count: int


class IgnoredContextEntry(BaseModel):
    entry_id: str
    reason: str


class AuditRow(BaseModel):
    """Per-bet trust audit row for the end-of-session bet-level audit table.

    Every recommended bet (or research candidate) gets one row so the operator
    can assess at a glance which bets are clean, which are weak, and which are
    research-only without reading individual cards.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    sport: str | None = None
    league: str | None = None
    event_id: str | None = None
    matchup: str | None = None
    market_type: str | None = None
    selection: str | None = None
    line: str | None = None
    odds: str | None = None
    bookmaker: str | None = None
    odds_resolved_at: str | None = None
    output_mode: str | None = None
    confidence_tier: str | None = None
    calibration_eligible: str | None = None
    aggregate_quality: str | None = None
    context_source: str | None = None
    evidence_count: int = 0
    prior_coverage_status: str | None = None
    fallback_usage: str | None = None
    resolver_warnings: list[str] = Field(default_factory=list)
    ledger_status: str | None = None


class IntakeReportData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_kind: Literal["intake"] = "intake"
    generated_at: str
    source_db_path: str
    source_db_fingerprint: str
    source_session_id: str | None = None
    context_mode: ContextMode
    context_bundle_id: str | None = None
    trace_count: int
    ledger_count: int
    sidecar_status: str
    coverage: list[CoverageRow]
    ledger_linkage: list[CoverageRow]
    provenance_split: list[CoverageRow]
    cards: list[TraceReportCard]
    audit_rows: list[AuditRow] = Field(default_factory=list)
    unmatched_ledger_rows: list[str] = Field(default_factory=list)
    ignored_context_entries: list[IgnoredContextEntry] = Field(default_factory=list)
