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
    CALIBRATION_REGISTRY = "calibration_registry"  # calibration profiles.json registry
    SIGNAL_PERFORMANCE = "signal_performance"  # signal_performance scoring table
    RUNTIME = "runtime"  # console runtime/store metadata (db path, schema, counts)
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
# Normalized read views (Milestone B.1)
#
# Pydantic mirrors of the read-only dataclasses in ``omega.ui.normalizers``. The
# normalizer owns ALL interpretation/derivation; these models only carry its
# output over the JSON API and into templates with strict, JSON-safe typing (no
# arbitrary types). Conversion is dataclass -> asdict -> model_validate in the
# service layer — normalizers must not import these view models (it already
# imports ``Source`` from this module, so the dependency stays one-directional).
# ---------------------------------------------------------------------------

# Scalar union for an extracted display value. ``bool`` first so a real bool is
# not coerced to int under Pydantic's union handling.
ScalarValue = bool | int | float | str | None


class ExtractedFieldModel(BaseModel):
    """One extracted scalar with provenance (mirror of ``ExtractedField``)."""

    model_config = ConfigDict(extra="forbid")

    value: ScalarValue = None
    source: str
    source_path: str | None = None
    computed: bool = False
    display: str | None = None


class OperatorWarningModel(BaseModel):
    """Operator-facing warning (mirror of ``OperatorWarning``)."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: str  # "info" | "warn" | "fail" | "unknown"
    message: str
    source_path: str | None = None
    suggested_action: str | None = None


class NormalizedRecommendationModel(BaseModel):
    """One normalized recommendation (mirror of ``NormalizedRecommendation``)."""

    model_config = ConfigDict(extra="forbid")

    is_primary: bool
    rank: int | None = None
    market: ExtractedFieldModel
    selection: ExtractedFieldModel
    line: ExtractedFieldModel
    odds: ExtractedFieldModel
    raw_probability: ExtractedFieldModel
    calibrated_probability: ExtractedFieldModel
    implied_probability: ExtractedFieldModel
    engine_edge: ExtractedFieldModel
    computed_edge: ExtractedFieldModel
    kelly_fraction: ExtractedFieldModel
    recommended_units: ExtractedFieldModel
    raw_confidence_tier: ExtractedFieldModel
    display_confidence_band: ExtractedFieldModel
    warnings: list[OperatorWarningModel] = Field(default_factory=list)


class EvidenceCoverageModel(BaseModel):
    """Evidence coverage metrics (mirror of ``EvidenceCoverage``) — counts only."""

    model_config = ConfigDict(extra="forbid")

    total_signals: int
    applied_signals: int
    shadow_signals: int
    signals_with_confidence: int
    avg_confidence: float | None = None
    signal_types_present: list[str] = Field(default_factory=list)
    warnings: list[OperatorWarningModel] = Field(default_factory=list)


class EvidenceCoverageSummary(BaseModel):
    """Compact per-row evidence counts for the traces table (not a score)."""

    model_config = ConfigDict(extra="forbid")

    total_signals: int = 0
    applied_signals: int = 0
    shadow_signals: int = 0


class TraceRecommendationViewModel(BaseModel):
    """Complete normalized recommendation view (mirror of ``TraceRecommendationView``)."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    kind: str
    recommendations: list[NormalizedRecommendationModel] = Field(default_factory=list)
    evidence_coverage: EvidenceCoverageModel
    raw_payload_available: bool = False


class SessionHealthViewModel(BaseModel):
    """Computed session health (mirror of ``SessionHealthView``)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    quality_gate_status: str
    total_traces: int
    traces_with_outcomes: int
    traces_with_bets: int
    traces_with_evidence: int
    traces_zero_evidence: int
    total_evidence_signals: int
    avg_evidence_signals_per_trace: float
    evidence_coverage_ratio: float
    sidecar_valid: bool
    assumption_count: int
    bug_count: int
    audit_event_count: int
    failed_audit_events: int
    pipeline_steps_failed: list[str] = Field(default_factory=list)
    warnings: list[OperatorWarningModel] = Field(default_factory=list)


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
    # Compact evidence counts (Milestone B.1). Populated only for the visible
    # (paginated) window of the trace list; None elsewhere. Backward compatible.
    evidence_coverage: EvidenceCoverageSummary | None = None
    field_sources: dict[str, str] = Field(default_factory=dict)


class TraceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[TraceRow]
    pagination: Pagination
    filters: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 1


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
    # Normalized recommendation read view (Milestone B.1). New key only; all the
    # raw fields above are unchanged for backward compatibility. Carries the
    # selection-aware recommendations and evidence coverage. None when the
    # normalizer produced nothing (e.g. malformed payload).
    recommendation_view: TraceRecommendationViewModel | None = None
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
    schema_version: int = 1


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
    schema_version: int = 1


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
    # Computed session health beyond quality_gate_status (Milestone B.1). New key
    # only; aggregated from DB-backed per-trace facts + sidecar counts/flags
    # (never sidecar prose numbers).
    health: SessionHealthViewModel | None = None
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


# ---------------------------------------------------------------------------
# Diagnostics + Calibration Status (Milestone B.2)
#
# Read-only operator views over runtime/registry/scoring health. All numbers are
# DB- or registry-sourced; warnings reuse OperatorWarningModel; provenance is
# labelled via the Source.* constants above.
# ---------------------------------------------------------------------------


class CalibrationSummary(BaseModel):
    """Counts of calibration profiles by lifecycle status (registry-sourced)."""

    model_config = ConfigDict(extra="forbid")

    registry_available: bool = False
    total_profiles: int = 0
    production: int = 0
    candidate: int = 0
    archived: int = 0
    rejected: int = 0
    leagues_with_production: list[str] = Field(default_factory=list)


class SignalScoringSummary(BaseModel):
    """Freshness of the most recent signal-performance scoring run."""

    model_config = ConfigDict(extra="forbid")

    last_scored_at: str | None = None
    rows_in_latest_run: int = 0
    league_count: int = 0


class DiagnosticsView(BaseModel):
    """System-health snapshot: runtime DB, calibration registry, signal scoring."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"  # "ok" | "degraded"
    db_path: str
    db_source: str
    schema_version: int
    trace_count: int
    session_count: int
    bet_count: int
    bet_count_capped: bool = False
    latest_trace_ts: str | None = None
    calibration: CalibrationSummary
    signal_scoring: SignalScoringSummary
    generated_at: str
    warnings: list[OperatorWarningModel] = Field(default_factory=list)
    field_sources: dict[str, str] = Field(default_factory=dict)
    schema_version_response: int = 1


class CalibrationProfileRow(BaseModel):
    """One calibration profile, flattened for display (registry-sourced)."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    league: str
    version: int
    method: str
    market: str = "game"
    context_slice: str | None = None
    status: str  # candidate | production | archived | rejected
    is_active: bool = False  # matches registry.get_production() for its (league, slice, market)
    sample_size: int = 0
    brier: float | None = None
    calibration_error: float | None = None
    log_loss: float | None = None
    n_eval: int | None = None
    training_window: str | None = None
    created_at: str | None = None
    promoted_at: str | None = None
    field_sources: dict[str, str] = Field(default_factory=dict)


class CalibrationStatusView(BaseModel):
    """All calibration profiles with the active one marked (registry-sourced)."""

    model_config = ConfigDict(extra="forbid")

    registry_available: bool = False
    rows: list[CalibrationProfileRow] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[OperatorWarningModel] = Field(default_factory=list)
    schema_version: int = 1


# ---------------------------------------------------------------------------
# Signal Performance + Review Queue + Market Movement/CLV (Milestone B.3)
# ---------------------------------------------------------------------------


class SignalPerformanceRow(BaseModel):
    """One row of the latest signal-performance scoring run (DB-sourced)."""

    model_config = ConfigDict(extra="forbid")

    signal_type: str | None = None
    source: str | None = None
    obs_window: str | None = None
    league: str | None = None
    sample_size: int | None = None
    direction_correct: int | None = None
    direction_accuracy: float | None = None
    mean_confidence: float | None = None
    realized_hit_rate: float | None = None
    calibration_gap: float | None = None
    brier: float | None = None
    scored_at: str | None = None


class SignalPerformanceView(BaseModel):
    """The most recent signal-performance scoring run (signal_performance table)."""

    model_config = ConfigDict(extra="forbid")

    rows: list[SignalPerformanceRow] = Field(default_factory=list)
    last_scored_at: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[OperatorWarningModel] = Field(default_factory=list)
    schema_version: int = 1


class ReviewItem(BaseModel):
    """A single sample item inside a review bucket (link + label only)."""

    model_config = ConfigDict(extra="forbid")

    kind: str  # "trace" | "bet" | "session"
    id: str
    label: str | None = None
    detail: str | None = None
    href: str | None = None


class ReviewBucket(BaseModel):
    """A category of items needing operator attention (bounded-scan count)."""

    model_config = ConfigDict(extra="forbid")

    code: str
    title: str
    severity: str  # "info" | "warn" | "fail"
    count: int
    scan_capped: bool = False
    source: str
    items: list[ReviewItem] = Field(default_factory=list)  # bounded sample


class ReviewQueueView(BaseModel):
    """Operator work buckets aggregated from existing DB/sidecar reads."""

    model_config = ConfigDict(extra="forbid")

    buckets: list[ReviewBucket] = Field(default_factory=list)
    warnings: list[OperatorWarningModel] = Field(default_factory=list)
    schema_version: int = 1


class ClvRow(BaseModel):
    """One bet's closing-line value (taken price vs closing price)."""

    model_config = ConfigDict(extra="forbid")

    ledger_id: str
    trace_id: str | None = None
    bet_date: str | None = None
    league: str | None = None
    matchup: str | None = None
    market: str | None = None
    selection: str | None = None
    status: str | None = None
    taken_odds: float | None = None
    closing_odds: float | None = None
    taken_implied: float | None = None  # computed: raw implied (incl. vig)
    closing_implied: float | None = None  # computed: raw implied (incl. vig)
    clv_points: float | None = None  # computed: closing_implied - taken_implied
    beat_close: bool | None = None
    closing_source: str | None = None
    field_sources: dict[str, str] = Field(default_factory=dict)


class ClvSummary(BaseModel):
    """CLV coverage + aggregate across the scanned bets."""

    model_config = ConfigDict(extra="forbid")

    bets_scanned: int = 0
    with_closing_line: int = 0
    beat_close: int = 0
    avg_clv_points: float | None = None


class ClvView(BaseModel):
    """Closing-line-value report joining bet_ledger with closing_lines."""

    model_config = ConfigDict(extra="forbid")

    rows: list[ClvRow] = Field(default_factory=list)
    summary: ClvSummary
    filters: dict[str, Any] = Field(default_factory=dict)
    scan_capped: bool = False
    warnings: list[OperatorWarningModel] = Field(default_factory=list)
    schema_version: int = 1


# ---------------------------------------------------------------------------
# Edge Scanner (V2) — recent DB-backed recommendations with HONEST columns.
#
# This is NOT a live feed and NOT a fabricated "value score". Every column is a
# real engine value from the trace payload (or an explicitly computed/labeled
# derivation). ``recorded_price`` is the price recorded on the trace at decision
# time — Omega has no live multi-book quote, so we never claim "best price".
# ---------------------------------------------------------------------------


class EdgeScannerRow(BaseModel):
    """One recent recommendation, surfaced with honest, source-labeled columns."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    timestamp: str | None = None
    league: str | None = None
    matchup: str | None = None
    kind: str | None = None
    market: ExtractedFieldModel
    selection: ExtractedFieldModel
    # Market-aware model output (never forced into "line" language):
    # Spread→Model Spread, Total→Model Total, Moneyline→Model Probability,
    # Props→Model Projection, Unknown→Model Output. ``model_output_is_pct`` tells
    # the template to format the value as a probability percentage.
    model_output_label: str
    model_output: ExtractedFieldModel
    model_output_is_pct: bool = False
    # The price recorded on the trace at decision time (NOT a live/best quote).
    recorded_price: ExtractedFieldModel
    edge: ExtractedFieldModel  # engine_edge (real; db_trace_payload)
    # Server-formatted edge so neither template nor JS assumes a unit. The
    # source value may be a fraction (0.04) or already a percent (4.0); the
    # formatter normalizes both to a percent string. None when no edge exists.
    edge_display: str | None = None
    edge_positive: bool | None = None
    # Confidence via the source hierarchy: model tier → calibrated-prob bucket
    # (computed) → unavailable. ``confidence_source`` names which tier produced it.
    confidence: str | None = None
    confidence_source: str = "unavailable"
    confidence_computed: bool = False
    # Data quality: clean | warn | fail | unknown (from trace_quality + evidence).
    data_quality: str = "unknown"
    data_quality_detail: str | None = None
    has_outcome: bool = False
    field_sources: dict[str, str] = Field(default_factory=dict)


class EdgeScannerView(BaseModel):
    """Recent DB-backed recommendations ranked by engine edge (read-only)."""

    model_config = ConfigDict(extra="forbid")

    rows: list[EdgeScannerRow] = Field(default_factory=list)
    scan_capped: bool = False
    generated_at: str
    filters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[OperatorWarningModel] = Field(default_factory=list)
    schema_version: int = 1


# ---------------------------------------------------------------------------
# Calibration chart (V2) — a SINGLE-UNIT, server-computed time series.
#
# Doctrine: the chart never mixes incompatible units. It declares its Y-axis
# ``unit`` explicitly, and the SVG geometry (polylines/dots) is computed
# server-side so the template only drops coordinates in — no math in Jinja/JS,
# and the frontend never computes a protected betting value.
# ---------------------------------------------------------------------------


class CalibrationChartPoint(BaseModel):
    """One time bucket: the model and market values share ONE unit."""

    model_config = ConfigDict(extra="forbid")

    label: str
    model_value: float | None = None
    market_value: float | None = None
    n: int = 0


class CalibrationChartDot(BaseModel):
    """Pre-laid-out hover anchor (pixel coords) for one model point."""

    model_config = ConfigDict(extra="forbid")

    cx: float
    cy: float
    label: str
    model_value: float | None = None
    market_value: float | None = None


class CalibrationChart(BaseModel):
    """Single-unit model-vs-market time series with server-computed geometry."""

    model_config = ConfigDict(extra="forbid")

    mode: str  # e.g. "implied_prob_model_vs_market"
    unit: str  # explicit Y-axis unit, e.g. "implied probability (%)"
    y_label: str
    model_series_label: str
    market_series_label: str
    points: list[CalibrationChartPoint] = Field(default_factory=list)
    # Server-computed SVG geometry (no client math):
    view_w: int = 680
    view_h: int = 220
    model_polyline: str = ""
    market_polyline: str = ""
    dots: list[CalibrationChartDot] = Field(default_factory=list)
    y_min: float = 0.0
    y_max: float = 1.0
    sample: int = 0
    filters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[OperatorWarningModel] = Field(default_factory=list)
    schema_version: int = 1


# ---------------------------------------------------------------------------
# Command Center (V2 landing) — a SUMMARY composed from the existing read views.
#
# Each panel carries its own PanelState so the landing degrades one panel at a
# time (failure isolation) instead of 500-ing the whole page. The CommandCenter
# introduces NO new DB access: it only aggregates HealthResponse / DiagnosticsView
# / ReviewQueueView / ClvView (and, later, the Edge Scanner + calibration chart).
# ---------------------------------------------------------------------------


class PanelState(BaseModel):
    """Per-panel render state for the Command Center (failure isolation).

    ``state`` is one of ``data`` | ``empty`` | ``degraded``. ``message`` carries
    the honest empty/degraded copy; ``source`` is the provenance label so the
    panel can show where its numbers came from.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    title: str
    state: str  # "data" | "empty" | "degraded"
    message: str | None = None
    source: str | None = None


class CommandCenterView(BaseModel):
    """V2 landing summary: bounded, per-panel-isolated aggregation of reads."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    panels: dict[str, PanelState] = Field(default_factory=dict)
    health: HealthResponse | None = None
    review: ReviewQueueView | None = None
    diagnostics: DiagnosticsView | None = None
    clv: ClvView | None = None
    scanner: EdgeScannerView | None = None
    calibration_chart: CalibrationChart | None = None
    review_count: int = 0
    schema_version: int = 1
