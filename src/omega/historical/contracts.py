"""Typed contracts for the historical replay + walk-forward backtest module.

All contracts are Pydantic models with ``extra="forbid"`` to fail closed on
unexpected source fields. Stable hashes use canonical JSON + sha256, mirroring
``omega.trace.market_snapshot.MarketSnapshot.stable_id``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def stable_hash(payload: Any, *, length: int = 20) -> str:
    """Deterministic short hash of a JSON-serializable payload.

    Keys are sorted and separators are tight so logically-equal payloads hash
    identically regardless of dict ordering or whitespace.
    """
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def current_code_version() -> str:
    """Best-effort code version used in determinism keys and report provenance."""
    try:
        from importlib.metadata import version

        return f"omega-{version('omega')}"
    except Exception:  # pragma: no cover - editable installs without metadata
        return "omega-unknown"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Dataset-level contracts
# ---------------------------------------------------------------------------


class HistoricalEvent(BaseModel):
    """A single historical fixture, identity-resolved and as-of safe.

    ``home_team``/``away_team`` are the *nominal* home/away exactly as the source
    encodes them; neutral sites are flagged via ``is_neutral_site`` and never
    swapped (home advantage is adjusted downstream instead).
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(description="Stable id for the fixture (source-derived)")
    league: str = Field(description="League code, e.g. NFL, EPL, ATP")
    sport_family: str = Field(description="Archetype name, e.g. american_football, soccer")
    season: str | None = Field(default=None, description="Season label, e.g. '2023' or '2023-24'")
    start_time: str = Field(description="Event start, ISO 8601 UTC")
    home_team: str = Field(description="Canonical nominal home team/participant")
    away_team: str = Field(description="Canonical nominal away team/participant")
    is_neutral_site: bool = Field(default=False, description="True → adjust home edge, never swap")
    is_playoff: bool = Field(default=False)
    identity_status: str = Field(
        default="complete", description="complete | missing — both sides resolved?"
    )
    raw_home: str | None = Field(default=None, description="Original source home name (provenance)")
    raw_away: str | None = Field(default=None, description="Original source away name (provenance)")
    source_name: str = Field(description="Dataset/source identifier")
    source_row_ref: str | None = Field(default=None, description="e.g. 'file.csv:42' provenance")

    def stable_event_id(self) -> str:
        """Deterministic event id from identity + kickoff (used when source lacks one)."""
        return stable_hash(
            {
                "league": self.league.upper(),
                "start_time": self.start_time,
                "home": self.home_team,
                "away": self.away_team,
            }
        )


class HistoricalPropOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_name: str
    stat_type: str
    stat_value: float


class HistoricalOutcome(BaseModel):
    """Final result for an event. Post-game by definition — never enters a snapshot."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    home_score: int | None = None
    away_score: int | None = None
    result: str | None = Field(default=None, description="home_win | away_win | draw")
    prop_outcomes: list[HistoricalPropOutcome] = Field(default_factory=list)
    source: str = Field(default="historical_dataset")

    @staticmethod
    def derive_result(home_score: int | None, away_score: int | None) -> str | None:
        if home_score is None or away_score is None:
            return None
        if home_score > away_score:
            return "home_win"
        if away_score > home_score:
            return "away_win"
        return "draw"


# ---------------------------------------------------------------------------
# Odds-level contracts
# ---------------------------------------------------------------------------


class OddsQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(description="moneyline | spread | total | home_draw_away | ...")
    selection_descriptor: str = Field(description="Canonical snake_case selection")
    odds: float = Field(description="American odds")
    line: float | None = Field(default=None, description="Point/total; None for moneyline")
    book: str | None = Field(default=None)
    timestamp: str | None = Field(default=None, description="ISO 8601 of the quote")


class OddsObservation(BaseModel):
    """A single raw odds observation emitted by an odds adapter.

    Observations are grouped per event and resolved into opening/decision/closing
    tiers by ``odds_snapshots.build_odds_snapshot`` under an as-of policy. The
    ``tier_hint`` carries a source-declared tier (e.g. football-data closing
    odds) when one exists; otherwise tiering is decided purely by timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    event_key: str = Field(description="Normalization key linking to a HistoricalEvent")
    market: str
    selection_descriptor: str
    odds: float
    line: float | None = None
    book: str | None = None
    timestamp: str | None = Field(default=None, description="ISO 8601 of the quote, if known")
    tier_hint: Literal["opening", "closing"] | None = None


class HistoricalMarketSnapshot(BaseModel):
    """Opening / decision / closing odds for an event.

    ``closing`` quotes are CLV-only and must never be read for bet selection.
    A snapshot with no decision quotes still permits probability-only replay
    (``missing_odds=True``).
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    decision_time: str = Field(description="As-of cutoff for decision odds")
    decision_policy: str = Field(default="latest_before_decision")
    opening: list[OddsQuote] = Field(default_factory=list)
    decision: list[OddsQuote] = Field(default_factory=list)
    closing: list[OddsQuote] = Field(default_factory=list)
    missing_odds: bool = Field(default=False)
    odds_snapshot_hash: str = Field(default="")

    def compute_hash(self) -> str:
        return stable_hash(
            {
                "event_id": self.event_id,
                "decision_time": self.decision_time,
                "decision_policy": self.decision_policy,
                "decision": [q.model_dump() for q in self.decision],
                "opening": [q.model_dump() for q in self.opening],
                # closing intentionally excluded — it is CLV-only and must not
                # change the identity of the *decision* snapshot.
            }
        )

    def decision_quote(self, market: str, selection_descriptor: str) -> OddsQuote | None:
        for q in self.decision:
            if q.market == market and q.selection_descriptor == selection_descriptor:
                return q
        return None

    def closing_quote(self, market: str, selection_descriptor: str) -> OddsQuote | None:
        for q in self.closing:
            if q.market == market and q.selection_descriptor == selection_descriptor:
                return q
        return None


# ---------------------------------------------------------------------------
# Feature snapshot
# ---------------------------------------------------------------------------


class HistoricalFeatureSnapshot(BaseModel):
    """Pre-decision Omega-compatible context for one event.

    Carries the exact shapes ``analyze()`` consumes: ``home_context``,
    ``away_context``, ``game_context`` (always including ``is_playoff`` and
    ``rest_days``), and ``context_labels`` for calibration slicing.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    league: str
    sport_family: str
    decision_time: str
    home_context: dict[str, Any] = Field(default_factory=dict)
    away_context: dict[str, Any] = Field(default_factory=dict)
    game_context: dict[str, Any] = Field(default_factory=dict)
    context_labels: dict[str, Any] = Field(default_factory=dict)
    context_source: str = Field(
        default="provided", description="provided | backfilled | default"
    )
    is_stale: bool = Field(default=False, description="True when as-of data is older than policy")
    as_of: str | None = Field(default=None, description="Latest source row date used")
    feature_snapshot_hash: str = Field(default="")

    def compute_hash(self) -> str:
        return stable_hash(
            {
                "event_id": self.event_id,
                "league": self.league.upper(),
                "decision_time": self.decision_time,
                "home_context": self.home_context,
                "away_context": self.away_context,
                "game_context": self.game_context,
                "context_labels": self.context_labels,
            }
        )


# ---------------------------------------------------------------------------
# Replay configuration + manifest
# ---------------------------------------------------------------------------


class ReplayConfig(BaseModel):
    """Deterministic configuration for a replay run."""

    model_config = ConfigDict(extra="forbid")

    dataset_manifest_id: str
    backtest_db_path: str = Field(description="Isolated sqlite path; never the production DB")
    session_id: str = Field(default="historical-replay")
    bankroll: float = Field(default=1000.0, gt=0)
    n_iterations: int = Field(default=1000, ge=100, le=100000)
    simulation_backend: str = Field(default="fast_score")
    decision_odds_policy: str = Field(default="latest_before_decision")
    enable_staking: bool = Field(default=False)
    leakage_policy: Literal["skip", "fail"] = Field(default="skip")
    code_version: str = Field(default_factory=current_code_version)
    seed_namespace: str = Field(default="omega-historical-replay-v1")

    def config_hash(self) -> str:
        """Stable hash over the determinism-relevant fields (excludes db path)."""
        return stable_hash(
            {
                "dataset_manifest_id": self.dataset_manifest_id,
                "bankroll": self.bankroll,
                "n_iterations": self.n_iterations,
                "simulation_backend": self.simulation_backend,
                "decision_odds_policy": self.decision_odds_policy,
                "enable_staking": self.enable_staking,
                "leakage_policy": self.leakage_policy,
                "code_version": self.code_version,
                "seed_namespace": self.seed_namespace,
            }
        )


class ReplayEventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    trace_id: str | None = None
    decision_time: str
    feature_snapshot_hash: str | None = None
    odds_snapshot_hash: str | None = None
    leakage_status: str = Field(description="clean | skipped | failed")
    leakage_reasons: list[str] = Field(default_factory=list)
    identity_status: str = Field(default="complete", description="complete | missing | skipped")
    context_source: str = Field(default="provided", description="provided | backfilled | default")
    is_stale: bool = False
    missing_odds: bool = False
    ledger_ids: list[str] = Field(default_factory=list)


class ReplayTraceManifest(BaseModel):
    """Per-run audit manifest binding every replayed event to its trace + hashes."""

    model_config = ConfigDict(extra="forbid")

    replay_id: str
    dataset_manifest_id: str
    league: str
    code_version: str = Field(default_factory=current_code_version)
    config_hash: str = ""
    created_at: str = Field(default_factory=_utc_now_iso)
    records: list[ReplayEventRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Walk-forward configuration + results
# ---------------------------------------------------------------------------


class WalkForwardConfig(BaseModel):
    """Chronological walk-forward configuration. No random train/test split."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["expanding", "rolling"] = "expanding"
    train_window_days: int | None = Field(
        default=None, description="Rolling train width in days; None = all prior (expanding)"
    )
    test_window_days: int = Field(default=30, ge=1)
    step_days: int | None = Field(default=None, description="Fold step; defaults to test_window_days")
    min_train_samples: int = Field(default=50, ge=1)
    min_slice_samples: int = Field(default=30, ge=1)
    markets: list[str] = Field(
        default_factory=lambda: ["game"],
        description="Calibration planes to evaluate: game, prop, draw",
    )
    slices: list[str] = Field(
        default_factory=list, description="Context-slice labels to attempt per fold"
    )


class FrozenProfileRef(BaseModel):
    """Snapshot + hash of a calibration profile frozen for one fold (auditable)."""

    model_config = ConfigDict(extra="forbid")

    market: str
    context_slice: str | None = None
    method: str
    profile_id: str
    profile_hash: str
    sample_size: int
    params_snapshot: dict[str, Any] = Field(default_factory=dict)


class MetricBlock(BaseModel):
    """Raw-vs-calibrated probability metrics for one market/slice."""

    model_config = ConfigDict(extra="forbid")

    raw_brier: float | None = None
    calibrated_brier: float | None = None
    raw_ece: float | None = None
    calibrated_ece: float | None = None
    raw_log_loss: float | None = None
    calibrated_log_loss: float | None = None
    n: int = 0


class BettingBlock(BaseModel):
    """Betting performance — kept strictly separate from probability accuracy."""

    model_config = ConfigDict(extra="forbid")

    roi: float | None = None
    net_pnl: float | None = None
    hit_rate: float | None = None
    profit_factor: float | None = None
    max_drawdown: float | None = None
    avg_clv: float | None = None
    n_bets: int = 0


class HealthBlock(BaseModel):
    """Visibility rates for leakage / identity / odds / calibration fallbacks."""

    model_config = ConfigDict(extra="forbid")

    missing_odds_rate: float = 0.0
    leakage_skip_count: int = 0
    identity_failure_count: int = 0
    fallback_profile_rate: float = 0.0
    default_context_rate: float = 0.0
    stale_context_rate: float = 0.0


class FoldResult(BaseModel):
    """One walk-forward fold: chronological train → frozen profiles → test eval."""

    model_config = ConfigDict(extra="forbid")

    fold_index: int
    train_start: str | None = None
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    metrics_by_market: dict[str, MetricBlock] = Field(default_factory=dict)
    betting: BettingBlock | None = None
    health: HealthBlock = Field(default_factory=HealthBlock)
    frozen_profiles: list[FrozenProfileRef] = Field(default_factory=list)


class BacktestReport(BaseModel):
    """Top-level backtest report. Probability accuracy and ROI are separate."""

    model_config = ConfigDict(extra="forbid")

    manifest_id: str = Field(description="Dataset manifest id (provenance)")
    replay_id: str
    league: str
    walk_forward_config: WalkForwardConfig
    folds: list[FoldResult] = Field(default_factory=list)
    aggregate_metrics_by_market: dict[str, MetricBlock] = Field(default_factory=dict)
    aggregate_betting: BettingBlock | None = None
    aggregate_health: HealthBlock = Field(default_factory=HealthBlock)
    code_version: str = Field(default_factory=current_code_version)
    generated_at: str = Field(default_factory=_utc_now_iso)


# ---------------------------------------------------------------------------
# Cross-layer audit bridge
# ---------------------------------------------------------------------------


class ReplayCandidateSelection(BaseModel):
    """Audit bridge: trace ↔ calibration profile ↔ odds ↔ staking ↔ ledger.

    One row per selected market on a replayed event. ``clv`` is reporting-only
    and is computed from the closing line; it never influences selection.
    """

    model_config = ConfigDict(extra="forbid")

    replay_id: str
    event_id: str
    trace_id: str
    market: str
    selection_descriptor: str
    raw_prob: float
    calibrated_prob: float | None = None
    profile_id: str = Field(default="none", description="'none' when no profile applied")
    profile_hash: str = Field(default="")
    decision_odds: float | None = None
    decision_line: float | None = None
    book: str | None = None
    decision_time: str
    edge: float | None = None
    ev: float | None = None
    units: float | None = None
    kelly_fraction: float | None = None
    stake_amount: float | None = None
    capped_by: list[str] = Field(default_factory=list)
    ledger_id: str | None = None
    clv: float | None = Field(default=None, description="Closing-line value; reporting only")
