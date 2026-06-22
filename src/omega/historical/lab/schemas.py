"""Typed artifacts for a Historical Validation Lab run.

Four net-new artifacts (everything else is reused in place):

* :class:`HistoricalLabRun`        → ``LAB_RUN.json``
* :class:`AttemptedVariantLedger`  → ``ATTEMPTED_VARIANTS.json``
* :class:`PromotionEvidenceBundle` → ``PROMOTION_EVIDENCE.json``
* (the Markdown ``REPORT.md`` is rendered from the three above + the BacktestReport)

All models are Pydantic v2 with ``extra="forbid"`` to fail closed on stray fields.
Probability/betting numbers are **not** redefined here — variants embed values
produced by the single metric path (``omega.historical.metrics`` /
``CalibrationFitter.evaluate``) and the bundle embeds the gate report dict
produced by ``evaluate_promotion_gates``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omega.core.calibration.market import calibration_market_for_plane
from omega.historical.contracts import current_code_version

UTC = timezone.utc

Plane = Literal["game", "prop", "draw"]
PromotionStatus = Literal["evidence_ready", "promoted", "blocked", "not_recommended", "shadow_only"]
VariantStatus = Literal["selected", "rejected", "shadow", "skipped", "error"]
ParityVerdict = Literal["PASS", "FAIL", "INCONCLUSIVE", "no_incumbent"]
RiskLevel = Literal["low", "elevated", "high"]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


class Window(BaseModel):
    """An inclusive ``[start, end]`` date/datetime window (ISO-8601 strings).

    Comparisons are lexicographic, which is correct for zero-padded ISO dates.
    """

    model_config = ConfigDict(extra="forbid")

    start: str = Field(min_length=1)
    end: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> Window:
        if self.end < self.start:
            raise ValueError(f"window end {self.end!r} precedes start {self.start!r}")
        return self


def windows_overlap(a: Window, b: Window) -> bool:
    """True when two inclusive windows share any point (including boundary dates)."""
    return a.start <= b.end and b.start <= a.end


# ---------------------------------------------------------------------------
# Attempted variants
# ---------------------------------------------------------------------------


class AttemptedVariant(BaseModel):
    """One attempted calibration fit in the grid — selected, rejected, or skipped.

    Recording *every* attempt (not just the winner) is the point: the count and
    the winner's validation→holdout degradation quantify winner's-curse risk.

    Validation metrics are the *selection* basis. ``holdout_*`` are populated for
    the **selected winner only** (touch-once discipline; enforced by the ledger).
    ``roi``/``clv`` are nullable and sourced from a walk-forward ``BettingBlock``
    when a variant is escalated — never recomputed here.
    """

    model_config = ConfigDict(extra="forbid")

    variant_id: str
    profile_family: str = Field(description="Calibration method, e.g. isotonic | shrinkage")
    plane: Plane = "game"
    context_slice: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    train_window: Window
    validation_window: Window
    holdout_window: Window

    sample_size: int = Field(default=0, ge=0, description="Train-window pair count")
    n_validation: int = Field(default=0, ge=0)

    # Validation metrics (selection basis) — produced by CalibrationFitter.evaluate.
    brier: float | None = None
    log_loss: float | None = None
    ece: float | None = None
    cv_ece: float | None = None

    # Holdout metrics — populated for the SELECTED winner only (touch-once).
    holdout_brier: float | None = None
    holdout_ece: float | None = None
    n_holdout: int | None = None

    # Betting diagnostics — from a walk-forward BettingBlock when escalated; else null.
    roi: float | None = None
    clv: float | None = None

    status: VariantStatus = "rejected"
    rejection_reason: str | None = None

    dataset_hash: str = ""
    profile_hash: str = ""
    profile_id: str | None = None

    @property
    def touched_holdout(self) -> bool:
        return self.holdout_brier is not None or self.holdout_ece is not None


class AttemptedVariantLedger(BaseModel):
    """The full grid of attempted variants for one lab run + holdout-seal invariants."""

    model_config = ConfigDict(extra="forbid")

    lab_run_id: str
    plane: Plane = "game"
    profile_grid_hash: str = ""
    variants: list[AttemptedVariant] = Field(default_factory=list)

    @property
    def selected(self) -> AttemptedVariant | None:
        for v in self.variants:
            if v.status == "selected":
                return v
        return None

    @property
    def holdout_access_count(self) -> int:
        return sum(1 for v in self.variants if v.touched_holdout)

    @model_validator(mode="after")
    def _seal_invariants(self) -> AttemptedVariantLedger:
        selected = [v for v in self.variants if v.status == "selected"]
        if len(selected) > 1:
            raise ValueError(f"at most one variant may be 'selected'; found {len(selected)}")
        # Touch-once: only the selected winner may carry holdout metrics.
        for v in self.variants:
            if v.touched_holdout and v.status != "selected":
                raise ValueError(
                    f"variant {v.variant_id!r} carries holdout metrics but is not the "
                    "selected winner — holdout must be touched once, for the winner only"
                )
        return self


# ---------------------------------------------------------------------------
# Promotion evidence
# ---------------------------------------------------------------------------


class WinnersCurse(BaseModel):
    """Winner's-curse accounting for the selected variant."""

    model_config = ConfigDict(extra="forbid")

    n_variants: int = Field(default=0, ge=0)
    val_to_holdout_ece_delta: float | None = Field(
        default=None,
        description="holdout_ece - validation_ece for the winner; positive = realized optimism",
    )
    risk: RiskLevel = "low"


class PromotionEvidenceBundle(BaseModel):
    """The evidence the existing fail-closed promote path consumes.

    ``gate_inputs`` are exactly the candidate metrics fed to
    ``evaluate_promotion_gates``; ``gate_report`` is the dict that function
    returns (``GateReport.to_dict()``) when the gate is actually evaluated. This
    bundle is what ``omega-promote-profile --parity-report/--clv-report`` reads
    and what the lab's own auto-promote passes to ``registry.promote()``.
    """

    model_config = ConfigDict(extra="forbid")

    lab_run_id: str
    candidate_id: str | None = None
    incumbent_id: str | None = None
    plane: Plane = "game"
    market: str = "game"

    gate_inputs: dict[str, Any] = Field(default_factory=dict)

    backtest_parity_path: str | None = None
    backtest_parity_verdict: ParityVerdict | None = None
    clv_walk_forward_path: str | None = None
    clv_walk_forward_verdict: ParityVerdict | None = None
    historical_live_parity_verdict: ParityVerdict | None = None
    registry_audit_path: str | None = None

    holdout_sealed: bool = False
    attempted_variant_count: int = Field(default=0, ge=0)
    winners_curse: WinnersCurse | None = None
    working_tree_dirty: bool = False

    gate_report: dict[str, Any] | None = None
    recommended: bool = False
    decision: PromotionStatus = "blocked"


# ---------------------------------------------------------------------------
# Lab run manifest
# ---------------------------------------------------------------------------


class HistoricalLabRun(BaseModel):
    """Top-level manifest binding provenance, dataset, replay, grid, and outcome."""

    model_config = ConfigDict(extra="forbid")

    lab_run_id: str
    created_at: str = Field(default_factory=_utc_now_iso)

    # Provenance
    code_version: str = Field(default_factory=current_code_version)
    git_commit: str = "unknown"
    working_tree_dirty: bool = False

    # Dataset
    dataset_manifest_id: str
    dataset_hash: str = ""
    league: str
    plane: Plane = "game"
    market: str = "game"

    # Replay
    replay_id: str
    replay_db_path: str
    production_db_path: str | None = None
    replay_config_hash: str = ""

    # Grid
    profile_grid_hash: str = ""
    attempted_variant_count: int = Field(default=0, ge=0)

    # Chronological windows
    train_window: Window
    validation_window: Window
    holdout_window: Window
    holdout_sealed: bool = False
    holdout_access_count: int = Field(default=0, ge=0)

    # Promotion
    auto_promote_armed: bool = False
    promotion_candidate_id: str | None = None
    promotion_status: PromotionStatus = "evidence_ready"

    result_paths: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _fill_market(cls, data: Any) -> Any:
        """Derive ``market`` from ``plane`` when omitted (ergonomic + safe)."""
        if isinstance(data, dict) and data.get("market") is None and data.get("plane"):
            data = {**data, "market": calibration_market_for_plane(str(data["plane"]))}
        return data

    @field_validator("replay_db_path")
    @classmethod
    def _reject_production_db(cls, v: str) -> str:
        from omega.paths import is_production_trace_db

        if is_production_trace_db(v):
            raise ValueError(
                "replay_db_path must be an isolated DB, never the production trace DB "
                "(var/omega_traces.db): lab replay traces are synthetic."
            )
        return v

    @model_validator(mode="after")
    def _market_matches_plane(self) -> HistoricalLabRun:
        expected = calibration_market_for_plane(self.plane)
        if self.market != expected:
            raise ValueError(
                f"market={self.market!r} must equal "
                f"calibration_market_for_plane(plane={self.plane!r})={expected!r}"
            )
        return self

    @model_validator(mode="after")
    def _windows_chronological(self) -> HistoricalLabRun:
        # train ≤ validation ≤ holdout, no overlap (adjacency permitted).
        if self.validation_window.start < self.train_window.end:
            raise ValueError("validation_window must start at/after train_window end")
        if self.holdout_window.start < self.validation_window.end:
            raise ValueError("holdout_window must start at/after validation_window end")
        return self

    @model_validator(mode="after")
    def _holdout_access_bound(self) -> HistoricalLabRun:
        if self.holdout_sealed and self.holdout_access_count > 1:
            raise ValueError(
                f"sealed holdout must be accessed at most once; "
                f"holdout_access_count={self.holdout_access_count}"
            )
        return self


# ---------------------------------------------------------------------------
# Cross-artifact consistency
# ---------------------------------------------------------------------------


def assert_consistent(lab_run: HistoricalLabRun, ledger: AttemptedVariantLedger) -> None:
    """Fail closed if the manifest and ledger disagree on count/seal/grid identity.

    The manifest carries summary fields (``attempted_variant_count``,
    ``holdout_access_count``, ``profile_grid_hash``) that MUST match the ledger
    they summarize — otherwise the audit trail is internally contradictory.
    """
    if lab_run.attempted_variant_count != len(ledger.variants):
        raise ValueError(
            f"attempted_variant_count={lab_run.attempted_variant_count} != "
            f"len(ledger.variants)={len(ledger.variants)}"
        )
    if lab_run.holdout_access_count != ledger.holdout_access_count:
        raise ValueError(
            f"holdout_access_count={lab_run.holdout_access_count} != "
            f"ledger.holdout_access_count={ledger.holdout_access_count}"
        )
    if lab_run.profile_grid_hash != ledger.profile_grid_hash:
        raise ValueError(
            f"profile_grid_hash mismatch: lab_run={lab_run.profile_grid_hash!r} "
            f"ledger={ledger.profile_grid_hash!r}"
        )
    if lab_run.lab_run_id != ledger.lab_run_id:
        raise ValueError(
            f"lab_run_id mismatch: lab_run={lab_run.lab_run_id!r} ledger={ledger.lab_run_id!r}"
        )
