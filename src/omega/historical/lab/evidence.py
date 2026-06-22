"""Promotion evidence + auto-promote — through the single fail-closed gate.

Per the user decision, the lab can auto-promote on all-green — but only by
calling the one existing gate (``CalibrationRegistry.promote`` →
``evaluate_promotion_gates``). There is no second gate and no bypass. Promotion
is **default-off** (the orchestrator passes ``armed=False`` unless ``--auto-promote``)
and **fail-closed**: it refuses on any non-pass parity/CLV artifact, an
INCONCLUSIVE or FAIL live-parity verdict, an unsealed holdout, or a dirty tree.

This module is pure-decision + a single registry side effect:

* :func:`evaluate_decision` — the pre-gate verdict (pure).
* :func:`resolve` — runs the decision, optionally registers+promotes the winner
  through the single gate, and returns the :class:`PromotionEvidenceBundle`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omega.core.calibration.profiles import CalibrationProfile
from omega.core.calibration.promotion import PromotionGateError, artifact_indicates_pass
from omega.core.calibration.registry import CalibrationRegistry
from omega.historical.lab.schemas import (
    ParityVerdict,
    PromotionEvidenceBundle,
    PromotionStatus,
    WinnersCurse,
)
from omega.ops.fit_calibration import _next_version, _unique_profile_id


@dataclass
class EvidenceContext:
    """Everything the promotion decision needs, gathered by the orchestrator."""

    lab_run_id: str
    league: str
    plane: str
    market: str
    winner_profile: CalibrationProfile | None
    winners_curse: WinnersCurse
    holdout_sealed: bool
    attempted_variant_count: int
    working_tree_dirty: bool
    armed: bool = False
    incumbent_id: str | None = None
    backtest_parity: dict[str, Any] | None = None
    backtest_parity_path: str | None = None
    clv_walk_forward: dict[str, Any] | None = None
    clv_walk_forward_path: str | None = None
    live_parity: dict[str, Any] | None = None
    registry_audit_path: str | None = None


def gate_inputs(profile: CalibrationProfile | None) -> dict[str, Any]:
    """The candidate metrics the promotion gate reads — surfaced for auditability."""
    if profile is None:
        return {}
    m = profile.metrics
    return {
        "sample_size": profile.sample_size,
        "brier_score": m.get("brier_score"),
        "calibration_error": m.get("calibration_error"),
        "cv_calibration_error": m.get("cv_calibration_error"),
        "cv_n_folds": m.get("cv_n_folds"),
        "log_loss": m.get("log_loss"),
    }


def _bt_pass(ctx: EvidenceContext) -> bool:
    return artifact_indicates_pass(ctx.backtest_parity)[0]


def _clv_pass(ctx: EvidenceContext) -> bool:
    return artifact_indicates_pass(ctx.clv_walk_forward)[0]


def _live_state(ctx: EvidenceContext) -> str:
    return str((ctx.live_parity or {}).get("state", "")).upper()


def evaluate_decision(ctx: EvidenceContext) -> tuple[PromotionStatus, bool]:
    """Pre-gate verdict + recommendation, independent of the ``armed`` flag.

    Precedence (fail-closed): no winner → not_recommended; unsealed holdout,
    non-pass backtest-parity/CLV, or live FAIL → blocked; live INCONCLUSIVE/missing
    → shadow_only (the normal new-market case); live PASS → evidence_ready, unless
    the working tree is dirty (then blocked — never promote from a dirty tree).
    """
    if ctx.winner_profile is None:
        return "not_recommended", False
    if not ctx.holdout_sealed:
        return "blocked", False
    if not _bt_pass(ctx) or not _clv_pass(ctx):
        return "blocked", False
    live = _live_state(ctx)
    if live == "FAIL":
        return "blocked", False
    if live in ("", "INCONCLUSIVE"):
        return "shadow_only", False
    # live == PASS
    if ctx.working_tree_dirty:
        return "blocked", False
    return "evidence_ready", True


def _verdict(artifact: dict[str, Any] | None) -> ParityVerdict | None:
    """Map a parity/CLV artifact to a display verdict (does NOT gate anything).

    Honours an artifact's own three-state verdict (``state`` for live-parity,
    ``verdict`` for the CLV artifact) so a genuinely INCONCLUSIVE result is shown
    as such rather than collapsed to FAIL. Gating still flows through
    ``artifact_indicates_pass`` in :func:`evaluate_decision`.
    """
    if artifact is None:
        return None
    if "state" in artifact:  # live-parity shape
        state = str(artifact["state"]).upper()
        return state if state in ("PASS", "FAIL", "INCONCLUSIVE") else None  # type: ignore[return-value]
    if "verdict" in artifact:  # CLV artifact carries its own three-state verdict
        verdict = str(artifact["verdict"]).upper()
        return verdict if verdict in ("PASS", "FAIL", "INCONCLUSIVE") else None  # type: ignore[return-value]
    if "no_incumbent_baseline" in (artifact.get("reasons") or []):
        return "no_incumbent"
    return "PASS" if artifact_indicates_pass(artifact)[0] else "FAIL"


def _register_and_promote(
    registry: CalibrationRegistry, ctx: EvidenceContext
) -> tuple[PromotionStatus, str | None, dict[str, Any] | None]:
    """Register the winner as a CANDIDATE then promote via the single gate.

    On PromotionGateError the candidate stays registered (inspectable) and the
    outcome is ``blocked`` with the gate report attached — never an exception leak.
    """
    profile = ctx.winner_profile
    assert profile is not None
    version = _next_version(registry, ctx.league, profile.method, ctx.market)
    profile.version = version
    profile.profile_id = _unique_profile_id(
        profile.method, ctx.league, version, profile.dataset_hash, ctx.market, profile.context_slice
    )
    try:
        registry.register(profile)
        report = registry.promote(
            profile.profile_id,
            confirm_backtest_parity=True,
            parity_evidence=ctx.backtest_parity,
            confirm_clv_non_regression=True,
            clv_evidence=ctx.clv_walk_forward,
        )
        return "promoted", profile.profile_id, report.to_dict()
    except PromotionGateError as exc:
        return "blocked", profile.profile_id, exc.report.to_dict()


def resolve(registry: CalibrationRegistry, ctx: EvidenceContext) -> PromotionEvidenceBundle:
    """Produce the evidence bundle; auto-promote through the single gate when armed."""
    decision, recommended = evaluate_decision(ctx)
    candidate_id: str | None = None
    gate_report: dict[str, Any] | None = None

    if ctx.armed and decision == "evidence_ready" and ctx.winner_profile is not None:
        decision, candidate_id, gate_report = _register_and_promote(registry, ctx)

    return PromotionEvidenceBundle(
        lab_run_id=ctx.lab_run_id,
        candidate_id=candidate_id,
        incumbent_id=ctx.incumbent_id,
        plane=ctx.plane,
        market=ctx.market,
        gate_inputs=gate_inputs(ctx.winner_profile),
        backtest_parity_path=ctx.backtest_parity_path,
        backtest_parity_verdict=_verdict(ctx.backtest_parity),
        clv_walk_forward_path=ctx.clv_walk_forward_path,
        clv_walk_forward_verdict=_verdict(ctx.clv_walk_forward),
        historical_live_parity_verdict=_verdict(ctx.live_parity),
        registry_audit_path=ctx.registry_audit_path,
        holdout_sealed=ctx.holdout_sealed,
        attempted_variant_count=ctx.attempted_variant_count,
        winners_curse=ctx.winners_curse,
        working_tree_dirty=ctx.working_tree_dirty,
        gate_report=gate_report,
        recommended=recommended or decision == "promoted",
        decision=decision,
    )
