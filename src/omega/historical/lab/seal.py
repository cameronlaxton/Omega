"""Holdout-sealing guard: pick the winner on validation, touch the holdout once.

The existing engine enforces chronological no-leak *within a single fit*
(``omega-fit-calibration`` date-window guard; ``walk_forward`` assert). Nothing
enforces the discipline that matters across a *grid*: variants are selected using
the validation window only, and the sealed holdout is evaluated exactly once —
for the selected winner — so its number is an honest out-of-sample estimate
rather than a quantity that leaked into model choice.

This module is the single place that reads the holdout. It (1) selects the winner
from the ledger's validation metrics, (2) refits that one cell on all pre-holdout
data (train + validation) via the single fitter, (3) evaluates it once on the
holdout to produce a promotion-ready :class:`CalibrationProfile`, and (4) reports
winner's-curse from the validation→holdout ECE degradation and the attempt count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from omega.core.calibration.fitter import CalibrationFitter
from omega.core.calibration.market import calibration_market_for_plane
from omega.core.calibration.profiles import CalibrationProfile
from omega.core.calibration.promotion import DEFAULT_ECE_FLOOR
from omega.historical.lab.grid import fit_variant
from omega.historical.lab.schemas import (
    AttemptedVariant,
    AttemptedVariantLedger,
    Window,
    WinnersCurse,
    windows_overlap,
)
from omega.ops.fit_calibration import _decision_date, _extract_plane_pairs, _in_window

SelectionMetric = Literal["cv_ece", "ece"]


@dataclass
class SealResult:
    """Outcome of sealing: updated ledger + the promotion-ready winner profile."""

    ledger: AttemptedVariantLedger
    winner: AttemptedVariant | None
    winner_profile: CalibrationProfile | None
    winners_curse: WinnersCurse
    holdout_sealed: bool

    @property
    def holdout_access_count(self) -> int:
        return self.ledger.holdout_access_count


def select_winner(
    ledger: AttemptedVariantLedger, *, metric: SelectionMetric = "cv_ece"
) -> AttemptedVariant | None:
    """Lowest-calibration-error fitted variant on the validation window.

    Prefers cross-validated ECE (robust) when present, else the single-split
    validation ECE; Brier breaks ties. Only successfully-fit variants
    (``status="rejected"`` with an ECE) are eligible — never holdout numbers.
    """
    eligible = [v for v in ledger.variants if v.status == "rejected" and v.ece is not None]
    if not eligible:
        return None

    def _key(v: AttemptedVariant) -> tuple[float, float]:
        primary = v.cv_ece if (metric == "cv_ece" and v.cv_ece is not None) else v.ece
        return (
            primary if primary is not None else float("inf"),
            v.brier if v.brier is not None else float("inf"),
        )

    return min(eligible, key=_key)


def _winners_curse(
    n_variants: int, val_ece: float | None, holdout_ece: float | None, *, ece_floor: float
) -> WinnersCurse:
    if val_ece is None or holdout_ece is None:
        return WinnersCurse(n_variants=n_variants, val_to_holdout_ece_delta=None, risk="high")
    delta = round(holdout_ece - val_ece, 6)
    if holdout_ece > ece_floor or delta > ece_floor:
        risk: Literal["low", "elevated", "high"] = "high"
    elif delta > 0:
        risk = "elevated"
    else:
        risk = "low"
    return WinnersCurse(n_variants=n_variants, val_to_holdout_ece_delta=delta, risk=risk)


def _assert_chronological(train: Window, validation: Window, holdout: Window) -> None:
    if windows_overlap(train, holdout) or windows_overlap(validation, holdout):
        raise ValueError("holdout window overlaps train/validation — cannot seal")


def seal_winner(
    ledger: AttemptedVariantLedger,
    graded: list[dict[str, Any]],
    *,
    league: str,
    plane: str = "game",
    train_window: Window,
    validation_window: Window,
    holdout_window: Window,
    sport_family: str | None = None,
    selection_metric: SelectionMetric = "cv_ece",
    ece_floor: float = DEFAULT_ECE_FLOOR,
    fitter: CalibrationFitter | None = None,
) -> SealResult:
    """Select the winner and evaluate it once on the sealed holdout.

    Returns a :class:`SealResult`. When no variant is eligible, the winner and
    profile are ``None`` and ``holdout_sealed`` is False (the orchestrator then
    cannot promote). The holdout is read here and nowhere else.
    """
    _assert_chronological(train_window, validation_window, holdout_window)
    fitter = fitter or CalibrationFitter()
    market = calibration_market_for_plane(plane)
    n_total = len(ledger.variants)

    winner = select_winner(ledger, metric=selection_metric)
    if winner is None:
        return SealResult(
            ledger=ledger,
            winner=None,
            winner_profile=None,
            winners_curse=WinnersCurse(n_variants=n_total, risk="high"),
            holdout_sealed=False,
        )

    # Restrict to the winner's slice, then split pre-holdout (train+validation) vs holdout.
    from omega.core.calibration.context_slices import context_slice_for_trace

    slice_traces = [
        t
        for t in graded
        if context_slice_for_trace(t, sport_family=sport_family) == winner.context_slice
    ]
    fit_traces = [
        t
        for t in slice_traces
        if _in_window(_decision_date(t), train_window.start, validation_window.end)
    ]
    hold_traces = [
        t
        for t in slice_traces
        if _in_window(_decision_date(t), holdout_window.start, holdout_window.end)
    ]
    fit_p, fit_o, _ = _extract_plane_pairs(fitter, fit_traces, plane)
    hold_p, hold_o, _ = _extract_plane_pairs(fitter, hold_traces, plane)

    winner_profile: CalibrationProfile | None = None
    holdout_metrics: dict[str, float] | None = None
    if len(fit_p) >= 30 and hold_p:
        winner_profile = fit_variant(
            fitter, winner.profile_family, fit_p, fit_o, league=league, market=market
        )
        winner_profile.context_slice = winner.context_slice
        winner_profile.training_window = f"{train_window.start}..{validation_window.end}"
        holdout_metrics = fitter.evaluate(winner_profile, hold_p, hold_o)
        cv = fitter.cross_validated_ece(
            fit_p, fit_o, league=league, market=market, method=winner.profile_family
        )
        promo_metrics = dict(holdout_metrics)
        promo_metrics.update(cv)
        winner_profile.metrics = promo_metrics

    holdout_ece = holdout_metrics["calibration_error"] if holdout_metrics else None
    curse = _winners_curse(n_total, winner.ece, holdout_ece, ece_floor=ece_floor)

    # Rebuild the ledger: flip the winner, annotate the rejected, leave skipped/error.
    new_variants: list[AttemptedVariant] = []
    for v in ledger.variants:
        if v.variant_id == winner.variant_id and holdout_metrics is not None:
            new_variants.append(
                v.model_copy(
                    update={
                        "status": "selected",
                        "holdout_brier": holdout_metrics["brier_score"],
                        "holdout_ece": holdout_metrics["calibration_error"],
                        "n_holdout": holdout_metrics["n_eval"],
                        "rejection_reason": None,
                    }
                )
            )
        elif v.status == "rejected":
            new_variants.append(v.model_copy(update={"rejection_reason": "not selected"}))
        else:
            new_variants.append(v)

    sealed_ledger = AttemptedVariantLedger(
        lab_run_id=ledger.lab_run_id,
        plane=ledger.plane,
        profile_grid_hash=ledger.profile_grid_hash,
        variants=new_variants,
    )
    return SealResult(
        ledger=sealed_ledger,
        winner=sealed_ledger.selected,
        winner_profile=winner_profile,
        winners_curse=curse,
        holdout_sealed=holdout_metrics is not None,
    )
