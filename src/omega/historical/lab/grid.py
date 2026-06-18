"""Variant grid driver — fit a sweep of calibration candidates for selection.

Every ``(method, context_slice)`` cell is fit on the **train** window and
evaluated on the **validation** window via the single fitter
(:class:`~omega.core.calibration.fitter.CalibrationFitter`). The grid is
*selection-only*: it never reads the holdout window and never touches the
calibration registry (no ``register``/``promote``) — so running a grid of N
variants cannot pollute ``profiles.json`` with N-1 junk candidates. Pair
extraction, windowing, and context slicing reuse the exact production helpers
from ``omega.ops.fit_calibration`` / ``omega.core.calibration`` so a lab variant
matches what a production fit of the same cell would compute.

The output is an :class:`AttemptedVariantLedger`; winner selection and the
single sealed holdout evaluation live in :mod:`omega.historical.lab.seal`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omega.core.calibration.context_slices import (
    BASE_CONTEXT_SLICE,
    context_slice_for_trace,
)
from omega.core.calibration.fitter import CalibrationFitter
from omega.core.calibration.market import calibration_market_for_plane
from omega.historical.contracts import stable_hash
from omega.historical.lab.schemas import (
    AttemptedVariant,
    AttemptedVariantLedger,
    Window,
)
from omega.ops.fit_calibration import _decision_date, _extract_plane_pairs, _in_window

DEFAULT_METHODS: tuple[str, ...] = ("isotonic", "shrinkage")
_MIN_FIT = 30  # CalibrationFitter._MIN_SAMPLES floor


def _grid_spec(
    league: str,
    plane: str,
    methods: Sequence[str],
    slices: Sequence[str],
    train: Window,
    validation: Window,
    holdout: Window,
) -> dict[str, Any]:
    return {
        "league": league.upper(),
        "plane": plane,
        "methods": sorted(methods),
        "slices": sorted(slices),
        "train": [train.start, train.end],
        "validation": [validation.start, validation.end],
        "holdout": [holdout.start, holdout.end],
    }


def profile_grid_hash(
    league: str,
    plane: str,
    methods: Sequence[str],
    slices: Sequence[str],
    train: Window,
    validation: Window,
    holdout: Window,
) -> str:
    """Deterministic identity for a grid specification."""
    return stable_hash(_grid_spec(league, plane, methods, slices, train, validation, holdout))


def _compact_params(profile: Any) -> dict[str, Any]:
    """A light, ledger-friendly param summary (the full isotonic map is heavy)."""
    if profile.method == "isotonic":
        return {"n_bins": len(profile.params.get("calibration_map", {}))}
    return dict(profile.params)


def fit_variant(
    fitter: CalibrationFitter, method: str, preds: list[float], outs: list[int], *, league: str, market: str
):
    """Fit one method on the train pairs via the single fitter. Raises on bad method."""
    if method == "isotonic":
        return fitter.fit_isotonic(preds, outs, league=league, market=market)
    if method == "shrinkage":
        return fitter.fit_shrinkage(
            preds, outs, league=league, market=market, eligible_sample_size=len(preds)
        )
    raise ValueError(f"Unknown calibration method: {method!r}")


def _group_by_slice(
    graded: list[dict[str, Any]], slices: Sequence[str], sport_family: str | None
) -> dict[str | None, list[dict[str, Any]]]:
    """Partition traces into the base group + each requested slice group.

    Mirrors ``omega-fit-calibration`` base-only grouping: each trace maps to its
    single highest-precedence slice; the base group holds only base-slice traces.
    """
    groups: dict[str | None, list[dict[str, Any]]] = {BASE_CONTEXT_SLICE: []}
    for s in slices:
        groups.setdefault(s, [])
    for t in graded:
        label = context_slice_for_trace(t, sport_family=sport_family)
        if label is BASE_CONTEXT_SLICE:
            groups[BASE_CONTEXT_SLICE].append(t)
        elif label in groups:
            groups[label].append(t)
    return groups


def run_grid(
    graded: list[dict[str, Any]],
    *,
    lab_run_id: str,
    league: str,
    plane: str = "game",
    train_window: Window,
    validation_window: Window,
    holdout_window: Window,
    methods: Sequence[str] = DEFAULT_METHODS,
    slices: Sequence[str] = (),
    sport_family: str | None = None,
    fitter: CalibrationFitter | None = None,
) -> AttemptedVariantLedger:
    """Fit + evaluate every ``(method, slice)`` cell; return the attempted-variant ledger.

    Successful fits get a provisional ``status="rejected"`` (resting state); the
    seal step flips exactly one to ``"selected"``. Insufficient data → ``skipped``;
    a fitter error → ``error``. No variant is ``selected`` and none carries holdout
    metrics here — that is the seal's job.
    """
    fitter = fitter or CalibrationFitter()
    market = calibration_market_for_plane(plane)
    grid_hash = profile_grid_hash(
        league, plane, methods, slices, train_window, validation_window, holdout_window
    )
    groups = _group_by_slice(graded, slices, sport_family)

    variants: list[AttemptedVariant] = []
    for slice_name, slice_traces in groups.items():
        train_traces = [
            t for t in slice_traces if _in_window(_decision_date(t), train_window.start, train_window.end)
        ]
        val_traces = [
            t
            for t in slice_traces
            if _in_window(_decision_date(t), validation_window.start, validation_window.end)
        ]
        train_p, train_o, _ = _extract_plane_pairs(fitter, train_traces, plane)
        val_p, val_o, _ = _extract_plane_pairs(fitter, val_traces, plane)

        for method in methods:
            base = dict(
                variant_id=f"{plane}_{method}_{slice_name or 'base'}",
                profile_family=method,
                plane=plane,
                context_slice=slice_name,
                train_window=train_window,
                validation_window=validation_window,
                holdout_window=holdout_window,
                sample_size=len(train_p),
                n_validation=len(val_p),
            )

            if len(train_p) < _MIN_FIT or not val_p:
                variants.append(
                    AttemptedVariant(
                        **base,
                        status="skipped",
                        rejection_reason=(
                            f"insufficient pairs: train={len(train_p)} (need {_MIN_FIT}), "
                            f"validation={len(val_p)}"
                        ),
                    )
                )
                continue

            try:
                profile = fit_variant(fitter, method, train_p, train_o, league=league, market=market)
                metrics = fitter.evaluate(profile, val_p, val_o)
                cv = fitter.cross_validated_ece(
                    train_p, train_o, league=league, market=market, method=method
                )
            except ValueError as exc:
                variants.append(AttemptedVariant(**base, status="error", rejection_reason=str(exc)))
                continue

            variants.append(
                AttemptedVariant(
                    **base,
                    params=_compact_params(profile),
                    brier=metrics["brier_score"],
                    log_loss=metrics["log_loss"],
                    ece=metrics["calibration_error"],
                    cv_ece=(cv.get("cv_calibration_error") if cv.get("cv_n_folds") else None),
                    status="rejected",  # provisional; seal selects the winner
                    dataset_hash=profile.dataset_hash,
                    profile_hash=stable_hash(
                        {
                            "league": league.upper(),
                            "market": market,
                            "plane": plane,
                            "method": method,
                            "context_slice": slice_name,
                            "params": profile.params,
                        }
                    ),
                    profile_id=profile.profile_id,
                )
            )

    return AttemptedVariantLedger(
        lab_run_id=lab_run_id, plane=plane, profile_grid_hash=grid_hash, variants=variants
    )
