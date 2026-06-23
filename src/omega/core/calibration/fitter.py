"""
Calibration fitter — fit profiles from graded traces.

Consumes (prediction, outcome) pairs extracted from TraceStore.get_graded_traces()
and produces CalibrationProfile candidates. Pure Python, no sklearn dependency.

Supports:
- Isotonic fitting via Pool-Adjacent-Violators (PAV) algorithm
- Shrinkage fitting via grid search on Brier score
- Held-out evaluation with Brier, ECE, and log loss
- Candidate-vs-incumbent comparison with promotion recommendation
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np

from omega.core.calibration.profiles import CalibrationProfile

UTC = timezone.utc

logger = logging.getLogger("omega.core.calibration.fitter")

# Minimum samples required for fitting
_MIN_SAMPLES = int(os.environ.get("OMEGA_MIN_SAMPLES", 30))

# Epsilon for log loss clipping
_LOG_EPS = 1e-15

# Isotonic output clip: prevents extreme 0.0/1.0 PAV outputs from tail bins with
# sparse training samples, which otherwise produce large ECE gaps on the holdout.
# 0.15 is consistent with empirical upset rates across supported sports (MLB ~15–30%,
# NBA ~15–25%); draws in soccer are naturally lower but 0.15 is a safe floor for
# game-plane moneyline calibration. Revisit per sport family if draw or tennis
# calibration profiles show systematic floor violations.
_ISO_CLIP = 0.15

# Phase 7 red-team finding 4: early low-liquidity captures carry phantom edges and
# must not contaminate the production calibration profile. Traces tagged with this
# liquidity profile are excluded from fitting by default; opting in routes them
# into a dedicated context slice rather than the production base slice.
EARLY_MARKET_SLICE = "early_market_low_liq"


def _is_early_market_trace(trace: dict[str, Any]) -> bool:
    """True if a graded trace was produced from an early low-liquidity capture.

    Checked against either a ``context_labels.liquidity_profile`` label or a
    top-level ``liquidity_profile`` field so it works regardless of how the
    capture path tagged the trace.
    """
    labels = trace.get("context_labels") or {}
    return (
        labels.get("liquidity_profile") == EARLY_MARKET_SLICE
        or trace.get("liquidity_profile") == EARLY_MARKET_SLICE
    )


class CalibrationFitter:
    """Fit calibration profiles from historical prediction-outcome pairs."""

    # ------------------------------------------------------------------
    # Extract (prediction, outcome) pairs from graded traces
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pairs(
        graded_traces: list[dict[str, Any]],
        include_early_snapshots: bool = False,
    ) -> tuple[list[float], list[int]]:
        """Extract (predicted_prob, actual_outcome) pairs from game-level
        graded traces.

        Prop traces are skipped silently — they have neither ``home_win_prob``
        nor ``_outcome``. Use :func:`extract_prop_pairs` for prop calibration.

        Args:
            graded_traces: Output of TraceStore.get_graded_traces().
                Each dict has ``predictions`` and ``_outcome`` keys.
            include_early_snapshots: When False (default), traces tagged
                ``liquidity_profile == EARLY_MARKET_SLICE`` are excluded so
                phantom early-line edges never drift the production profile.

        Returns:
            (predictions, outcomes) — parallel lists of floats and 0/1 ints.
        """
        predictions: list[float] = []
        outcomes: list[int] = []

        for trace in graded_traces:
            if not include_early_snapshots and _is_early_market_trace(trace):
                continue
            preds = trace.get("predictions")
            outcome = trace.get("_outcome")
            if not preds or not outcome:
                continue

            home_win_prob = preds.get("home_win_prob")
            result = outcome.get("result")
            if home_win_prob is None or result is None:
                continue

            # Normalize to [0, 1]
            prob = float(home_win_prob) / 100.0 if home_win_prob > 1.0 else float(home_win_prob)
            prob = max(0.0, min(1.0, prob))

            actual = 1 if result == "home_win" else 0
            predictions.append(prob)
            outcomes.append(actual)

        return predictions, outcomes

    @staticmethod
    def extract_draw_pairs(
        graded_traces: list[dict[str, Any]],
        include_early_snapshots: bool = False,
    ) -> tuple[list[float], list[int]]:
        """Extract (predicted_draw_prob, actual_draw) pairs from 3-way game
        graded traces (soccer, hockey regulation).

        Prediction is the trace's ``draw_prob``; outcome is 1 when the game
        ended in a draw (``result == "draw"``) and 0 otherwise. Traces without
        a ``draw_prob`` (non-3-way sports) are skipped silently.

        Args:
            graded_traces: Output of TraceStore.get_graded_traces().

        Returns:
            (predictions, outcomes) — parallel lists of floats and 0/1 ints.
        """
        predictions: list[float] = []
        outcomes: list[int] = []

        for trace in graded_traces:
            if not include_early_snapshots and _is_early_market_trace(trace):
                continue
            preds = trace.get("predictions")
            outcome = trace.get("_outcome")
            if not preds or not outcome:
                continue

            draw_prob = preds.get("draw_prob")
            result = outcome.get("result")
            if draw_prob is None or result is None:
                continue

            prob = float(draw_prob) / 100.0 if draw_prob > 1.0 else float(draw_prob)
            prob = max(0.0, min(1.0, prob))

            actual = 1 if result == "draw" else 0
            predictions.append(prob)
            outcomes.append(actual)

        return predictions, outcomes

    @staticmethod
    def extract_prop_pairs(
        graded_traces: list[dict[str, Any]],
        include_early_snapshots: bool = False,
    ) -> tuple[list[float], list[int]]:
        """Extract (predicted_prob, actual_outcome) pairs from prop-level
        graded traces.

        For each prop_outcome row attached to a trace, the prediction is the
        trace's ``over_prob`` (when side='over') or ``under_prob`` (when
        side='under'). The outcome is 1 for ``result='win'``, 0 for
        ``result='loss'``. Pushes and voids (DNP / no-action) are excluded —
        both are no-action results that carry no calibration signal.

        Game traces are skipped silently.

        Args:
            graded_traces: Output of TraceStore.query_traces / get_graded_traces.
                Each prop trace has ``predictions`` and ``_prop_outcomes`` keys.

        Returns:
            (predictions, outcomes) — parallel lists of floats and 0/1 ints.
        """
        predictions: list[float] = []
        outcomes: list[int] = []

        for trace in graded_traces:
            if not include_early_snapshots and _is_early_market_trace(trace):
                continue
            preds = trace.get("predictions") or {}
            prop_outcomes = trace.get("_prop_outcomes")
            if not prop_outcomes:
                continue
            over_p = preds.get("over_prob")
            under_p = preds.get("under_prob")
            for row in prop_outcomes:
                side = (row.get("side") or "").lower()
                result = row.get("result")
                if result in ("push", "void"):
                    continue
                if side == "over" and over_p is not None:
                    prob = float(over_p)
                elif side == "under" and under_p is not None:
                    prob = float(under_p)
                else:
                    continue
                # Normalize percentage form
                if prob > 1.0:
                    prob = prob / 100.0
                prob = max(0.0, min(1.0, prob))
                actual = 1 if result == "win" else 0
                predictions.append(prob)
                outcomes.append(actual)

        return predictions, outcomes

    @staticmethod
    def extract_pairs_by_context(
        graded_traces: list[dict[str, Any]],
        context_fn: Callable[[dict[str, Any]], str | None],
        extractor: str = "game",
    ) -> dict[str | None, tuple[list[float], list[int]]]:
        """Partition (prediction, outcome) pairs by context slice.

        Args:
            graded_traces: Graded traces from TraceStore.
            context_fn: Maps a single trace dict to a slice label (str) or
                None (base slice). Example::

                    lambda t: "playoff" if t.get("context_labels", {}).get("is_playoff") else "regular"

            extractor: "game" (default) or "prop" — which extraction method
                to apply within each partition.

        Returns:
            Dict keyed by slice label (or None for base). Each value is the
            (predictions, outcomes) tuple for that slice. Slices with fewer
            than _MIN_SAMPLES pairs are still returned; callers are responsible
            for the minimum-sample gate before fitting.
        """
        # Group traces by slice first, then extract (prediction, outcome) pairs per group.
        grouped: dict[str | None, list[dict[str, Any]]] = {}
        for trace in graded_traces:
            label = context_fn(trace)
            grouped.setdefault(label, []).append(trace)

        fitter = CalibrationFitter()
        result: dict[str | None, tuple[list[float], list[int]]] = {}
        for label, traces in grouped.items():
            # Within a partition, allow early-market traces — the partitioning is
            # already the opt-in mechanism that quarantines them into their slice.
            if extractor == "prop":
                preds, outcomes = fitter.extract_prop_pairs(traces, include_early_snapshots=True)
            else:
                preds, outcomes = fitter.extract_pairs(traces, include_early_snapshots=True)
            result[label] = (preds, outcomes)
        return result

    @staticmethod
    def extract_pairs_with_early_slice(
        graded_traces: list[dict[str, Any]],
        extractor: str = "game",
    ) -> dict[str | None, tuple[list[float], list[int]]]:
        """Opt-in partition that quarantines early-market traces in their slice.

        Returns a dict whose ``None`` (base) slice contains only closing-line
        grounded traces and whose ``EARLY_MARKET_SLICE`` key contains the early
        low-liquidity captures. This is the ``include_early_snapshots=True`` path:
        early traces still get a calibration candidate, but in a dedicated slice
        so a promoted production profile never inherits phantom early edges.
        """
        return CalibrationFitter.extract_pairs_by_context(
            graded_traces,
            lambda t: EARLY_MARKET_SLICE if _is_early_market_trace(t) else None,
            extractor=extractor,
        )

    # ------------------------------------------------------------------
    # Isotonic fitting (PAV algorithm)
    # ------------------------------------------------------------------

    def fit_isotonic(
        self,
        predictions: list[float],
        outcomes: list[int],
        league: str,
        n_bins: int = 10,
        market: str = "game",
    ) -> CalibrationProfile:
        """Fit an isotonic calibration profile using Pool-Adjacent-Violators.

        Bins predictions into ``n_bins`` equal-width buckets, computes observed
        win rates per bin, then enforces monotonicity via PAV.

        Args:
            predictions: Model probabilities (0-1).
            outcomes: Binary outcomes (0 or 1).
            league: League code.
            n_bins: Number of calibration bins (default 10).

        Returns:
            CalibrationProfile with method="isotonic".

        Raises:
            ValueError: If fewer than _MIN_SAMPLES pairs provided.
        """
        if len(predictions) < _MIN_SAMPLES:
            raise ValueError(
                f"Need at least {_MIN_SAMPLES} samples for fitting, got {len(predictions)}"
            )

        # Bin predictions
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        bin_counts: list[int] = [0] * n_bins
        bin_sums: list[float] = [0.0] * n_bins
        bin_pred_sums: list[float] = [0.0] * n_bins

        for pred, out in zip(predictions, outcomes):
            idx = min(int(pred * n_bins), n_bins - 1)
            bin_counts[idx] += 1
            bin_sums[idx] += out
            bin_pred_sums[idx] += pred

        # Observed rate per bin (handle empty bins via interpolation)
        observed: list[float | None] = []
        for i in range(n_bins):
            if bin_counts[i] > 0:
                observed.append(bin_sums[i] / bin_counts[i])
            else:
                observed.append(None)

        # Fill empty bins via linear interpolation from neighbors
        observed_filled = _interpolate_empty_bins(observed)

        # Apply PAV for monotonicity
        pav_values, pav_counts = _pool_adjacent_violators(observed_filled, bin_counts)

        # Build calibration map: bin_center → calibrated_prob
        # Clip to [_ISO_CLIP, 1 - _ISO_CLIP] to prevent tail bins with sparse
        # training samples from clamping to exactly 0.0 or 1.0, which produces
        # large ECE gaps on the holdout.
        calibration_map: dict[str, float] = {}
        for i in range(n_bins):
            center = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            clipped = max(_ISO_CLIP, min(1.0 - _ISO_CLIP, pav_values[i]))
            calibration_map[str(round(center, 4))] = round(clipped, 6)

        dataset_hash = _compute_hash(predictions, outcomes)

        market_tag = "" if market == "game" else f"{market}_"
        return CalibrationProfile(
            profile_id=f"iso_{league.lower()}_{market_tag}v1",
            version=1,
            method="isotonic",
            league=league.upper(),
            market=market,
            params={"calibration_map": {float(k): v for k, v in calibration_map.items()}},
            training_window=_infer_window(predictions),
            sample_size=len(predictions),
            dataset_hash=dataset_hash,
            metrics={},  # Filled by evaluate() on held-out set
        )

    # ------------------------------------------------------------------
    # Shrinkage fitting (grid search)
    # ------------------------------------------------------------------

    def fit_shrinkage(
        self,
        predictions: list[float],
        outcomes: list[int],
        league: str,
        eligible_sample_size: int,
        market: str = "game",
    ) -> CalibrationProfile:
        """Fit a shrinkage/sharpening calibration profile by minimizing Brier score.

        Grid searches the calibration slope (conceptually calibration_slope) in
        [0.3, 0.4, ..., 2.0] using np.linspace.
        Values < 1.0 represent shrinkage (softening overconfident predictions).
        Values > 1.0 represent sharpening (strengthening underconfident predictions).

        Sharpening (slope > 1.0) is only permitted when:
        1. eligible_sample_size is at least MIN_SHARPEN_SAMPLE (50).
        2. out-of-sample validation shows a strictly better Brier score than the 1.0 baseline.

        All calibrated probabilities are strictly clamped to [1e-4, 1 - 1e-4].
        """
        if len(predictions) < _MIN_SAMPLES:
            raise ValueError(
                f"Need at least {_MIN_SAMPLES} samples for fitting, got {len(predictions)}"
            )

        MIN_SHARPEN_SAMPLE = 50
        CLV_EPS = 1e-4

        # Determine if sharpening is structurally allowed by sample size
        sharpening_allowed = eligible_sample_size >= MIN_SHARPEN_SAMPLE

        # Perform a deterministic 70/30 train/validation split to guard sharpening
        split_idx = int(0.7 * len(predictions))
        train_preds = predictions[:split_idx]
        train_outs = outcomes[:split_idx]
        val_preds = predictions[split_idx:]
        val_outs = outcomes[split_idx:]

        # Baseline factor=1.0 Brier score on validation set
        def _get_val_brier(f: float) -> float:
            tot = 0.0
            for pred, out in zip(val_preds, val_outs):
                cal = 0.5 + f * (pred - 0.5)
                cal = np.clip(cal, CLV_EPS, 1.0 - CLV_EPS)
                tot += (cal - out) ** 2
            return tot / len(val_preds) if val_preds else 0.0

        baseline_val_brier = _get_val_brier(1.0)

        best_factor = 1.0
        best_brier = float("inf")

        # Grid search: use np.linspace to build the search space
        if sharpening_allowed:
            # 0.3 to 2.0 (18 points: steps of 0.1)
            search_space = np.linspace(0.3, 2.0, 18)
        else:
            # truncate at 1.0 (8 points: 0.3 to 1.0)
            search_space = np.linspace(0.3, 1.0, 8)

        # Grid search is performed on the training set (or full set if sharpening is not allowed)
        fit_preds = train_preds if sharpening_allowed else predictions
        fit_outs = train_outs if sharpening_allowed else outcomes

        for factor in search_space:
            brier = 0.0
            for pred, out in zip(fit_preds, fit_outs):
                cal = 0.5 + factor * (pred - 0.5)
                cal = np.clip(cal, CLV_EPS, 1.0 - CLV_EPS)
                brier += (cal - out) ** 2
            brier /= len(fit_preds)

            if brier < best_brier:
                best_brier = brier
                best_factor = float(factor)

        # Gated validation check for sharpening:
        # If best_factor is > 1.0, we must verify it shows validation improvement over 1.0.
        if best_factor > 1.0:
            cand_val_brier = _get_val_brier(best_factor)
            # Must strictly improve Brier score (lower is better)
            if cand_val_brier >= baseline_val_brier:
                logger.info(
                    "Sharpening factor %f did not improve validation Brier score "
                    "(%f >= %f); capping at 1.0.",
                    best_factor,
                    cand_val_brier,
                    baseline_val_brier,
                )
                best_factor = 1.0

        dataset_hash = _compute_hash(predictions, outcomes)

        market_tag = "" if market == "game" else f"{market}_"
        return CalibrationProfile(
            profile_id=f"shrink_{league.lower()}_{market_tag}v1",
            version=1,
            method="shrinkage",
            league=league.upper(),
            market=market,
            params={
                "shrink_factor": best_factor
            },  # Retained shrink_factor for DB schema compatibility
            training_window=_infer_window(predictions),
            sample_size=len(predictions),
            dataset_hash=dataset_hash,
            metrics={},
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        profile: CalibrationProfile,
        predictions: list[float],
        outcomes: list[int],
    ) -> dict[str, float]:
        """Evaluate a profile on held-out data.

        Applies the profile's calibration to each prediction, then computes
        Brier score, Expected Calibration Error (ECE), and log loss.

        Returns:
            Dict with keys: brier_score, calibration_error, log_loss, n_eval.
        """
        from omega.core.calibration.probability import calibrate_probability

        if not predictions:
            return {"brier_score": 0.0, "calibration_error": 0.0, "log_loss": 0.0, "n_eval": 0}

        calibrated: list[float] = []
        for pred in predictions:
            result = calibrate_probability(pred, method=profile.method, **profile.params)
            calibrated.append(result["calibrated"])

        n = len(predictions)

        # Brier score
        brier = sum((cal - out) ** 2 for cal, out in zip(calibrated, outcomes)) / n

        # ECE via equal-frequency (quantile) bins whose count scales with sample
        # size. A fixed 10 equal-width bins on a ~20-point holdout left most bins
        # with 0-2 samples, so ECE was dominated by 1-2 singleton bins — the
        # variance that made well-calibrated profiles fail the promotion gate on
        # holdout noise. Equal-frequency binning keeps each bin populated.
        ece = _adaptive_calibration_error(calibrated, outcomes)

        # Log loss
        log_loss = 0.0
        for cal, out in zip(calibrated, outcomes):
            p = max(_LOG_EPS, min(1.0 - _LOG_EPS, cal))
            log_loss += -(out * math.log(p) + (1 - out) * math.log(1 - p))
        log_loss /= n

        return {
            "brier_score": round(brier, 6),
            "calibration_error": round(ece, 6),
            "log_loss": round(log_loss, 6),
            "n_eval": n,
        }

    # ------------------------------------------------------------------
    # Cross-validated evaluation (robust ECE for the promotion gate)
    # ------------------------------------------------------------------

    def cross_validated_ece(
        self,
        predictions: list[float],
        outcomes: list[int],
        league: str,
        market: str,
        method: str,
        *,
        folds: int = 5,
        repeats: int = 5,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Repeated stratified k-fold cross-validated calibration error.

        A single 80/20 holdout ECE is high-variance and upward-biased at small n
        (sparse-bin bias), so whether a candidate clears the absolute ECE floor is
        partly luck-of-the-split. This estimates the *method's* generalization ECE
        by fitting on each train fold and scoring the held-out fold — every sample
        is scored out-of-sample exactly ``repeats`` times — and reuses the exact
        ``fit_*``/``evaluate`` primitives (so the same adaptive-ECE estimator the
        floor checks against is used end to end).

        Returns aggregates: ``cv_calibration_error`` (mean), ``cv_ece_ci_low`` /
        ``cv_ece_ci_high`` (95% normal CI), ``cv_brier_score`` (mean), and
        ``cv_n_folds`` (total folds scored). Empty/degenerate inputs return a
        ``cv_n_folds`` of 0 so callers can fall back to the single-split metric.
        """
        n = len(predictions)
        if n < _MIN_SAMPLES * 2 or len(set(outcomes)) < 2:
            return {"cv_calibration_error": 0.0, "cv_brier_score": 0.0, "cv_n_folds": 0}

        if seed is None:
            seed = _seed_from_pairs(predictions, outcomes, league)

        fold_eces: list[float] = []
        fold_briers: list[float] = []
        for r in range(repeats):
            assignment = stratified_folds(outcomes, folds, seed + r)
            for f in range(folds):
                test_idx = set(assignment[f])
                tr_p = [predictions[i] for i in range(n) if i not in test_idx]
                tr_o = [outcomes[i] for i in range(n) if i not in test_idx]
                te_p = [predictions[i] for i in test_idx]
                te_o = [outcomes[i] for i in test_idx]
                if len(tr_p) < _MIN_SAMPLES or not te_p:
                    continue
                if method == "isotonic":
                    profile = self.fit_isotonic(tr_p, tr_o, league=league, market=market)
                elif method == "shrinkage":
                    profile = self.fit_shrinkage(
                        tr_p, tr_o, league=league, market=market, eligible_sample_size=len(tr_p)
                    )
                else:
                    raise ValueError(f"Unknown method: {method!r}")
                m = self.evaluate(profile, te_p, te_o)
                fold_eces.append(m["calibration_error"])
                fold_briers.append(m["brier_score"])

        k = len(fold_eces)
        if k == 0:
            return {"cv_calibration_error": 0.0, "cv_brier_score": 0.0, "cv_n_folds": 0}
        mean = sum(fold_eces) / k
        var = sum((e - mean) ** 2 for e in fold_eces) / (k - 1) if k > 1 else 0.0
        se = math.sqrt(var) / math.sqrt(k)
        return {
            "cv_calibration_error": round(mean, 6),
            "cv_ece_ci_low": round(mean - 1.96 * se, 6),
            "cv_ece_ci_high": round(mean + 1.96 * se, 6),
            "cv_brier_score": round(sum(fold_briers) / k, 6),
            "cv_n_folds": k,
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        candidate: CalibrationProfile,
        incumbent: CalibrationProfile,
        predictions: list[float],
        outcomes: list[int],
    ) -> dict[str, Any]:
        """Compare a candidate profile against the incumbent on held-out data.

        Returns:
            Dict with candidate_metrics, incumbent_metrics, improvement dict,
            and recommend_promote bool.
        """
        cand_metrics = self.evaluate(candidate, predictions, outcomes)
        inc_metrics = self.evaluate(incumbent, predictions, outcomes)

        brier_improvement = inc_metrics["brier_score"] - cand_metrics["brier_score"]
        ece_degradation = cand_metrics["calibration_error"] - inc_metrics["calibration_error"]

        # Recommend promotion if Brier improves AND ECE doesn't degrade by > 0.02
        recommend = brier_improvement > 0 and ece_degradation <= 0.02

        return {
            "candidate_metrics": cand_metrics,
            "incumbent_metrics": inc_metrics,
            "improvement": {
                "brier_improvement": round(brier_improvement, 6),
                "ece_degradation": round(ece_degradation, 6),
            },
            "recommend_promote": recommend,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _seed_from_pairs(predictions: list[float], outcomes: list[int], league: str) -> int:
    """Deterministic CV seed from the data + league (repo determinism norm)."""
    raw = (str(sorted(zip([round(p, 6) for p in predictions], outcomes))) + league).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def stratified_folds(outcomes: list[int], k: int, seed: int) -> list[list[int]]:
    """Partition indices into ``k`` folds preserving class balance.

    Dependency-free stratified k-fold (the fitter avoids sklearn): split indices
    by class, deterministically shuffle each class, then deal round-robin across
    folds so every fold sees ~the same base rate. This matters for the draw plane,
    where positives are only ~20%.
    """
    if k <= 0:
        raise ValueError(f"Number of folds k must be a positive integer, got {k}")

    import random

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, o in enumerate(outcomes):
        by_class.setdefault(int(o), []).append(i)

    folds: list[list[int]] = [[] for _ in range(k)]
    for _cls, idxs in sorted(by_class.items()):
        rng.shuffle(idxs)
        for pos, idx in enumerate(idxs):
            folds[pos % k].append(idx)
    return folds


def _pool_adjacent_violators(
    values: list[float],
    counts: list[int],
) -> tuple[list[float], list[int]]:
    """Pool-Adjacent-Violators algorithm for isotonic regression.

    Enforces monotonically non-decreasing values by merging adjacent
    violating bins (weighted by count).

    Returns:
        (adjusted_values, adjusted_counts) — same length as input.
    """
    result_vals = list(values)
    result_counts = list(counts)

    # Iteratively merge violations
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result_vals) - 1:
            if result_vals[i] > result_vals[i + 1]:
                # Merge bins i and i+1
                total_count = result_counts[i] + result_counts[i + 1]
                if total_count > 0:
                    merged = (
                        result_vals[i] * result_counts[i]
                        + result_vals[i + 1] * result_counts[i + 1]
                    ) / total_count
                else:
                    merged = (result_vals[i] + result_vals[i + 1]) / 2.0
                result_vals[i] = merged
                result_vals[i + 1] = merged
                result_counts[i] = total_count
                result_counts[i + 1] = total_count
                changed = True
            i += 1

    # Expand back to original length if needed
    # (our implementation keeps the same length, just equalizes merged pairs)
    # Re-scan to ensure full monotonicity after all merges
    for i in range(len(result_vals) - 1):
        if result_vals[i] > result_vals[i + 1]:
            avg = (result_vals[i] + result_vals[i + 1]) / 2.0
            result_vals[i] = avg
            result_vals[i + 1] = avg

    return result_vals, result_counts


def _interpolate_empty_bins(observed: list[float | None]) -> list[float]:
    """Fill None entries via linear interpolation from nearest non-None neighbors."""
    n = len(observed)
    result = [0.0] * n

    # Copy known values
    for i in range(n):
        value = observed[i]
        if value is not None:
            result[i] = value

    # Forward-fill then backward-fill for edges
    last_known = None
    for i in range(n):
        if observed[i] is not None:
            last_known = observed[i]
        elif last_known is not None:
            result[i] = last_known

    last_known = None
    for i in range(n - 1, -1, -1):
        if observed[i] is not None:
            last_known = observed[i]
        elif last_known is not None and observed[i] is None:
            result[i] = last_known

    # Linear interpolation between known points
    known_indices = [i for i in range(n) if observed[i] is not None]
    if len(known_indices) >= 2:
        for k in range(len(known_indices) - 1):
            left = known_indices[k]
            right = known_indices[k + 1]
            if right - left > 1:
                left_val = observed[left]
                right_val = observed[right]
                if left_val is None or right_val is None:
                    continue
                for j in range(left + 1, right):
                    t = (j - left) / (right - left)
                    result[j] = left_val + t * (right_val - left_val)

    return result


#: ECE binning policy. Bin count is ``clamp(n // ECE_MIN_PER_BIN, 1, ECE_MAX_BINS)``
#: so each quantile bin holds ~ECE_MIN_PER_BIN points where possible. On a ~20-point
#: holdout this collapses to ~2 well-populated bins instead of ten sparse equal-width
#: ones; as the holdout grows with sample size it approaches full ECE_MAX_BINS
#: resolution.
ECE_MIN_PER_BIN = 10
ECE_MAX_BINS = 10


def _adaptive_calibration_error(
    calibrated: list[float],
    outcomes: list[int],
    *,
    min_per_bin: int = ECE_MIN_PER_BIN,
    max_bins: int = ECE_MAX_BINS,
) -> float:
    """Expected Calibration Error with equal-frequency (quantile) bins.

    Standard ECE uses fixed equal-width bins; on a small holdout most bins hold
    0-2 points, so the estimate is dominated by a couple of singleton bins and is
    high-variance — which made well-calibrated profiles fail the promotion gate on
    holdout noise. This estimator instead groups by predicted probability into
    ``n_bins`` quantile bins, with ``n_bins`` scaled to the sample size, so every
    populated bin's observed rate is estimated from a comparable (non-trivial)
    count. It still detects genuine miscalibration — a systematically over/under-
    confident profile shows a large per-bin gap — without manufacturing error from
    sparse-bin noise.

    Tied predictions are kept in the same bin (isotonic calibration maps produce
    many tied values; splitting a tie and sorting by outcome would invent
    miscalibration). The result is therefore deterministic and independent of
    input order. (A fuller fix is cross-validated ECE over all samples; this keeps
    the existing single-split fit flow.)
    """
    n = len(calibrated)
    if n == 0:
        return 0.0

    # Aggregate by unique predicted value: [count, pred_sum, outcome_sum].
    groups: dict[float, list[float]] = {}
    for pred, out in zip(calibrated, outcomes):
        g = groups.setdefault(pred, [0.0, 0.0, 0.0])
        g[0] += 1.0
        g[1] += pred
        g[2] += out

    n_bins = max(1, min(max_bins, n // min_per_bin))
    bins: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n_bins)]

    # Assign each whole value-group to the quantile bin its cumulative midpoint
    # falls in (never splits a tie). Order-independent (sorted by value).
    cum = 0.0
    for value in sorted(groups):
        count, pred_sum, outcome_sum = groups[value]
        midpoint = cum + count / 2.0
        bin_idx = min(int(midpoint / n * n_bins), n_bins - 1)
        b = bins[bin_idx]
        b[0] += count
        b[1] += pred_sum
        b[2] += outcome_sum
        cum += count

    ece = 0.0
    for count, pred_sum, outcome_sum in bins:
        if count > 0:
            ece += abs(pred_sum / count - outcome_sum / count) * count / n
    return ece


def _compute_hash(predictions: list[float], outcomes: list[int]) -> str:
    """Deterministic hash of prediction-outcome pairs for reproducibility."""
    # Round predictions to avoid floating-point noise
    data = sorted(zip([round(p, 6) for p in predictions], outcomes))
    raw = str(data).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _infer_window(predictions: list[float]) -> str:
    """Placeholder training window — actual dates would come from traces."""
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"fitted_{now}"
