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
from datetime import datetime, timezone
UTC = timezone.utc
from typing import Any, Callable

from omega.core.calibration.profiles import CalibrationProfile

logger = logging.getLogger("omega.core.calibration.fitter")

# Minimum samples required for fitting
_MIN_SAMPLES = 30

# Epsilon for log loss clipping
_LOG_EPS = 1e-15


class CalibrationFitter:
    """Fit calibration profiles from historical prediction-outcome pairs."""

    # ------------------------------------------------------------------
    # Extract (prediction, outcome) pairs from graded traces
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pairs(
        graded_traces: list[dict[str, Any]],
    ) -> tuple[list[float], list[int]]:
        """Extract (predicted_prob, actual_outcome) pairs from game-level
        graded traces.

        Prop traces are skipped silently — they have neither ``home_win_prob``
        nor ``_outcome``. Use :func:`extract_prop_pairs` for prop calibration.

        Args:
            graded_traces: Output of TraceStore.get_graded_traces().
                Each dict has ``predictions`` and ``_outcome`` keys.

        Returns:
            (predictions, outcomes) — parallel lists of floats and 0/1 ints.
        """
        predictions: list[float] = []
        outcomes: list[int] = []

        for trace in graded_traces:
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
    def extract_prop_pairs(
        graded_traces: list[dict[str, Any]],
    ) -> tuple[list[float], list[int]]:
        """Extract (predicted_prob, actual_outcome) pairs from prop-level
        graded traces.

        For each prop_outcome row attached to a trace, the prediction is the
        trace's ``over_prob`` (when side='over') or ``under_prob`` (when
        side='under'). The outcome is 1 for ``result='win'``, 0 for
        ``result='loss'``. Pushes are excluded (they carry no calibration
        signal).

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
            preds = trace.get("predictions") or {}
            prop_outcomes = trace.get("_prop_outcomes")
            if not prop_outcomes:
                continue
            over_p = preds.get("over_prob")
            under_p = preds.get("under_prob")
            for row in prop_outcomes:
                side = (row.get("side") or "").lower()
                result = row.get("result")
                if result == "push":
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
        partitions: dict[str | None, tuple[list[float], list[int]]] = {}
        for trace in graded_traces:
            label = context_fn(trace)
            if label not in partitions:
                partitions[label] = ([], [])
            partitions[label][0].append(0.0)   # placeholder; replaced below
            partitions[label][1].append(0)

        # Re-partition cleanly: group traces first, then extract pairs per group.
        grouped: dict[str | None, list[dict[str, Any]]] = {}
        for trace in graded_traces:
            label = context_fn(trace)
            grouped.setdefault(label, []).append(trace)

        fitter = CalibrationFitter()
        result: dict[str | None, tuple[list[float], list[int]]] = {}
        for label, traces in grouped.items():
            if extractor == "prop":
                preds, outcomes = fitter.extract_prop_pairs(traces)
            else:
                preds, outcomes = fitter.extract_pairs(traces)
            result[label] = (preds, outcomes)
        return result

    # ------------------------------------------------------------------
    # Isotonic fitting (PAV algorithm)
    # ------------------------------------------------------------------

    def fit_isotonic(
        self,
        predictions: list[float],
        outcomes: list[int],
        league: str,
        n_bins: int = 10,
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
        calibration_map: dict[str, float] = {}
        for i in range(n_bins):
            center = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            # Use string keys for JSON serialization
            calibration_map[str(round(center, 4))] = round(pav_values[i], 6)

        dataset_hash = _compute_hash(predictions, outcomes)

        return CalibrationProfile(
            profile_id=f"iso_{league.lower()}_v1",
            version=1,
            method="isotonic",
            league=league.upper(),
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
    ) -> CalibrationProfile:
        """Fit a shrinkage calibration profile by minimizing Brier score.

        Grid searches shrink_factor in [0.3, 0.4, ..., 1.0].

        Args:
            predictions: Model probabilities (0-1).
            outcomes: Binary outcomes (0 or 1).
            league: League code.

        Returns:
            CalibrationProfile with method="shrinkage".
        """
        if len(predictions) < _MIN_SAMPLES:
            raise ValueError(
                f"Need at least {_MIN_SAMPLES} samples for fitting, got {len(predictions)}"
            )

        best_factor = 1.0
        best_brier = float("inf")

        for factor_int in range(3, 11):  # 0.3 to 1.0 in steps of 0.1
            factor = factor_int / 10.0
            brier = 0.0
            for pred, out in zip(predictions, outcomes):
                cal = 0.5 + factor * (pred - 0.5)
                brier += (cal - out) ** 2
            brier /= len(predictions)
            if brier < best_brier:
                best_brier = brier
                best_factor = factor

        dataset_hash = _compute_hash(predictions, outcomes)

        return CalibrationProfile(
            profile_id=f"shrink_{league.lower()}_v1",
            version=1,
            method="shrinkage",
            league=league.upper(),
            params={"shrink_factor": best_factor},
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

        # ECE (10-bin)
        ece = _expected_calibration_error(calibrated, outcomes, n_bins=10)

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


def _expected_calibration_error(
    calibrated: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    n = len(calibrated)
    if n == 0:
        return 0.0

    bin_counts: list[int] = [0] * n_bins
    bin_pred_sums: list[float] = [0.0] * n_bins
    bin_outcome_sums: list[float] = [0.0] * n_bins

    for pred, out in zip(calibrated, outcomes):
        idx = min(int(pred * n_bins), n_bins - 1)
        bin_counts[idx] += 1
        bin_pred_sums[idx] += pred
        bin_outcome_sums[idx] += out

    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_pred = bin_pred_sums[i] / bin_counts[i]
            avg_outcome = bin_outcome_sums[i] / bin_counts[i]
            ece += abs(avg_pred - avg_outcome) * bin_counts[i] / n

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
