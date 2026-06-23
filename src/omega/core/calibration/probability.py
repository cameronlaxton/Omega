"""
Probability Calibration Module

Calibrates model probabilities to fix unrealistic extremes (>90% or <10% too frequently).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger("omega.core.calibration.probability")

# ---------------------------------------------------------------------------
# Active-registry override (single, shared selection path stays intact)
# ---------------------------------------------------------------------------
# When set, production-profile lookups read this isolated registry file instead
# of the default ``profiles.json``. Default (None) preserves the exact production
# lookup, so this is behavior-preserving for the live path. It exists so the
# historical lab can replay a window with a SPECIFIC candidate/incumbent profile
# active — the only honest way to compute a candidate-vs-incumbent betting delta
# (the engine applies calibration at selection time) — without touching or
# polluting the production registry. ContextVar → thread/async-safe.
_REGISTRY_PATH_OVERRIDE: ContextVar[str | None] = ContextVar(
    "omega_calibration_registry_path", default=None
)


@contextmanager
def calibration_registry_override(path: str | None):
    """Temporarily point active-profile lookups at an isolated registry file.

    ``path=None`` is a no-op (restores the default production registry). Nesting
    is supported; the previous value is restored on exit.
    """
    token = _REGISTRY_PATH_OVERRIDE.set(path)
    try:
        yield
    finally:
        _REGISTRY_PATH_OVERRIDE.reset(token)


def shrinkage_calibration(raw_prob: float, shrink_factor: float = 0.7) -> float:
    """
    Calibrates probability by shrinking toward 0.5 (conceptually a calibration_slope).

    Formula: p_calibrated = 0.5 + calibration_slope * (p_raw - 0.5)

    This reduces extreme probabilities while preserving relative ordering, and also
    supports sharpening (factor > 1.0) under sample size gates.

    Args:
        raw_prob: Raw model probability (0.0 to 1.0)
        shrink_factor: Calibration slope (conceptually calibration_slope, 0.3 to 2.0)
                      - 1.0 = no shrinkage (returns raw_prob)
                      - < 1.0 = softening (reduces extreme probabilities)
                      - > 1.0 = sharpening (amplifies probabilities)

    Returns:
        Calibrated probability clamped strictly to [1e-4, 1 - 1e-4]
    """
    raw_prob = max(0.0, min(1.0, raw_prob))
    calibrated = 0.5 + shrink_factor * (raw_prob - 0.5)

    if np is not None:
        return float(np.clip(calibrated, 1e-4, 1.0 - 1e-4))
    else:
        return max(1e-4, min(1.0 - 1e-4, calibrated))


def cap_calibration(raw_prob: float, cap_max: float = 0.85, cap_min: float = 0.15) -> float:
    """
    Calibrates probability by capping extremes.

    This prevents probabilities from exceeding reasonable bounds.

    Args:
        raw_prob: Raw model probability (0.0 to 1.0)
        cap_max: Maximum allowed probability (default 0.85)
        cap_min: Minimum allowed probability (default 0.15)

    Returns:
        Calibrated probability (capped between cap_min and cap_max)
    """
    raw_prob = max(0.0, min(1.0, raw_prob))
    if raw_prob > cap_max:
        return cap_max
    elif raw_prob < cap_min:
        return cap_min
    else:
        return raw_prob


def isotonic_calibration(raw_prob: float, calibration_map: dict[float, float]) -> float:
    """
    Calibrates probability using isotonic regression mapping.

    This uses a lookup table/mapping derived from historical performance.
    The calibration_map should map raw probability bins to calibrated probabilities.

    Args:
        raw_prob: Raw model probability (0.0 to 1.0)
        calibration_map: Dict mapping raw_prob bins to calibrated probs
                        Example: {0.0: 0.15, 0.2: 0.18, 0.5: 0.50, 0.8: 0.75, 1.0: 0.85}

    Returns:
        Calibrated probability (interpolated from calibration_map)
    """
    raw_prob = max(0.0, min(1.0, raw_prob))

    if not calibration_map:
        return raw_prob

    sorted_keys = sorted(calibration_map.keys())

    if raw_prob <= sorted_keys[0]:
        return calibration_map[sorted_keys[0]]
    if raw_prob >= sorted_keys[-1]:
        return calibration_map[sorted_keys[-1]]

    for i in range(len(sorted_keys) - 1):
        if sorted_keys[i] <= raw_prob <= sorted_keys[i + 1]:
            lower_key = sorted_keys[i]
            upper_key = sorted_keys[i + 1]
            lower_val = calibration_map[lower_key]
            upper_val = calibration_map[upper_key]

            t = (raw_prob - lower_key) / (upper_key - lower_key)
            return lower_val + t * (upper_val - lower_val)

    return raw_prob


def calibrate_probability(
    raw_prob: float,
    method: str = "shrinkage",
    shrink_factor: float = 0.7,
    cap_max: float = 0.85,
    cap_min: float = 0.15,
    calibration_map: dict[float, float] | None = None,
) -> dict[str, Any]:
    """
    Main calibration function that applies specified calibration method.

    Args:
        raw_prob: Raw model probability (0.0 to 1.0)
        method: Calibration method ("shrinkage", "cap", "isotonic", "combined")
        shrink_factor: Shrinkage factor for shrinkage method (default 0.7)
        cap_max: Maximum cap for cap method (default 0.85)
        cap_min: Minimum cap for cap method (default 0.15)
        calibration_map: Calibration map for isotonic method

    Returns:
        Dict with keys:
            - "calibrated": float (calibrated probability)
            - "raw": float (original raw probability, for debugging)
            - "method": str (method used)
    """
    raw_prob = max(0.0, min(1.0, raw_prob))

    if method == "shrinkage":
        calibrated = shrinkage_calibration(raw_prob, shrink_factor)
    elif method == "cap":
        calibrated = cap_calibration(raw_prob, cap_max, cap_min)
    elif method == "isotonic":
        # Ensure calibration_map keys are floats (JSON round-trip makes them strings)
        cmap = calibration_map or {}
        if cmap and isinstance(next(iter(cmap)), str):
            cmap = {float(k): v for k, v in cmap.items()}
        calibrated = isotonic_calibration(raw_prob, cmap)
    elif method == "combined":
        shrunk = shrinkage_calibration(raw_prob, shrink_factor)
        calibrated = cap_calibration(shrunk, cap_max, cap_min)
    else:
        calibrated = raw_prob
        method = "none"

    return {"calibrated": calibrated, "raw": raw_prob, "method": method}


def should_apply_calibration(raw_prob: float, strict_cap: bool = False) -> bool:
    """
    Determines if calibration should be applied based on raw probability.

    Args:
        raw_prob: Raw model probability
        strict_cap: If True, always apply calibration if outside [0.15, 0.85]

    Returns:
        True if calibration should be applied
    """
    if strict_cap:
        return raw_prob > 0.85 or raw_prob < 0.15
    else:
        return raw_prob > 0.90 or raw_prob < 0.10


# ---------------------------------------------------------------------------
# Shared calibration policy — single source of truth
# ---------------------------------------------------------------------------
# INVARIANT: Both production (service.py) and backtest (engine.py) MUST call
# apply_calibration(). Do not duplicate these parameters in other call sites.
# Phase 6 will replace this with profile-driven selection; until then this
# function is the canonical policy.
# ---------------------------------------------------------------------------

_POLICY_METHOD = "combined"
_POLICY_SHRINK_FACTOR = 0.7
_POLICY_CAP_MAX = 0.90
_POLICY_CAP_MIN = 0.10


def apply_calibration(
    raw_prob: float,
    league: str | None = None,
    context_hints: dict[str, Any] | None = None,
    market: str = "game",
) -> float:
    """Apply the canonical calibration policy. Used by both service and backtest.

    When a league is provided, attempts to use a learned production profile.
    When context_hints is also provided, a context_slice is derived from the
    hints and used to look up a slice-specific profile (e.g. 'playoff'). If
    no slice-specific profile exists, falls back to the base league profile,
    then to the static policy.

    Args:
        raw_prob: Raw model probability (0.0 to 1.0)
        league: Optional league code (e.g. "NBA").
        context_hints: Optional dict with context signals used to derive a
            context_slice. Recognised keys: is_playoff (bool),
            rest_days (int; 0=B2B).
        market: Calibration market plane — "game" (default), "prop" for
            player-prop over/under probabilities, or "draw" for 3-way draw
            probabilities. A "prop"/"draw" lookup falls back to the league's
            "game" profile when no market-specific profile is registered, then
            to the static policy.

    Returns:
        Calibrated probability as a float.

    Delegates to :func:`apply_calibration_audited` so both the value-only and
    audited paths share ONE selection + damping implementation (the drift rule):
    hierarchical fallback (league → sport_family → global → static) and maturity
    damping apply identically here.
    """
    calibrated, _audit = apply_calibration_audited(
        raw_prob, league=league, context_hints=context_hints, market=market
    )
    return calibrated


def apply_calibration_audited(
    raw_prob: float,
    league: str | None = None,
    context_hints: dict[str, Any] | None = None,
    market: str = "game",
) -> tuple[float, dict[str, Any]]:
    """Like apply_calibration() but also returns an audit dict.

    The audit dict documents exactly which calibration path was taken:
      path: "profile" | "base_profile_fallback" | "static_calibrated" | "static_identity"
      profile_id: str | None
      context_slice: str | None  (the slice that was requested)
      resolved_slice: str | None (the profile's actual slice; None if base fallback)
      method_resolved: str | None
      raw_prob: float
      calibrated_prob: float
    """
    raw = max(0.0, min(1.0, raw_prob))
    context_slice = _derive_context_slice(context_hints, league) if context_hints else None

    if league is not None:
        profile, fallback_level = _get_applicable_profile(
            league, context_slice=context_slice, market=market
        )
        if profile is not None:
            maturity = profile.effective_maturity().value
            result = calibrate_probability(raw, method=profile.method, **profile.params)
            calibrated = result["calibrated"]
            # Low-trust (provisional/probation) profiles may only make a small,
            # bounded correction; none/retired collapse back to raw.
            calibrated, maturity_damped = _damp_for_maturity(raw, calibrated, maturity)
            resolved_slice = getattr(profile, "context_slice", None)
            path = (
                "base_profile_fallback"
                if context_slice is not None and resolved_slice is None
                else "profile"
            )
            return calibrated, {
                "path": path,
                "profile_id": profile.profile_id,
                "context_slice": context_slice,
                "resolved_slice": resolved_slice,
                "method_resolved": profile.method,
                "raw_prob": raw,
                "calibrated_prob": calibrated,
                "maturity": maturity,
                "profile_status": profile.status.value,
                "sample_size": profile.sample_size,
                "ece": profile.ece,
                "brier": profile.brier,
                "fallback_level": fallback_level,
                "maturity_damped": maturity_damped,
            }

    # No applicable profile -> static policy. None for all profile-provenance
    # keys so the audit dict shape is identical to the profile path.
    static_provenance = {
        "profile_id": None,
        "resolved_slice": None,
        "maturity": None,
        "profile_status": None,
        "sample_size": None,
        "ece": None,
        "brier": None,
        "fallback_level": None,
        "maturity_damped": False,
    }

    # Static fallback: if within threshold, return raw unchanged
    if not should_apply_calibration(raw, strict_cap=False):
        return raw, {
            "path": "static_identity",
            "context_slice": context_slice,
            "method_resolved": None,
            "raw_prob": raw,
            "calibrated_prob": raw,
            **static_provenance,
        }

    result = calibrate_probability(
        raw,
        method=_POLICY_METHOD,
        shrink_factor=_POLICY_SHRINK_FACTOR,
        cap_max=_POLICY_CAP_MAX,
        cap_min=_POLICY_CAP_MIN,
    )
    calibrated = result["calibrated"]
    return calibrated, {
        "path": "static_calibrated",
        "context_slice": context_slice,
        "method_resolved": _POLICY_METHOD,
        "raw_prob": raw,
        "calibrated_prob": calibrated,
        **static_provenance,
    }


def _derive_context_slice(
    context_hints: dict[str, Any] | None, league: str | None = None
) -> str | None:
    """Derive a calibration context_slice string from context hints using the canonical resolution."""
    if not context_hints:
        return None

    from omega.core.calibration.context_slices import context_slice_for_trace
    from omega.core.calibration.sport_family import sport_family_for_league

    # Pack hints into a dummy trace structure to reuse the canonical extractor
    dummy_trace = {"context_hints": context_hints}
    sport_family = sport_family_for_league(league) if league else None

    return context_slice_for_trace(dummy_trace, sport_family=sport_family)


def _get_active_profile(league: str, context_slice: str | None = None, market: str = "game"):
    """Look up the production calibration profile for (league, context_slice, market).

    Returns None on any failure (missing file, import error, etc.)
    so that callers always fall back to the static policy gracefully.
    The registry get_production() already handles the slice->base and
    market->game fallbacks.
    """
    try:
        from omega.core.calibration.registry import CalibrationRegistry

        override = _REGISTRY_PATH_OVERRIDE.get()
        registry = CalibrationRegistry(path=override) if override else CalibrationRegistry()
        return registry.get_production(league, context_slice=context_slice, market=market)
    except Exception:
        return None


def _get_applicable_profile(league: str, context_slice: str | None = None, market: str = "game"):
    """Highest-trust applicable profile + fallback level (hierarchical walk).

    Returns ``(profile, level)`` where level is one of league|sport_family|global,
    or ``(None, None)`` when no profile applies (caller uses the static policy).

    Implemented as a walk over :func:`_get_active_profile` (one call per bucket)
    so the production registry-override seam is honored at every level and the
    same lookup is patchable in tests. Mirrors
    ``CalibrationRegistry.get_applicable`` (the registry-native equivalent).
    """
    try:
        profile = _get_active_profile(league, context_slice=context_slice, market=market)
        if profile is not None:
            return profile, "league"

        from omega.core.calibration.registry import GLOBAL_BUCKET
        from omega.core.calibration.sport_family import sport_family_for_league

        family = sport_family_for_league(league)
        if family and family != "unknown":
            profile = _get_active_profile(
                family.upper(), context_slice=context_slice, market=market
            )
            if profile is not None:
                return profile, "sport_family"

        profile = _get_active_profile(GLOBAL_BUCKET, context_slice=context_slice, market=market)
        if profile is not None:
            return profile, "global"
    except Exception:
        return None, None
    return None, None


# Maturity-based correction damping: a low-trust profile may only move the
# probability by a small bounded amount, so a thin fit makes an honest small
# correction instead of a confident large one. None of these constants touch a
# full ``production`` (or legacy) profile.
_MATURITY_MAX_SHIFT: dict[str, float] = {
    "provisional": 0.03,
    "probation": 0.05,
}


def _damp_for_maturity(raw: float, calibrated: float, maturity: str) -> tuple[float, bool]:
    """Cap the absolute probability shift for provisional/probation profiles.

    Returns ``(possibly_damped_value, was_damped)``. NONE/RETIRED maturities are
    not trusted to apply, so they collapse to the raw value (no correction);
    PRODUCTION applies the full correction unchanged.
    """
    if maturity in ("none", "retired"):
        return raw, raw != calibrated
    cap = _MATURITY_MAX_SHIFT.get(maturity)
    if cap is None:
        return calibrated, False  # production: full correction
    shift = calibrated - raw
    if abs(shift) <= cap:
        return calibrated, False
    sign = 1.0 if shift >= 0 else -1.0
    return raw + sign * cap, True
