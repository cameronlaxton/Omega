"""
Probability Calibration Module

Calibrates model probabilities to fix unrealistic extremes (>90% or <10% too frequently).
"""

from __future__ import annotations
from typing import Dict, Optional


def shrinkage_calibration(raw_prob: float, shrink_factor: float = 0.7) -> float:
    """
    Calibrates probability by shrinking toward 0.5.
    
    Formula: p_calibrated = 0.5 + shrink_factor * (p_raw - 0.5)
    
    This reduces extreme probabilities while preserving relative ordering.
    
    Args:
        raw_prob: Raw model probability (0.0 to 1.0)
        shrink_factor: Shrinkage factor (0.0 to 1.0)
                      - 1.0 = no shrinkage (returns raw_prob)
                      - 0.0 = maximum shrinkage (always returns 0.5)
                      - 0.7 = moderate shrinkage (default)
    
    Returns:
        Calibrated probability (0.0 to 1.0)
    """
    raw_prob = max(0.0, min(1.0, raw_prob))
    calibrated = 0.5 + shrink_factor * (raw_prob - 0.5)
    return max(0.0, min(1.0, calibrated))


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


def isotonic_calibration(raw_prob: float, calibration_map: Dict[float, float]) -> float:
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
    calibration_map: Optional[Dict[float, float]] = None
) -> Dict[str, Any]:
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
        calibrated = isotonic_calibration(raw_prob, calibration_map or {})
    elif method == "combined":
        shrunk = shrinkage_calibration(raw_prob, shrink_factor)
        calibrated = cap_calibration(shrunk, cap_max, cap_min)
    else:
        calibrated = raw_prob
        method = "none"
    
    return {
        "calibrated": calibrated,
        "raw": raw_prob,
        "method": method
    }


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
