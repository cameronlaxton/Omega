"""
Calibration package — probability calibration, profiles, fitting, and registry.

Public API:
    apply_calibration(raw_prob, league=None)           — shared calibration policy
    apply_calibration_audited(raw_prob, league=None)   — same, plus audit dict
    CalibrationProfile / ProfileStatus                 — versioned profile model
    CalibrationRegistry                                — JSON-backed profile storage
    CalibrationFitter                                  — fit profiles from graded traces
"""

from omega.core.calibration.fitter import CalibrationFitter
from omega.core.calibration.probability import apply_calibration, apply_calibration_audited
from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus
from omega.core.calibration.registry import CalibrationRegistry

__all__ = [
    "apply_calibration",
    "apply_calibration_audited",
    "CalibrationProfile",
    "ProfileStatus",
    "CalibrationRegistry",
    "CalibrationFitter",
]
