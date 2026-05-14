"""
Calibration package — probability calibration, profiles, fitting, and registry.

Public API:
    apply_calibration(raw_prob, league=None)  — shared calibration policy
    CalibrationProfile / ProfileStatus        — versioned profile model
    CalibrationRegistry                       — JSON-backed profile storage
    CalibrationFitter                         — fit profiles from graded traces
"""

from omega.core.calibration.probability import apply_calibration
from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus
from omega.core.calibration.registry import CalibrationRegistry
from omega.core.calibration.fitter import CalibrationFitter

__all__ = [
    "apply_calibration",
    "CalibrationProfile",
    "ProfileStatus",
    "CalibrationRegistry",
    "CalibrationFitter",
]
