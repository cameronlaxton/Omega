"""Per-source odds timing classification (calibration-backfill safety).

Gates **betting only** — selection, staking, ROI, CLV. It never gates probability
calibration, which does not read odds. Conservative default: any source not in the
registry is ``TIMING_UNKNOWN`` (excluded from ROI/CLV; may not drive selection).
"""

from __future__ import annotations

from enum import Enum


class OddsTimingClass(str, Enum):
    DECISION_TIME_SAFE = "decision_time_safe"
    CLOSING_ONLY = "closing_only"
    TIMING_UNKNOWN = "timing_unknown"


# Source (adapter source_name / ingest --source) → declared timing class.
_SOURCE_TIMING: dict[str, OddsTimingClass] = {
    "football_data": OddsTimingClass.DECISION_TIME_SAFE,
    "the_odds_api": OddsTimingClass.DECISION_TIME_SAFE,
}


def coerce(value: object) -> OddsTimingClass:
    if isinstance(value, OddsTimingClass):
        return value
    try:
        return OddsTimingClass(str(value))
    except ValueError:
        return OddsTimingClass.TIMING_UNKNOWN


def timing_class_for_source(source_name: str | None) -> OddsTimingClass:
    return _SOURCE_TIMING.get((source_name or "").lower(), OddsTimingClass.TIMING_UNKNOWN)


def is_selection_safe(timing_class: object) -> bool:
    """Only ``decision_time_safe`` odds may drive selection/staking."""
    return coerce(timing_class) is OddsTimingClass.DECISION_TIME_SAFE


def allows_roi(timing_class: object) -> bool:
    """``timing_unknown`` selections are excluded from ROI gates."""
    return coerce(timing_class) is OddsTimingClass.DECISION_TIME_SAFE


def allows_clv(timing_class: object) -> bool:
    """``decision_time_safe`` and ``closing_only`` contribute to CLV; unknown cannot."""
    return coerce(timing_class) in (
        OddsTimingClass.DECISION_TIME_SAFE,
        OddsTimingClass.CLOSING_ONLY,
    )
