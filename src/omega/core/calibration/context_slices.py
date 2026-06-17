"""
Canonical context slice definitions.

Provides a single source of truth for context slice names, label normalization,
precedence, and trace extraction across the Omega system.
"""

import re
from collections.abc import Mapping
from typing import Any

BASE_CONTEXT_SLICE = None

INITIAL_CONTEXT_SLICES = (
    "early_market_low_liq",
    "playoff",
    "back_to_back",
    "short_week",
    "weather_extreme",
    "neutral_site",
    "surface",
    "best_of_5",
)

LOW_CARDINALITY_CONTEXT_SLICES = (
    "early_market_low_liq",
    "playoff",
    "back_to_back",
    "rest_disadvantage",
    "short_week",
    "weather_extreme",
    "neutral_site",
    "backup_qb",
    "cup_match",
    "congested_fixture",
    "surface",
    "best_of_5",
    "goalie_confirmed",
    "starting_pitcher_change",
)

# Global precedence list. Lower index = higher precedence.
_PRECEDENCE = [
    "early_market_low_liq",
    "playoff",
    "short_week",
    "backup_qb",
    "weather_extreme",
    "neutral_site",
    "back_to_back",
    "rest_disadvantage",
    "congested_fixture",
    "cup_match",
    "surface",
    "best_of_5",
    "goalie_confirmed",
    "starting_pitcher_change",
    "bullpen_taxed",
    "lineup_uncertain",
]


def normalize_context_label(value: Any) -> str | None:
    """Normalize a raw label string into a canonical slice format."""
    if not isinstance(value, str):
        return None
    val = value.lower().strip()
    if not val:
        return None
    # Replace spaces, hyphens, slashes, and dots with underscores
    val = re.sub(r'[\s\-\/\.]+', '_', val)
    # Collapse duplicate underscores
    val = re.sub(r'_+', '_', val)
    val = val.strip('_')
    if not val:
        return None
    return val


def labels_from_trace(trace: Mapping[str, Any]) -> set[str]:
    """Extract and normalize context labels defensively from a trace."""
    labels = set()

    locations = [
        trace.get("context_labels"),
        trace.get("context_slice"),
        trace.get("input", {}).get("context_labels") if isinstance(trace.get("input"), dict) else None,
        trace.get("metadata", {}).get("context_labels") if isinstance(trace.get("metadata"), dict) else None,
        trace.get("run_context", {}).get("context_labels") if isinstance(trace.get("run_context"), dict) else None,
        trace.get("market", {}).get("context_labels") if isinstance(trace.get("market"), dict) else None,
        trace.get("features", {}).get("context_labels") if isinstance(trace.get("features"), dict) else None,
        trace.get("calibration", {}).get("context_hints") if isinstance(trace.get("calibration"), dict) else None,
        trace.get("context_hints"),
        trace.get("tags"),
        trace.get("labels"),
    ]

    def _extract(val: Any) -> None:
        if isinstance(val, str):
            norm = normalize_context_label(val)
            if norm:
                labels.add(norm)
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                _extract(item)
        elif isinstance(val, dict):
            # Pre-map legacy context_hints properties
            if val.get("is_playoff"):
                _extract("playoff")
            rest = val.get("rest_days")
            if rest is not None:
                try:
                    if int(rest) == 0:
                        _extract("back_to_back")
                except (ValueError, TypeError):
                    pass

            for k, v in val.items():
                if isinstance(v, bool):
                    if v:
                        _extract(k)
                elif isinstance(v, str):
                    if normalize_context_label(k) == "surface":
                        _extract("surface_" + v)  # Encode surface_{value}
                        _extract("surface")  # General fallback
                    else:
                        _extract(v)
                else:
                    _extract(v)

    for loc in locations:
        if loc:
            _extract(loc)

    # Some specialized dict structures: e.g. "liquidity_profile" -> "early_market_low_liq"
    if trace.get("liquidity_profile"):
        _extract(trace.get("liquidity_profile"))

    return labels


def _map_aliases(raw_labels: set[str], sport_family: str | None = None) -> set[str]:
    """Map raw normalized labels to their canonical slice names based on sport."""
    mapped = set(raw_labels)

    # Global
    if "b2b" in mapped:
        mapped.add("back_to_back")
    if "postseason" in mapped:
        mapped.add("playoff")

    if sport_family == "basketball":
        if "playoffs" in mapped:
            mapped.add("playoff")
        if "travel_disadvantage" in mapped:
            mapped.add("rest_disadvantage")
        if any(x in mapped for x in ("star_absent", "injury_uncertain")):
            mapped.add("lineup_uncertain")

    elif sport_family == "american_football":
        if "nfl_playoff" in mapped:
            mapped.add("playoff")
        if any(x in mapped for x in ("thursday", "thursday_night")):
            mapped.add("short_week")
        if any(x in mapped for x in ("qb_change", "starting_qb_out")):
            mapped.add("backup_qb")
        if any(x in mapped for x in ("heavy_wind", "snow", "rain_extreme")):
            mapped.add("weather_extreme")
        if any(x in mapped for x in ("london", "international_series")):
            mapped.add("neutral_site")
        if any(x in mapped for x in ("division_game", "divisional")):
            mapped.add("division_game")

    elif sport_family == "soccer":
        if any(x in mapped for x in ("knockout", "tournament", "world_cup", "ucl_knockout")):
            mapped.add("cup_match")
        if "rivalry" in mapped:
            mapped.add("derby")
        if any(x in mapped for x in ("fixture_congestion", "short_rest")):
            mapped.add("congested_fixture")
        if "squad_rotation" in mapped:
            mapped.add("rotation_risk")

    elif sport_family == "tennis":
        # Encode surface types directly if specified
        if any(x in mapped for x in ("clay", "grass", "hard", "indoor_hard")):
            mapped.add("surface")
            for s in ("clay", "grass", "hard", "indoor_hard"):
                if s in mapped:
                    mapped.add(f"surface_{s}")
        if any(x in mapped for x in ("bo5", "grand_slam_men")):
            mapped.add("best_of_5")
        if "serve_dominant" in mapped:
            mapped.add("serve_dominant")
        if "break_point_pressure" in mapped:
            mapped.add("pressure_state")
        if "injury_risk" in mapped:
            mapped.add("injury_retirement_risk")

    elif sport_family == "baseball":
        if any(x in mapped for x in ("pitcher_change", "opener_change")):
            mapped.add("starting_pitcher_change")
        if "bullpen_fatigue" in mapped:
            mapped.add("bullpen_taxed")
        if "wind_out" in mapped:
            mapped.add("weather_wind_out")

    elif sport_family == "hockey":
        if "confirmed_goalie" in mapped:
            mapped.add("goalie_confirmed")
        if "goalie_change" in mapped:
            mapped.add("goalie_uncertain")
        if "three_in_four" in mapped:
            mapped.add("three_in_four")

    # Handle implicit surface if no sport family was provided but surfaces exist
    if sport_family is None and any(x in mapped for x in ("clay", "grass", "hard", "indoor_hard")):
        mapped.add("surface")
        for s in ("clay", "grass", "hard", "indoor_hard"):
            if s in mapped:
                mapped.add(f"surface_{s}")

    return mapped


def context_slice_for_trace(trace: Mapping[str, Any], *, sport_family: str | None = None) -> str | None:
    """Determine the best single context_slice from normalized labels based on precedence."""
    raw_labels = labels_from_trace(trace)
    if not raw_labels:
        return BASE_CONTEXT_SLICE

    mapped_labels = _map_aliases(raw_labels, sport_family)

    # Check data_quality_quarantine labels if any exist
    if "data_quality_quarantine" in mapped_labels:
        return "data_quality_quarantine"

    # Evaluate precedence
    for candidate in _PRECEDENCE:
        if candidate in mapped_labels:
            # Special case for tennis surface subslices (e.g., surface_clay)
            if candidate == "surface":
                # Check if a specific surface subslice is present
                for surface_type in ("surface_clay", "surface_grass", "surface_hard", "surface_indoor_hard"):
                    if surface_type in mapped_labels:
                        return surface_type
            return candidate

    return BASE_CONTEXT_SLICE
