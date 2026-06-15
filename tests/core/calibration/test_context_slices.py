import pytest

from omega.core.calibration.context_slices import (
    BASE_CONTEXT_SLICE,
    context_slice_for_trace,
    labels_from_trace,
    normalize_context_label,
)


def test_normalize_context_label():
    assert normalize_context_label(" Playoff ") == "playoff"
    assert normalize_context_label("early-market low liq.") == "early_market_low_liq"
    assert normalize_context_label("BACK__TO__BACK") == "back_to_back"
    assert normalize_context_label("") is None
    assert normalize_context_label(123) is None


def test_labels_from_trace_empty():
    assert labels_from_trace({}) == set()
    assert labels_from_trace({"unrelated": "field"}) == set()


def test_labels_from_trace_types():
    trace = {
        "context_labels": ["playoff", "short week"],
        "tags": {"weather_extreme": True, "neutral_site": False},
        "context_hints": "b2b",
        "liquidity_profile": "early_market_low_liq",
    }
    labels = labels_from_trace(trace)
    assert "playoff" in labels
    assert "short_week" in labels
    assert "weather_extreme" in labels
    assert "neutral_site" not in labels
    assert "b2b" in labels
    assert "early_market_low_liq" in labels


def test_context_slice_for_trace_precedence():
    # early_market_low_liq wins over everything
    trace = {"context_labels": ["playoff", "early_market_low_liq", "weather_extreme"]}
    assert context_slice_for_trace(trace) == "early_market_low_liq"

    # Playoff beats short_week
    trace = {"context_labels": ["short_week", "playoff"]}
    assert context_slice_for_trace(trace) == "playoff"

    # short_week beats backup_qb
    trace = {"context_labels": ["backup_qb", "short_week"]}
    assert context_slice_for_trace(trace) == "short_week"


def test_context_slice_for_trace_soccer_aliases():
    trace = {"context_labels": ["knockout"]}
    assert context_slice_for_trace(trace, sport_family="soccer") == "cup_match"
    
    trace = {"context_labels": ["world_cup"]}
    assert context_slice_for_trace(trace, sport_family="soccer") == "cup_match"


def test_context_slice_for_trace_tennis_surface():
    trace = {"context_labels": ["clay"]}
    # If no more specific label is present, it uses surface_clay
    assert context_slice_for_trace(trace, sport_family="tennis") == "surface_clay"
    
    trace = {"context_labels": ["playoff", "clay"]}
    # Playoff beats surface
    assert context_slice_for_trace(trace, sport_family="tennis") == "playoff"


def test_context_slice_for_trace_malformed():
    # Should not crash
    assert context_slice_for_trace({"context_labels": None}) == BASE_CONTEXT_SLICE
    assert context_slice_for_trace({"context_labels": [1, 2, None]}) == BASE_CONTEXT_SLICE
