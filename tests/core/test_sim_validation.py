"""Tests for omega.core.simulation.validation — sim-input boundary hardening."""

import math
import pytest

from omega.core.simulation.validation import validate_sim_context, SIM_INPUT_BOUNDS


# ---------------------------------------------------------------------------
# Happy path: valid data passes through
# ---------------------------------------------------------------------------

class TestValidDataPassthrough:
    def test_basketball_valid_context(self):
        ctx = {"off_rating": 112.5, "def_rating": 108.3, "pace": 100.0}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {"off_rating": 112.5, "def_rating": 108.3, "pace": 100.0}

    def test_football_valid_context(self):
        ctx = {"off_rating": 95.0, "def_rating": 90.0, "pass_eff": 7.5, "rush_eff": 4.2}
        result = validate_sim_context(ctx, "NFL", "away")
        assert result == ctx

    def test_baseball_valid_context(self):
        ctx = {"off_rating": 100.0, "def_rating": 98.0, "era": 3.50, "batting_avg": 0.265}
        result = validate_sim_context(ctx, "MLB", "home")
        assert result == ctx

    def test_hockey_valid_context(self):
        ctx = {"off_rating": 105.0, "def_rating": 100.0, "save_pct": 0.920, "shots_per_game": 32.0}
        result = validate_sim_context(ctx, "NHL", "home")
        assert result == ctx

    def test_soccer_valid_context(self):
        ctx = {"off_rating": 110.0, "def_rating": 95.0, "xg_for": 1.8, "possession_pct": 55.0}
        result = validate_sim_context(ctx, "EPL", "home")
        assert result == ctx

    def test_tennis_valid_context(self):
        ctx = {"serve_win_pct": 0.65, "return_win_pct": 0.42}
        result = validate_sim_context(ctx, "ATP", "home")
        assert result == ctx

    def test_golf_valid_context(self):
        ctx = {"strokes_gained_total": 2.5, "sg_putting": 0.8}
        result = validate_sim_context(ctx, "PGA", "home")
        assert result == ctx

    def test_fighting_valid_context(self):
        ctx = {"win_pct": 0.75, "finish_rate": 0.60, "ko_tko_rate": 0.40}
        result = validate_sim_context(ctx, "UFC", "home")
        assert result == ctx

    def test_esports_valid_context(self):
        ctx = {"map_win_rate": 0.58, "recent_form": 1.2}
        result = validate_sim_context(ctx, "CS2", "home")
        assert result == ctx


# ---------------------------------------------------------------------------
# String coercion
# ---------------------------------------------------------------------------

class TestStringCoercion:
    def test_string_numeric_coerced(self):
        ctx = {"off_rating": "112.5", "def_rating": "108.0"}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {"off_rating": 112.5, "def_rating": 108.0}

    def test_string_integer_coerced(self):
        ctx = {"pace": "100"}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {"pace": 100.0}


# ---------------------------------------------------------------------------
# Bounds rejection
# ---------------------------------------------------------------------------

class TestBoundsRejection:
    def test_off_rating_too_low(self):
        ctx = {"off_rating": 0.5, "def_rating": 105.0}
        result = validate_sim_context(ctx, "NBA", "home")
        assert "off_rating" not in result
        assert result["def_rating"] == 105.0

    def test_off_rating_too_high(self):
        ctx = {"off_rating": 9999.0}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {}

    def test_pace_out_of_bounds(self):
        ctx = {"pace": 9000.0}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {}

    def test_win_pct_over_one(self):
        ctx = {"win_pct": 1.5, "finish_rate": 0.6}
        result = validate_sim_context(ctx, "UFC", "home")
        assert "win_pct" not in result
        assert result["finish_rate"] == 0.6

    def test_boundary_values_accepted(self):
        # Exact boundary values should be accepted
        ctx = {"off_rating": 80.0, "def_rating": 140.0}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {"off_rating": 80.0, "def_rating": 140.0}


# ---------------------------------------------------------------------------
# Garbage rejection
# ---------------------------------------------------------------------------

class TestGarbageRejection:
    def test_non_numeric_string_dropped(self):
        ctx = {"off_rating": "not a number", "def_rating": 105.0}
        result = validate_sim_context(ctx, "NBA", "home")
        assert "off_rating" not in result
        assert result["def_rating"] == 105.0

    def test_none_value_dropped(self):
        ctx = {"off_rating": None, "def_rating": 105.0}
        result = validate_sim_context(ctx, "NBA", "home")
        assert "off_rating" not in result

    def test_list_value_dropped(self):
        ctx = {"off_rating": [112.5], "def_rating": 105.0}
        result = validate_sim_context(ctx, "NBA", "home")
        assert "off_rating" not in result

    def test_dict_value_dropped(self):
        ctx = {"off_rating": {"value": 112.5}}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {}

    def test_nan_dropped(self):
        ctx = {"off_rating": float("nan")}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {}

    def test_inf_dropped(self):
        ctx = {"off_rating": float("inf")}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {}

    def test_negative_inf_dropped(self):
        ctx = {"off_rating": float("-inf")}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {}

    def test_bool_dropped(self):
        ctx = {"off_rating": True}
        # True is technically an int in Python but coerces to 1.0 — within bounds or not
        # depends on the key. For off_rating bounds [80, 140], 1.0 is out of bounds.
        result = validate_sim_context(ctx, "NBA", "home")
        assert "off_rating" not in result


# ---------------------------------------------------------------------------
# Unknown key stripping
# ---------------------------------------------------------------------------

class TestUnknownKeyStripping:
    def test_unknown_keys_stripped(self):
        ctx = {
            "off_rating": 112.0,
            "garbage_key": 42.0,
            "another_unknown": "hello",
        }
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {"off_rating": 112.0}
        assert "garbage_key" not in result
        assert "another_unknown" not in result

    def test_internal_keys_stripped(self):
        ctx = {"off_rating": 112.0, "_raw_text": "some text"}
        result = validate_sim_context(ctx, "NBA", "home")
        assert "_raw_text" not in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_none_context(self):
        result = validate_sim_context(None, "NBA", "home")
        assert result == {}

    def test_empty_context(self):
        result = validate_sim_context({}, "NBA", "home")
        assert result == {}

    def test_unknown_league_passthrough(self):
        ctx = {"some_key": 42.0}
        result = validate_sim_context(ctx, "UNKNOWN_LEAGUE", "home")
        assert result == {"some_key": 42.0}

    def test_integer_values_accepted(self):
        ctx = {"off_rating": 112, "def_rating": 108}
        result = validate_sim_context(ctx, "NBA", "home")
        assert result == {"off_rating": 112.0, "def_rating": 108.0}

    def test_strict_param_accepted(self):
        # strict=True with sufficient valid data passes without error
        ctx = {"off_rating": 112.0, "def_rating": 108.0}
        result = validate_sim_context(ctx, "NBA", "home", strict=True)
        assert result == {"off_rating": 112.0, "def_rating": 108.0}


# ---------------------------------------------------------------------------
# Bounds table coverage
# ---------------------------------------------------------------------------

class TestBoundsTableCoverage:
    def test_all_bounds_have_valid_ranges(self):
        for key, (lo, hi) in SIM_INPUT_BOUNDS.items():
            assert lo < hi, f"Bounds for {key}: min ({lo}) >= max ({hi})"

    def test_all_archetype_critical_keys_have_bounds(self):
        """Every critical key in every archetype should have a bounds entry."""
        from omega.core.simulation.archetypes import ARCHETYPE_REGISTRY
        missing = []
        for name, arch in ARCHETYPE_REGISTRY.items():
            for key in arch.critical_team_keys:
                if key not in SIM_INPUT_BOUNDS:
                    missing.append(f"{name}.{key}")
        assert missing == [], f"Critical keys without bounds: {missing}"
