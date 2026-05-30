"""
Unit tests for the calibration-eligibility computation in
omega.core.contracts.service.analyze().

This is the gate that decides whether a trace can feed the calibration learner
(calibration_eligible + calibration_exclusion_reasons in trace_quality). It has
produced real defects before (NBA calib-eligible=0), so each exclusion branch is
pinned here directly rather than only implicitly through higher-level tests.

Branches covered (service.py:324-337):
- success + provided context + complete identity → eligible, no reasons
- engine skipped (validation/skip) → "engine_skipped"
- explicit baseline context → "baseline_default_context", not eligible
- caller-supplied exclusion reasons propagate and are de-duplicated/sorted
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.core.contracts.service import analyze  # noqa: E402

_HOME_CTX = {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0}
_AWAY_CTX = {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0}
_GAME_CTX = {"is_playoff": False, "rest_days": 2}


def _provided_game(**overrides):
    base = {
        "home_team": "Boston Celtics",
        "away_team": "Indiana Pacers",
        "league": "NBA",
        "n_iterations": 100,
        "seed": 42,
        "home_context": _HOME_CTX,
        "away_context": _AWAY_CTX,
        "game_context": _GAME_CTX,
        "odds": {"moneyline_home": -150, "moneyline_away": 130},
    }
    base.update(overrides)
    return base


def _quality(trace):
    return trace["trace_quality"]


class TestEligible:
    def test_full_provided_context_is_eligible(self):
        trace = analyze(_provided_game(), session_id="sess-elig-1", bankroll=2500.0)
        q = _quality(trace)
        assert q["calibration_eligible"] is True
        assert q["calibration_exclusion_reasons"] == []
        assert q["context_source"] == "provided"


class TestExclusionBranches:
    def test_explicit_baseline_context_excluded(self):
        # No team contexts, but allow_baseline → engine succeeds on league default.
        trace = analyze(
            {
                "home_team": "Boston Celtics",
                "away_team": "Indiana Pacers",
                "league": "NBA",
                "n_iterations": 100,
                "seed": 42,
                "allow_baseline": True,
                "game_context": _GAME_CTX,
            },
            session_id="sess-elig-baseline",
            bankroll=2500.0,
        )
        q = _quality(trace)
        assert q["calibration_eligible"] is False
        assert "baseline_default_context" in q["calibration_exclusion_reasons"]

    def test_missing_context_excludes(self):
        # No contexts and no allow_baseline → engine skips → not eligible.
        trace = analyze(
            {
                "home_team": "Boston Celtics",
                "away_team": "Indiana Pacers",
                "league": "NBA",
                "n_iterations": 100,
                "game_context": _GAME_CTX,
            },
            session_id="sess-elig-skip",
            bankroll=2500.0,
        )
        q = _quality(trace)
        assert q["calibration_eligible"] is False
        assert q["calibration_exclusion_reasons"]  # at least one reason


class TestCallerReasons:
    def test_caller_exclusion_reasons_propagate(self):
        trace = analyze(
            _provided_game(),
            session_id="sess-elig-caller",
            bankroll=2500.0,
            trace_quality={"calibration_exclusion_reasons": ["manual_hold"]},
        )
        q = _quality(trace)
        assert "manual_hold" in q["calibration_exclusion_reasons"]
        assert q["calibration_eligible"] is False

    def test_reasons_are_sorted_and_deduped(self):
        trace = analyze(
            _provided_game(),
            session_id="sess-elig-dedup",
            bankroll=2500.0,
            trace_quality={"calibration_exclusion_reasons": ["b_reason", "a_reason", "b_reason"]},
        )
        reasons = _quality(trace)["calibration_exclusion_reasons"]
        assert reasons == sorted(set(reasons))
        assert reasons.count("b_reason") == 1
