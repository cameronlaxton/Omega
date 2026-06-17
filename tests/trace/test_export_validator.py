"""Tests for omega.trace.export_validator — pre-ingest shape/quality validation."""

from __future__ import annotations

from typing import Any

from omega.trace.export_validator import validate_export_block


def _good_game_block() -> dict[str, Any]:
    # MLB so the NBA game_context requirement does not apply to the clean case.
    return {
        "trace": {
            "trace_id": "sandbox-abc",
            "ran_at": "2026-05-28T00:00:00Z",
            "kind": "game",
            "session_id": "sess-20260528-aaaa",
            "input_snapshot": {"home_team": "BOS", "away_team": "NYY", "league": "MLB"},
            "result": {
                "status": "success",
                "simulation": {"home_win_prob": 0.55, "away_win_prob": 0.45},
            },
            "trace_quality": {"calibration_eligible": True},
        },
        "bet_record": None,
    }


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


class TestShape:
    def test_good_shape_a_passes_strict(self):
        r = validate_export_block(_good_game_block(), strict=True)
        assert r.ok, r.summary()
        assert r.trace_id == "sandbox-abc"

    def test_good_shape_b_bare_trace_passes(self):
        block = _good_game_block()["trace"]  # raw analyze() output, no wrapper
        r = validate_export_block(block, strict=True)
        assert r.ok, r.summary()

    def test_unsupported_wrapper_rejected(self):
        r = validate_export_block({"foo": 1, "bar": 2}, strict=True)
        assert not r.ok
        assert "shape" in _codes(r)


class TestErrorsMirrorIngest:
    def test_missing_trace_id_rejected(self):
        block = _good_game_block()
        block["trace"]["trace_id"] = ""
        r = validate_export_block(block, strict=False)
        assert not r.ok

    def test_prop_bet_missing_identity_rejected_in_both_modes(self):
        block = {
            "trace": {
                "trace_id": "sandbox-p",
                "ran_at": "2026-05-28T00:00:00Z",
                "kind": "prop",
                "session_id": "s",
                "input_snapshot": {"player_name": "X", "prop_type": "pts", "line": 1.5},
                "result": {"status": "success", "over_prob": 0.5, "under_prob": 0.5},
            },
            "bet_record": {
                "book": "BetMGM",
                "market": "player_prop:pts",
                "selection": "X over 1.5",
                "selection_descriptor": "X_over_1.5_pts",
                "odds_taken": -110,
                "stake_units": 1.0,
                "decision_timestamp": "2026-05-28T00:00:00Z",
            },
        }
        assert "prop_bet_identity" in _codes(validate_export_block(block, strict=False))

    def test_malformed_bet_record_rejected(self):
        block = {
            "trace": {
                "trace_id": "sandbox-p2",
                "ran_at": "2026-05-28T00:00:00Z",
                "kind": "prop",
                "session_id": "s",
                "input_snapshot": {
                    "player_name": "X",
                    "prop_type": "pts",
                    "line": 1.5,
                    "home_team": "BOS",
                    "away_team": "NYY",
                    "game_date": "2026-05-28",
                },
                "result": {"status": "success", "over_prob": 0.5, "under_prob": 0.5},
            },
            "bet_record": {"book": "BetMGM"},  # missing odds_taken/stake_units/etc.
        }
        assert "bet_record" in _codes(validate_export_block(block, strict=False))


class TestStrictVsLenient:
    def test_missing_session_id_strict_error_lenient_warn(self):
        block = _good_game_block()
        del block["trace"]["session_id"]
        strict = validate_export_block(block, strict=True)
        lenient = validate_export_block(block, strict=False)
        assert not strict.ok and "session_id" in {i.code for i in strict.errors}
        assert lenient.ok and "session_id" in {i.code for i in lenient.warnings}

    def test_missing_result_status_strict_error(self):
        block = _good_game_block()
        block["trace"]["result"].pop("status")
        r = validate_export_block(block, strict=True)
        assert not r.ok and "result_status" in _codes(r)

    def test_missing_predictions_strict_error(self):
        block = _good_game_block()
        block["trace"]["result"] = {"status": "success"}  # no simulation
        r = validate_export_block(block, strict=True)
        assert not r.ok and "predictions" in _codes(r)

    def test_missing_identity_strict_error(self):
        block = _good_game_block()
        block["trace"]["input_snapshot"].pop("home_team")
        r = validate_export_block(block, strict=True)
        assert not r.ok and "identity" in _codes(r)

    def test_nba_missing_game_context_strict_error(self):
        block = _good_game_block()
        block["trace"]["input_snapshot"]["league"] = "NBA"
        r = validate_export_block(block, strict=True)
        assert not r.ok and "nba_game_context" in _codes(r)

    def test_nba_with_game_context_passes(self):
        block = _good_game_block()
        block["trace"]["input_snapshot"]["league"] = "NBA"
        block["trace"]["input_snapshot"]["game_context"] = {"is_playoff": True, "rest_days": 2}
        r = validate_export_block(block, strict=True)
        assert r.ok, r.summary()
