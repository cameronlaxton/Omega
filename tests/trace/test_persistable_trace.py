from __future__ import annotations

from omega.trace.persistable import PersistableTrace
from omega.trace.store import TraceStore


def _analyze_trace(kind: str = "game") -> dict:
    if kind == "prop":
        return {
            "trace_id": "sandbox-prop-persist",
            "model_version": "omega-core-phase6h",
            "ran_at": "2026-05-21T12:00:00Z",
            "kind": "prop",
            "session_id": "sess-persist",
            "bankroll": 1000.0,
            "input_snapshot": {
                "player_name": "Jayson Tatum",
                "league": "NBA",
                "prop_type": "pts",
                "line": 27.5,
                "odds_over": -110,
                "home_team": "Heat",
                "away_team": "Celtics",
                "game_date": "2026-05-21",
                "game_context": {"is_playoff": True},
            },
            "result": {
                "league": "NBA",
                "over_prob": 0.58,
                "recommendation": "over",
                "recommended_units": 1.2,
                "kelly_fraction": 0.012,
                "bet_side_odds": -110,
                "over_calibration_audit": {"profile_id": "playoff-v1"},
            },
            "context_labels": {"is_playoff": True},
            "downgrades": [],
        }
    return {
        "trace_id": "sandbox-game-persist",
        "model_version": "omega-core-phase6h",
        "ran_at": "2026-05-21T12:00:00Z",
        "kind": "game",
        "input_snapshot": {
            "home_team": "Celtics",
            "away_team": "Pacers",
            "league": "NBA",
            "seed": 42,
            "odds": {"moneyline_home": -150},
            "game_context": {"rest_days": 0},
        },
        "result": {
            "matchup": "Pacers @ Celtics",
            "league": "NBA",
            "simulation": {"home_win_prob": 58.0},
            "edges": [{"calibration_audit": {"profile_id": "b2b-v1"}}],
        },
        "context_labels": {"rest_days": 0},
    }


def test_persistable_trace_from_game_analyze_output_carries_query_fields():
    trace = PersistableTrace.from_analyze_output(_analyze_trace("game"))
    record = trace.to_store_record()

    assert record["trace_id"] == "sandbox-game-persist"
    assert record["run_id"] == "sandbox-game-persist"
    assert record["timestamp"] == "2026-05-21T12:00:00Z"
    assert record["kind"] == "game"
    assert record["league"] == "NBA"
    assert record["matchup"] == "Pacers @ Celtics"
    assert record["context_labels"] == {"rest_days": 0}
    assert record["calibration_audit"] == [{"profile_id": "b2b-v1"}]


def test_from_analyze_output_falls_back_to_legacy_timestamp_key():
    """Pre-Phase-6h direct-analyze() exports carry a top-level `timestamp`
    instead of `ran_at`/`analyzed_at`; the adapter must not treat that as
    missing (see docs/bugs/ export-wrapper timestamp gap)."""
    analyze_out = _analyze_trace("game")
    del analyze_out["ran_at"]
    analyze_out["timestamp"] = "2026-06-25T22:05:32.476657+00:00"

    trace = PersistableTrace.from_analyze_output(analyze_out)

    assert trace.timestamp == "2026-06-25T22:05:32.476657+00:00"


def test_from_analyze_output_prefers_ran_at_over_legacy_timestamp():
    analyze_out = _analyze_trace("game")
    analyze_out["timestamp"] = "2020-01-01T00:00:00Z"  # should be ignored

    trace = PersistableTrace.from_analyze_output(analyze_out)

    assert trace.timestamp == "2026-05-21T12:00:00Z"


def test_trace_store_accepts_persistable_trace_model(tmp_path):
    store = TraceStore(db_path=str(tmp_path / "traces.db"))
    trace = PersistableTrace.from_analyze_output(_analyze_trace("prop"))

    trace_id = store.persist(trace)
    retrieved = store.get_trace(trace_id)
    store.close()

    assert retrieved is not None
    assert retrieved["recommendations"]["recommendation"] == "over"
    assert retrieved["context_labels"] == {"is_playoff": True}
