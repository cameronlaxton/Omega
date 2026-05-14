"""
Tests for frozen artifacts — conversion, deterministic IDs, round-trip, parity.

All tests are deterministic — no network calls, no LLM.
"""

import pytest


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _make_trace_dict():
    """A minimal ExecutionTrace dict as stored in TraceStore."""
    return {
        "trace_id": "abc123-def456",
        "run_id": "run-001",
        "timestamp": "2025-03-01T18:00:00+00:00",
        "prompt": "Who wins Celtics vs Pacers tonight?",
        "league": "NBA",
        "matchup": "Pacers @ Celtics",
        "execution_mode": "NATIVE_SIM",
        "simulation_seed": 98765,
        "aggregate_quality": 0.85,
        "predictions": {
            "home_win_prob": 65.0,
            "away_win_prob": 35.0,
            "predicted_spread": -5.2,
        },
        "odds_snapshot": {
            "moneyline_home": -180,
            "moneyline_away": 155,
            "spread_home": -4.5,
            "over_under": 224.5,
        },
        "execution_result": {
            "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            "away_context": {"off_rating": 112.0, "def_rating": 112.0, "pace": 98.0},
        },
        "downgrades": [],
    }


def _make_outcome_dict():
    """Outcome dict as stored in the outcomes table."""
    return {"home_score": 112, "away_score": 101}


def _make_legacy_game_dict():
    """A HistoricalGame-style dict (legacy format)."""
    return {
        "home_team": "Celtics",
        "away_team": "Pacers",
        "league": "NBA",
        "date": "2025-03-01",
        "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        "away_context": {"off_rating": 112.0, "def_rating": 112.0, "pace": 98.0},
        "odds": {"moneyline_home": -180, "moneyline_away": 155},
        "outcome": {"home_score": 112, "away_score": 101},
        "closing_odds": {"moneyline_home": -190, "moneyline_away": 165},
    }


# -----------------------------------------------------------------------
# FrozenArtifact model tests
# -----------------------------------------------------------------------

class TestFrozenArtifact:
    """Test the FrozenArtifact Pydantic model."""

    def test_create_valid(self):
        from omega.strategy.artifacts import FrozenArtifact
        artifact = FrozenArtifact(
            artifact_id="test123",
            home_team="Celtics",
            away_team="Pacers",
            league="NBA",
            date="2025-03-01",
            home_context={"off_rating": 118.0},
            away_context={"off_rating": 112.0},
            odds={"moneyline_home": -180},
            simulation_seed=42,
        )
        assert artifact.artifact_id == "test123"
        assert artifact.schema_version == 1
        assert artifact.calibration_policy == "static_v1"
        assert artifact.outcome is None

    def test_serialization_round_trip(self):
        from omega.strategy.artifacts import FrozenArtifact
        artifact = FrozenArtifact(
            artifact_id="rt-test",
            home_team="Lakers",
            away_team="Suns",
            league="NBA",
            date="2025-03-15",
            home_context={"off_rating": 115.0, "def_rating": 110.0},
            away_context={"off_rating": 114.0, "def_rating": 109.0},
            odds={"moneyline_home": -130, "moneyline_away": 110},
            simulation_seed=12345,
            outcome={"home_score": 108, "away_score": 105},
        )
        dumped = artifact.model_dump()
        restored = FrozenArtifact(**dumped)
        assert restored == artifact

    def test_json_round_trip(self):
        from omega.strategy.artifacts import FrozenArtifact
        artifact = FrozenArtifact(
            artifact_id="json-test",
            home_team="Nuggets",
            away_team="Heat",
            league="NBA",
            date="2025-03-20",
            odds={"moneyline_home": -150},
        )
        json_str = artifact.model_dump_json()
        restored = FrozenArtifact.model_validate_json(json_str)
        assert restored == artifact


# -----------------------------------------------------------------------
# Deterministic artifact ID
# -----------------------------------------------------------------------

class TestArtifactId:
    """artifact_id must be deterministic for the same event identity."""

    def test_same_inputs_same_id(self):
        from omega.strategy.artifacts import compute_artifact_id
        id1 = compute_artifact_id("Celtics", "Pacers", "NBA", "2025-03-01")
        id2 = compute_artifact_id("Celtics", "Pacers", "NBA", "2025-03-01")
        assert id1 == id2

    def test_different_date_different_id(self):
        from omega.strategy.artifacts import compute_artifact_id
        id1 = compute_artifact_id("Celtics", "Pacers", "NBA", "2025-03-01")
        id2 = compute_artifact_id("Celtics", "Pacers", "NBA", "2025-03-02")
        assert id1 != id2

    def test_different_teams_different_id(self):
        from omega.strategy.artifacts import compute_artifact_id
        id1 = compute_artifact_id("Celtics", "Pacers", "NBA", "2025-03-01")
        id2 = compute_artifact_id("Lakers", "Pacers", "NBA", "2025-03-01")
        assert id1 != id2

    def test_id_is_hex_string(self):
        from omega.strategy.artifacts import compute_artifact_id
        aid = compute_artifact_id("Celtics", "Pacers", "NBA", "2025-03-01")
        assert len(aid) == 16
        int(aid, 16)  # must be valid hex


# -----------------------------------------------------------------------
# trace_to_artifact converter
# -----------------------------------------------------------------------

class TestTraceToArtifact:
    """Test conversion from ExecutionTrace dict to FrozenArtifact."""

    def test_basic_conversion(self):
        from omega.strategy.artifacts import trace_to_artifact
        trace = _make_trace_dict()
        outcome = _make_outcome_dict()
        artifact = trace_to_artifact(trace, outcome)

        assert artifact.home_team == "Celtics"
        assert artifact.away_team == "Pacers"
        assert artifact.league == "NBA"
        assert artifact.date == "2025-03-01"
        assert artifact.simulation_seed == 98765
        assert artifact.source_trace_id == "abc123-def456"
        assert artifact.outcome == outcome
        assert artifact.odds["moneyline_home"] == -180

    def test_contexts_from_execution_result(self):
        from omega.strategy.artifacts import trace_to_artifact
        trace = _make_trace_dict()
        artifact = trace_to_artifact(trace)

        assert artifact.home_context["off_rating"] == 118.0
        assert artifact.away_context["off_rating"] == 112.0

    def test_no_outcome_produces_none(self):
        from omega.strategy.artifacts import trace_to_artifact
        trace = _make_trace_dict()
        artifact = trace_to_artifact(trace)
        assert artifact.outcome is None

    def test_deterministic_id_from_trace(self):
        from omega.strategy.artifacts import trace_to_artifact, compute_artifact_id
        trace = _make_trace_dict()
        artifact = trace_to_artifact(trace)
        expected_id = compute_artifact_id("Celtics", "Pacers", "NBA", "2025-03-01")
        assert artifact.artifact_id == expected_id


# -----------------------------------------------------------------------
# compat_dict_to_artifact converter
# -----------------------------------------------------------------------

class TestCompatDictToArtifact:
    """Test legacy HistoricalGame dict conversion."""

    def test_basic_conversion(self):
        from omega.strategy.artifacts import compat_dict_to_artifact
        game = _make_legacy_game_dict()
        artifact = compat_dict_to_artifact(game)

        assert artifact.home_team == "Celtics"
        assert artifact.away_team == "Pacers"
        assert artifact.league == "NBA"
        assert artifact.date == "2025-03-01"
        assert artifact.outcome == {"home_score": 112, "away_score": 101}
        assert artifact.closing_odds == {"moneyline_home": -190, "moneyline_away": 165}

    def test_missing_fields_use_defaults(self):
        from omega.strategy.artifacts import compat_dict_to_artifact
        artifact = compat_dict_to_artifact({"home_team": "X", "away_team": "Y"})
        assert artifact.league == "NBA"  # default
        assert artifact.outcome is None
        assert artifact.home_context == {}


# -----------------------------------------------------------------------
# Backtest parity: FrozenArtifact vs legacy dict
# -----------------------------------------------------------------------

class TestBacktestParity:
    """Same game via FrozenArtifact and legacy dict must produce identical results."""

    def test_artifact_vs_dict_parity(self):
        import numpy as np
        from omega.strategy.backtest.engine import BacktestEngine, HistoricalGame
        from omega.strategy.artifacts import compat_dict_to_artifact
        from omega.strategy.models import StrategyEntry

        strategy = StrategyEntry(
            strategy_id="parity-test",
            version=1,
            name="Parity Test",
            leagues=["NBA"],
            edge_threshold=0.01,
            confidence_tiers=["A", "B", "C"],
        )

        game_dict = _make_legacy_game_dict()
        artifact = compat_dict_to_artifact(game_dict)

        # Reset numpy RNG to ensure identical simulation outputs
        np.random.seed(42)
        engine_dict = BacktestEngine(n_iterations=500, seed=42)
        result_dict = engine_dict.run(strategy, [HistoricalGame(game_dict)])

        np.random.seed(42)
        engine_artifact = BacktestEngine(n_iterations=500, seed=42)
        result_artifact = engine_artifact.run(strategy, [artifact])

        # Core metrics must match
        assert result_dict.total_bets_placed == result_artifact.total_bets_placed
        assert result_dict.win_count == result_artifact.win_count
        assert result_dict.loss_count == result_artifact.loss_count
        assert result_dict.net_units == result_artifact.net_units
        assert result_dict.roi_pct == result_artifact.roi_pct

    def test_artifact_backtest_has_trace_ids(self):
        from omega.strategy.backtest.engine import BacktestEngine
        from omega.strategy.artifacts import FrozenArtifact
        from omega.strategy.models import StrategyEntry

        strategy = StrategyEntry(
            strategy_id="trace-link-test",
            version=1,
            name="Trace Link Test",
            leagues=["NBA"],
            edge_threshold=0.01,
            confidence_tiers=["A", "B", "C"],
        )

        artifact = FrozenArtifact(
            artifact_id="test-id",
            source_trace_id="trace-abc-123",
            home_team="Celtics",
            away_team="Pacers",
            league="NBA",
            date="2025-03-01",
            home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            away_context={"off_rating": 112.0, "def_rating": 112.0, "pace": 98.0},
            odds={"moneyline_home": -180, "moneyline_away": 155},
            outcome={"home_score": 112, "away_score": 101},
        )

        engine = BacktestEngine(n_iterations=500, seed=42)
        result = engine.run(strategy, [artifact])

        assert "trace-abc-123" in result.trace_ids


# -----------------------------------------------------------------------
# BacktestResult new fields
# -----------------------------------------------------------------------

class TestBacktestResultFields:
    """Test that BacktestResult includes Phase 6b fields."""

    def test_default_fields(self):
        from omega.strategy.models import BacktestResult
        result = BacktestResult(
            strategy_id="test",
            strategy_version=1,
            run_id="bt-test",
            started_at="2025-03-01T00:00:00Z",
        )
        assert result.artifact_schema_version == 1
        assert result.calibration_policy == "static_v1"
        assert result.trace_ids == []
