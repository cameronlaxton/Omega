"""
Tests for frozen artifacts — conversion, deterministic IDs, round-trip, parity.

All tests are deterministic — no network calls, no LLM.
"""


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
        "context_labels": {"is_playoff": True},
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
        # Schema v2 (plan 5.4): governed-substrate fields default to None so
        # v1-shaped construction stays valid and behavior-identical.
        assert artifact.schema_version == 2
        assert artifact.simulation_backend is None
        assert artifact.prior_payload is None
        assert artifact.substrate_unresolved is False
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
        assert artifact.game_context["is_playoff"] is True

    def test_no_outcome_produces_none(self):
        from omega.strategy.artifacts import trace_to_artifact

        trace = _make_trace_dict()
        artifact = trace_to_artifact(trace)
        assert artifact.outcome is None

    def test_deterministic_id_from_trace(self):
        from omega.strategy.artifacts import compute_artifact_id, trace_to_artifact

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

    def test_contexts_from_canonical_input_snapshot(self):
        from omega.strategy.artifacts import trace_to_artifact

        trace = {
            "trace_id": "canonical-1",
            "timestamp": "2026-05-21T12:00:00Z",
            "league": "NBA",
            "matchup": "Pacers @ Celtics",
            "input_snapshot": {
                "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
                "away_context": {"off_rating": 112.0, "def_rating": 112.0, "pace": 98.0},
                "game_context": {"rest_days": 0},
            },
            "odds_snapshot": {"moneyline_home": -150},
        }

        artifact = trace_to_artifact(trace)

        assert artifact.home_context["off_rating"] == 118.0
        assert artifact.away_context["off_rating"] == 112.0
        assert artifact.game_context == {"rest_days": 0}


# -----------------------------------------------------------------------
# Backtest parity: FrozenArtifact vs legacy dict
# -----------------------------------------------------------------------


class TestBacktestParity:
    """Same game via FrozenArtifact and legacy dict must produce identical results."""

    def test_artifact_vs_dict_parity(self):
        import numpy as np

        from omega.strategy.artifacts import compat_dict_to_artifact
        from omega.strategy.backtest.engine import BacktestEngine, HistoricalGame
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
        from omega.strategy.artifacts import FrozenArtifact
        from omega.strategy.backtest.engine import BacktestEngine
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


# -----------------------------------------------------------------------
# FrozenPropArtifact — prop-plane frozen inputs (structural sweep)
# -----------------------------------------------------------------------


def _make_prop_trace_dict():
    """A graded prop trace dict as returned by TraceStore.query_traces()."""
    return {
        "trace_id": "prop-trace-1",
        "kind": "prop",
        "timestamp": "2026-06-17T10:00:00+00:00",  # replay RUN date, not the game
        "decision_time": "2025-11-09T18:00:00+00:00",
        "league": "NFL",
        "simulation_seed": 4242,
        "input_snapshot": {
            "player_name": "D. Runner",
            "prop_type": "rushing_yards",
            "line": 74.5,
            "league": "NFL",
        },
        "predictions": {"over_prob": 0.55, "under_prob": 0.43},
        "simulation_distributions": [
            {
                "target": "player_stat",
                "market": "player_prop",
                "stat_key": "rushing_yards",
                "distribution_type": "negative_binomial_exact",
                "distribution_params": {"mu": 78.2, "k": 22.5, "p": 0.22},
                "sample_std": 19.4,
                "n_iterations": 2000,
                "seed": 4242,
            }
        ],
        "_prop_outcomes": [
            {"side": "over", "result": "win", "stat_value": 91.0, "line": 74.5},
            {"side": "under", "result": "loss", "stat_value": 91.0, "line": 74.5},
            {"side": "over", "result": "push", "stat_value": 74.5, "line": 74.5},
        ],
    }


class TestFrozenPropArtifact:
    """FrozenPropArtifact round-trip + the single prop-trace builder."""

    def test_builder_recovers_sim_param_point(self):
        from omega.strategy.artifacts import (
            compute_prop_artifact_id,
            prop_trace_to_frozen_artifact,
        )

        art = prop_trace_to_frozen_artifact(_make_prop_trace_dict())
        assert art is not None
        assert art.player_name == "D. Runner"
        assert art.stat_type == "rushing_yards"
        assert art.line == 74.5
        assert art.projection_mean == 78.2
        assert art.nb_dispersion_k == 22.5
        assert art.projection_std == 19.4
        assert art.simulation_seed == 4242
        assert art.source_trace_id == "prop-trace-1"
        # Dated by decision_time (no-leak split key), NOT the replay run day.
        assert art.date == "2025-11-09"
        assert art.artifact_id == compute_prop_artifact_id(
            "D. Runner", "NFL", "rushing_yards", 74.5, "2025-11-09"
        )
        # One-to-many outcome rows survive intact (push excluded only at grading).
        assert len(art.prop_outcomes) == 3

    def test_builder_falls_back_to_timestamp_date(self):
        from omega.strategy.artifacts import prop_trace_to_frozen_artifact

        trace = _make_prop_trace_dict()
        del trace["decision_time"]
        art = prop_trace_to_frozen_artifact(trace)
        assert art is not None and art.date == "2026-06-17"

    def test_builder_canonicalizes_stat_aliases(self):
        from omega.strategy.artifacts import prop_trace_to_frozen_artifact

        trace = _make_prop_trace_dict()
        trace["input_snapshot"]["prop_type"] = "rush_yds"
        art = prop_trace_to_frozen_artifact(trace)
        assert art is not None and art.stat_type == "rushing_yards"

    def test_builder_reads_store_attached_distribution_rows(self):
        from omega.strategy.artifacts import prop_trace_to_frozen_artifact

        trace = _make_prop_trace_dict()
        trace["_simulation_distributions"] = trace.pop("simulation_distributions")
        art = prop_trace_to_frozen_artifact(trace)
        assert art is not None and art.nb_dispersion_k == 22.5

    def test_builder_divides_out_echoed_production_scale(self):
        """A trace priced under a promoted profile persists the POST-scale k plus
        the echoed nb_k_scale; the builder recovers the PRE-scale base so a future
        sweep never layers a candidate scale on top of production's."""
        from omega.strategy.artifacts import prop_trace_to_frozen_artifact

        trace = _make_prop_trace_dict()
        params = trace["simulation_distributions"][0]["distribution_params"]
        params["k"] = 22.5 * 1.5  # final k the production sim ran with
        params["nb_k_scale"] = 1.5  # echoed by the backend when the profile applied
        art = prop_trace_to_frozen_artifact(trace)
        assert art is not None
        assert abs(art.nb_dispersion_k - 22.5) < 1e-9

    def test_builder_fails_closed_without_resim_inputs(self):
        from omega.strategy.artifacts import prop_trace_to_frozen_artifact

        # No NB distribution params -> cannot re-simulate faithfully.
        no_params = _make_prop_trace_dict()
        no_params["simulation_distributions"][0]["distribution_params"] = {}
        assert prop_trace_to_frozen_artifact(no_params) is None
        # Non-positive k is unusable for the NB backend.
        bad_k = _make_prop_trace_dict()
        bad_k["simulation_distributions"][0]["distribution_params"]["k"] = 0.0
        assert prop_trace_to_frozen_artifact(bad_k) is None
        # No line -> no market to price.
        no_line = _make_prop_trace_dict()
        del no_line["input_snapshot"]["line"]
        assert prop_trace_to_frozen_artifact(no_line) is None
        # Only push/void rows -> no calibration signal.
        no_grade = _make_prop_trace_dict()
        no_grade["_prop_outcomes"] = [{"side": "over", "result": "push"}]
        assert prop_trace_to_frozen_artifact(no_grade) is None
        # No outcome rows at all.
        ungraded = _make_prop_trace_dict()
        ungraded["_prop_outcomes"] = []
        assert prop_trace_to_frozen_artifact(ungraded) is None

    def test_round_trip_serialization(self):
        from omega.strategy.artifacts import FrozenPropArtifact, prop_trace_to_frozen_artifact

        art = prop_trace_to_frozen_artifact(_make_prop_trace_dict())
        assert art is not None
        restored = FrozenPropArtifact.model_validate_json(art.model_dump_json())
        assert restored == art
