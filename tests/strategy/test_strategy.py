"""
Tests for the strategy layer: registry, backtest, promotion.

All tests are deterministic — no network calls, no LLM.
"""

import pytest

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_registry():
    from omega.strategy.versioning.registry import StrategyRegistry

    return StrategyRegistry()  # in-memory only, no file


def _make_historical_games():
    """Generate a small set of historical NBA games with outcomes."""
    from omega.strategy.backtest.engine import HistoricalGame

    games = []
    # Game 1: Home team stronger, wins
    games.append(
        HistoricalGame(
            {
                "home_team": "Celtics",
                "away_team": "Pacers",
                "league": "NBA",
                "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
                "away_context": {"off_rating": 112.0, "def_rating": 112.0, "pace": 98.0},
                "odds": {"moneyline_home": -180, "moneyline_away": 155},
                "outcome": {"home_score": 112, "away_score": 101},
                "closing_odds": {"moneyline_home": -190, "moneyline_away": 165},
            }
        )
    )

    # Game 2: Away underdog wins (upset)
    games.append(
        HistoricalGame(
            {
                "home_team": "Lakers",
                "away_team": "Magic",
                "league": "NBA",
                "home_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 99.0},
                "away_context": {"off_rating": 108.0, "def_rating": 114.0, "pace": 96.0},
                "odds": {"moneyline_home": -250, "moneyline_away": 210},
                "outcome": {"home_score": 95, "away_score": 102},
            }
        )
    )

    # Game 3: Close game, home wins
    games.append(
        HistoricalGame(
            {
                "home_team": "Nuggets",
                "away_team": "Suns",
                "league": "NBA",
                "home_context": {"off_rating": 116.0, "def_rating": 109.0, "pace": 100.0},
                "away_context": {"off_rating": 114.0, "def_rating": 110.0, "pace": 99.0},
                "odds": {"moneyline_home": -130, "moneyline_away": 110},
                "outcome": {"home_score": 108, "away_score": 105},
            }
        )
    )

    return games


# -----------------------------------------------------------------------
# Registry tests
# -----------------------------------------------------------------------


class TestStrategyRegistry:
    """Test strategy registration, querying, and versioning."""

    def test_register_strategy(self):
        from omega.strategy.models import StrategyStatus

        registry = _make_registry()

        entry = registry.register(
            strategy_id="nba-ml-edge",
            name="NBA Moneyline Edge",
            leagues=["NBA"],
            markets=["moneyline"],
        )
        assert entry.strategy_id == "nba-ml-edge"
        assert entry.version == 1
        assert entry.status == StrategyStatus.CANDIDATE

    def test_auto_increment_version(self):
        registry = _make_registry()

        v1 = registry.register(
            strategy_id="nba-ml-edge",
            name="NBA ML Edge v1",
        )
        v2 = registry.register(
            strategy_id="nba-ml-edge",
            name="NBA ML Edge v2",
            params={"edge_threshold": 0.05},
        )
        assert v1.version == 1
        assert v2.version == 2

    def test_get_latest(self):
        registry = _make_registry()

        registry.register(strategy_id="test", name="v1")
        registry.register(strategy_id="test", name="v2")

        latest = registry.get("test")
        assert latest is not None
        assert latest.version == 2

    def test_get_specific_version(self):
        registry = _make_registry()

        registry.register(strategy_id="test", name="v1")
        registry.register(strategy_id="test", name="v2")

        v1 = registry.get("test", version=1)
        assert v1 is not None
        assert v1.version == 1
        assert v1.name == "v1"

    def test_list_all(self):
        registry = _make_registry()

        registry.register(strategy_id="a", name="A", leagues=["NBA"])
        registry.register(strategy_id="b", name="B", leagues=["NFL"])

        all_strats = registry.list_all()
        assert len(all_strats) == 2

    def test_list_by_league(self):
        registry = _make_registry()

        registry.register(strategy_id="a", name="A", leagues=["NBA"])
        registry.register(strategy_id="b", name="B", leagues=["NFL"])

        nba_only = registry.list_all(league="NBA")
        assert len(nba_only) == 1
        assert nba_only[0].strategy_id == "a"

    def test_get_nonexistent(self):
        registry = _make_registry()
        assert registry.get("nonexistent") is None


# -----------------------------------------------------------------------
# Backtest tests
# -----------------------------------------------------------------------


class TestBacktestEngine:
    """Test the backtest engine."""

    def test_run_backtest(self):
        from omega.strategy.backtest.engine import BacktestEngine
        from omega.strategy.models import StrategyEntry, StrategyType

        engine = BacktestEngine(n_iterations=100, seed=42)
        strategy = StrategyEntry(
            strategy_id="test-bt",
            name="Test Backtest",
            strategy_type=StrategyType.GAME_EDGE,
            leagues=["NBA"],
            markets=["moneyline"],
            edge_threshold=0.02,
            confidence_tiers=["A", "B", "C"],
        )

        games = _make_historical_games()
        result = engine.run(strategy, games)

        assert result.strategy_id == "test-bt"
        assert result.total_games == 3
        assert result.run_id.startswith("bt-")
        assert result.completed_at is not None
        # With only 3 games, should fail sample size check
        assert "Insufficient sample" in str(result.rejection_reasons)

    def test_league_filter(self):
        from omega.strategy.backtest.engine import BacktestEngine
        from omega.strategy.models import StrategyEntry

        engine = BacktestEngine(n_iterations=50)
        strategy = StrategyEntry(
            strategy_id="nfl-only",
            name="NFL Only",
            leagues=["NFL"],  # Won't match NBA games
            edge_threshold=0.01,
            confidence_tiers=["A", "B", "C"],
        )

        games = _make_historical_games()  # All NBA
        result = engine.run(strategy, games)

        assert result.total_bets_placed == 0

    def test_edge_threshold_filter(self):
        from omega.strategy.backtest.engine import BacktestEngine
        from omega.strategy.models import StrategyEntry

        engine = BacktestEngine(n_iterations=100)
        # Very high threshold — should filter out most bets
        strict = StrategyEntry(
            strategy_id="strict",
            name="Strict",
            leagues=["NBA"],
            edge_threshold=0.50,  # 50% edge required — virtually impossible
            confidence_tiers=["A", "B", "C"],
        )

        games = _make_historical_games()
        result = engine.run(strict, games)

        assert result.total_bets_placed == 0

    def test_backtest_result_structure(self):
        from omega.strategy.backtest.engine import BacktestEngine
        from omega.strategy.models import StrategyEntry

        engine = BacktestEngine(n_iterations=100)
        strategy = StrategyEntry(
            strategy_id="structure-test",
            name="Structure",
            leagues=["NBA"],
            edge_threshold=0.01,
            confidence_tiers=["A", "B", "C"],
        )

        result = engine.run(strategy, _make_historical_games())

        # Validate all fields are present
        assert isinstance(result.roi_pct, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.max_drawdown_units, float)
        assert isinstance(result.results_by_league, dict)
        assert isinstance(result.rejection_reasons, list)

    def test_draw_selection_grades_as_win_on_tie(self):
        """A draw moneyline bet WINS on a tie and is not a push (Gap 2)."""
        from omega.strategy.backtest.engine import BacktestEngine
        from omega.strategy.models import StrategyEntry

        engine = BacktestEngine(n_iterations=100)
        strategy = StrategyEntry(
            strategy_id="draw-grade",
            name="Draw Grade",
            leagues=["EPL"],
            edge_threshold=0.01,
            confidence_tiers=["A", "B", "C"],
        )
        tie = {"home_score": 1, "away_score": 1}

        # +200 → implied ~33.3%; model 50% clears the threshold.
        draw_bet = engine._evaluate_side(
            side="draw",
            team="Draw",
            model_prob=0.50,
            market_odds=200,
            strategy=strategy,
            outcome=tie,
            closing_odds=200,
        )
        assert draw_bet is not None
        assert draw_bet["won"] is True
        assert draw_bet["push"] is False
        assert draw_bet["net_units"] > 0

    def test_home_selection_pushes_on_tie(self):
        """A home moneyline bet still pushes (no win, no loss) on a tie (Gap 2)."""
        from omega.strategy.backtest.engine import BacktestEngine
        from omega.strategy.models import StrategyEntry

        engine = BacktestEngine(n_iterations=100)
        strategy = StrategyEntry(
            strategy_id="home-push",
            name="Home Push",
            leagues=["EPL"],
            edge_threshold=0.01,
            confidence_tiers=["A", "B", "C"],
        )
        tie = {"home_score": 1, "away_score": 1}

        home_bet = engine._evaluate_side(
            side="home",
            team="Arsenal",
            model_prob=0.50,
            market_odds=200,
            strategy=strategy,
            outcome=tie,
            closing_odds=200,
        )
        assert home_bet is not None
        assert home_bet["won"] is False
        assert home_bet["push"] is True
        assert home_bet["net_units"] == 0.0

    def test_exotic_grading_truth_table(self):
        """_grade_selection covers exotic soccer markets (Gap 5)."""
        from omega.strategy.backtest.engine import _grade_selection

        # Home win 2-1
        assert _grade_selection("double_chance", "home_draw", 2, 1) == (True, False)
        assert _grade_selection("double_chance", "home_away", 2, 1) == (True, False)
        assert _grade_selection("double_chance", "away_draw", 2, 1) == (False, False)
        # Draw 1-1
        assert _grade_selection("double_chance", "home_draw", 1, 1) == (True, False)
        assert _grade_selection("double_chance", "home_away", 1, 1) == (False, False)
        assert _grade_selection("double_chance", "away_draw", 1, 1) == (True, False)
        # Draw-no-bet voids (pushes) on a tie.
        assert _grade_selection("draw_no_bet", "home", 1, 1) == (False, True)
        assert _grade_selection("draw_no_bet", "home", 2, 1) == (True, False)
        assert _grade_selection("draw_no_bet", "away", 2, 1) == (False, False)
        # BTTS
        assert _grade_selection("both_teams_to_score", "yes", 1, 1) == (True, False)
        assert _grade_selection("both_teams_to_score", "yes", 1, 0) == (False, False)
        assert _grade_selection("both_teams_to_score", "no", 2, 0) == (True, False)
        # Correct score
        assert _grade_selection("correct_score", "2-1", 2, 1) == (True, False)
        assert _grade_selection("correct_score", "2-1", 1, 1) == (False, False)
        assert _grade_selection("correct_score", "bad", 1, 1) == (False, False)

    def test_backtest_passes_game_context_to_calibration(self, monkeypatch):
        import omega.strategy.backtest.engine as engine_mod
        from omega.strategy.artifacts import FrozenArtifact
        from omega.strategy.backtest.engine import BacktestEngine
        from omega.strategy.models import StrategyEntry

        seen_contexts = []

        def fake_apply_calibration(raw_prob, league=None, context_hints=None):
            seen_contexts.append(context_hints)
            return raw_prob

        monkeypatch.setattr(engine_mod, "apply_calibration", fake_apply_calibration)

        artifact = FrozenArtifact(
            artifact_id="ctx-test",
            home_team="Celtics",
            away_team="Pacers",
            league="NBA",
            date="2026-05-21",
            home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            away_context={"off_rating": 112.0, "def_rating": 112.0, "pace": 98.0},
            game_context={"is_playoff": True},
            odds={"moneyline_home": 300, "moneyline_away": 300},
            outcome={"home_score": 112, "away_score": 101},
            simulation_seed=42,
        )
        strategy = StrategyEntry(
            strategy_id="ctx-cal",
            name="Context Calibration",
            leagues=["NBA"],
            edge_threshold=0.01,
            confidence_tiers=["A", "B", "C"],
        )

        BacktestEngine(n_iterations=100, seed=7).run(strategy, [artifact])

        assert seen_contexts
        assert all(ctx == {"is_playoff": True} for ctx in seen_contexts)


# -----------------------------------------------------------------------
# Promotion tests
# -----------------------------------------------------------------------


class TestPromotion:
    """Test the promotion workflow."""

    def test_record_backtest_to_staging(self):
        from omega.strategy.models import BacktestResult, StrategyStatus

        registry = _make_registry()
        registry.register(strategy_id="promo-test", name="Promo Test")

        result = BacktestResult(
            strategy_id="promo-test",
            strategy_version=1,
            run_id="bt-test-001",
            started_at="2026-01-01T00:00:00Z",
            total_games=50,
            total_bets_placed=30,
            win_count=16,
            loss_count=14,
            win_rate=0.533,
            roi_pct=5.2,
            net_units=15.6,
            max_drawdown_units=8.0,
            avg_edge_pct=4.5,
            avg_closing_line_value=1.2,
            passed=True,
        )

        updated = registry.record_backtest("promo-test", 1, result)
        assert updated.status == StrategyStatus.STAGING

    def test_promote_to_production(self):
        from omega.strategy.models import BacktestResult, StrategyStatus

        registry = _make_registry()
        registry.register(strategy_id="promo-test", name="Promo Test")

        result = BacktestResult(
            strategy_id="promo-test",
            strategy_version=1,
            run_id="bt-test-001",
            started_at="2026-01-01T00:00:00Z",
            passed=True,
            total_bets_placed=30,
            win_rate=0.533,
            roi_pct=5.2,
        )
        registry.record_backtest("promo-test", 1, result)

        promoted = registry.promote("promo-test", 1, reason="Test promotion")
        assert promoted.status == StrategyStatus.PRODUCTION
        assert len(promoted.promotion_history) == 1

    def test_promote_archives_previous(self):
        from omega.strategy.models import BacktestResult, StrategyStatus

        registry = _make_registry()
        registry.register(strategy_id="evolve", name="v1")
        registry.register(strategy_id="evolve", name="v2")

        # Promote v1
        result_v1 = BacktestResult(
            strategy_id="evolve",
            strategy_version=1,
            run_id="bt-1",
            started_at="2026-01-01T00:00:00Z",
            passed=True,
            total_bets_placed=30,
            roi_pct=3.0,
        )
        registry.record_backtest("evolve", 1, result_v1)
        registry.promote("evolve", 1)

        # Promote v2 — should archive v1
        result_v2 = BacktestResult(
            strategy_id="evolve",
            strategy_version=2,
            run_id="bt-2",
            started_at="2026-02-01T00:00:00Z",
            passed=True,
            total_bets_placed=40,
            roi_pct=6.0,
        )
        registry.record_backtest("evolve", 2, result_v2)
        registry.promote("evolve", 2)

        v1 = registry.get("evolve", version=1)
        v2 = registry.get("evolve", version=2)
        assert v1.status == StrategyStatus.ARCHIVED
        assert v2.status == StrategyStatus.PRODUCTION

    def test_reject_strategy(self):
        from omega.strategy.models import StrategyStatus

        registry = _make_registry()
        registry.register(strategy_id="bad", name="Bad Strategy")

        rejected = registry.reject("bad", 1, reason="Terrible ROI")
        assert rejected.status == StrategyStatus.REJECTED
        assert len(rejected.promotion_history) == 1
        assert rejected.promotion_history[0].reason == "Terrible ROI"

    def test_cannot_promote_non_staging(self):
        registry = _make_registry()
        registry.register(strategy_id="raw", name="Raw")

        with pytest.raises(ValueError, match="must be 'staging'"):
            registry.promote("raw", 1)

    def test_auto_promote_or_reject(self):
        from omega.strategy.models import BacktestResult, StrategyStatus
        from omega.strategy.versioning.promotion import (
            PromotionCriteria,
            auto_promote_or_reject,
        )

        registry = _make_registry()
        registry.register(strategy_id="auto", name="Auto Test")

        # Good result — should auto-promote
        result = BacktestResult(
            strategy_id="auto",
            strategy_version=1,
            run_id="bt-auto",
            started_at="2026-01-01T00:00:00Z",
            passed=True,
            total_bets_placed=50,
            win_count=28,
            loss_count=22,
            win_rate=0.56,
            roi_pct=8.0,
            net_units=40.0,
            max_drawdown_units=10.0,
            avg_edge_pct=5.0,
            avg_closing_line_value=2.0,
        )

        entry = auto_promote_or_reject(
            registry,
            "auto",
            1,
            result,
            criteria=PromotionCriteria(min_roi_pct=2.0),
        )
        assert entry.status == StrategyStatus.PRODUCTION

    def test_auto_reject_low_roi(self):
        from omega.strategy.models import BacktestResult, StrategyStatus
        from omega.strategy.versioning.promotion import (
            PromotionCriteria,
            auto_promote_or_reject,
        )

        registry = _make_registry()
        registry.register(strategy_id="bad-auto", name="Bad Auto")

        result = BacktestResult(
            strategy_id="bad-auto",
            strategy_version=1,
            run_id="bt-bad",
            started_at="2026-01-01T00:00:00Z",
            passed=True,
            total_bets_placed=50,
            win_rate=0.50,
            roi_pct=0.5,
            net_units=2.5,
            max_drawdown_units=5.0,
            avg_closing_line_value=0.5,
        )

        entry = auto_promote_or_reject(
            registry,
            "bad-auto",
            1,
            result,
            criteria=PromotionCriteria(min_roi_pct=3.0),
        )
        assert entry.status == StrategyStatus.REJECTED


class TestPromotionCriteria:
    """Test the evaluation function directly."""

    def test_passing_criteria(self):
        from omega.strategy.models import BacktestResult
        from omega.strategy.versioning.promotion import evaluate_for_promotion

        result = BacktestResult(
            strategy_id="x",
            strategy_version=1,
            run_id="r",
            started_at="t",
            passed=True,
            total_bets_placed=50,
            win_rate=0.55,
            roi_pct=6.0,
            max_drawdown_units=8.0,
            avg_closing_line_value=1.5,
        )

        should, reasons = evaluate_for_promotion(result)
        assert should is True
        assert len(reasons) == 0

    def test_failing_multiple_criteria(self):
        from omega.strategy.models import BacktestResult
        from omega.strategy.versioning.promotion import evaluate_for_promotion

        result = BacktestResult(
            strategy_id="x",
            strategy_version=1,
            run_id="r",
            started_at="t",
            passed=True,
            total_bets_placed=10,  # too few
            win_rate=0.30,  # too low
            roi_pct=-5.0,  # negative
            max_drawdown_units=25.0,  # too high
            avg_closing_line_value=-3.0,  # negative CLV
        )

        should, reasons = evaluate_for_promotion(result)
        assert should is False
        assert len(reasons) >= 4


# -----------------------------------------------------------------------
# File persistence test
# -----------------------------------------------------------------------


class TestRegistryPersistence:
    """Test JSON file persistence."""

    def test_save_and_load(self, tmp_path):
        from omega.strategy.versioning.registry import StrategyRegistry

        path = str(tmp_path / "strategies.json")

        # Save
        reg1 = StrategyRegistry(storage_path=path)
        reg1.register(strategy_id="persist-test", name="Persistent")

        # Load in new instance
        reg2 = StrategyRegistry(storage_path=path)
        entry = reg2.get("persist-test")
        assert entry is not None
        assert entry.name == "Persistent"
