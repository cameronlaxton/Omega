"""
Tests for Omega core runtime: schemas, simulation, betting, config.

Tests the deterministic core — no network, no LLM.
"""

_GAME_CONTEXT = {"is_playoff": False, "rest_days": 2}


class TestModuleImports:
    """Verify core modules import without error."""

    def test_contracts_schemas(self):
        pass

    def test_league_config(self):
        pass

    def test_simulation_modules(self):
        pass

    def test_betting_modules(self):
        pass

    def test_calibration_modules(self):
        pass

    def test_archetypes(self):
        from omega.core.simulation.archetypes import (
            ARCHETYPE_REGISTRY,
            LEAGUE_TO_ARCHETYPE,
        )

        assert len(ARCHETYPE_REGISTRY) == 9
        assert len(LEAGUE_TO_ARCHETYPE) > 50


class TestContractSchemas:
    """Test Pydantic schemas."""

    def test_game_analysis_request(self):
        from omega.core.contracts.schemas import GameAnalysisRequest, OddsInput

        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            odds=OddsInput(
                moneyline_home=-150,
                moneyline_away=130,
                spread_home=-4.5,
                over_under=220.5,
            ),
            n_iterations=1000,
            game_context=_GAME_CONTEXT,
        )
        assert req.home_team == "Boston Celtics"
        assert req.odds.moneyline_home == -150
        assert req.odds.spread_home == -4.5

    def test_market_quote(self):
        from omega.core.contracts.schemas import MarketQuote

        quote = MarketQuote(
            market_type="moneyline",
            selection="Home",
            price=-150,
        )
        assert quote.market_type == "moneyline"
        assert quote.price == -150

    def test_odds_input_with_markets(self):
        from omega.core.contracts.schemas import MarketQuote, OddsInput

        odds = OddsInput(
            markets=[
                MarketQuote(market_type="moneyline", selection="Home", price=-150),
                MarketQuote(market_type="spread", selection="Home -4.5", price=-110, line=-4.5),
            ]
        )
        assert len(odds.markets) == 2
        assert odds.markets[1].line == -4.5


class TestSimulation:
    """Test the simulation engine."""

    def test_fast_game_simulation_with_context(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        result = engine.run_fast_game_simulation(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            away_context={"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
        )
        assert result["success"] is True
        assert result["context_source"] == "provided"
        assert result["baseline_used"] is False
        assert 0 <= result["home_win_prob"] <= 100
        assert 0 <= result["away_win_prob"] <= 100
        assert result["iterations"] == 100

    def test_allowed_rate_defense_is_monotonic_by_sport(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        cases = [
            (
                "NBA",
                {"off_rating": 114.0, "def_rating": 108.0, "pace": 100.0},
                {"off_rating": 112.0, "def_rating": 100.0, "pace": 100.0},
                {"off_rating": 112.0, "def_rating": 122.0, "pace": 100.0},
            ),
            (
                "MLB",
                {"off_rating": 4.6, "def_rating": 4.0},
                {"off_rating": 4.0, "def_rating": 3.0},
                {"off_rating": 4.0, "def_rating": 5.4},
            ),
            (
                "NHL",
                {"off_rating": 3.1, "def_rating": 3.0},
                {"off_rating": 3.0, "def_rating": 2.4},
                {"off_rating": 3.0, "def_rating": 3.8},
            ),
            (
                "EPL",
                {"off_rating": 1.8, "def_rating": 1.1},
                {"off_rating": 1.4, "def_rating": 0.7},
                {"off_rating": 1.4, "def_rating": 1.6},
            ),
        ]
        for league, home, strong_def, weak_def in cases:
            strong = engine.run_fast_game_simulation(
                "Home",
                "Away",
                league=league,
                n_iterations=4000,
                seed=123,
                home_context=home,
                away_context=strong_def,
            )
            weak = engine.run_fast_game_simulation(
                "Home",
                "Away",
                league=league,
                n_iterations=4000,
                seed=123,
                home_context=home,
                away_context=weak_def,
            )
            assert weak["predicted_home_score"] > strong["predicted_home_score"], league

    def test_no_draw_sports_resolve_ties_into_two_way_probability(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        for league, home, away in [
            (
                "NBA",
                {"off_rating": 110.0, "def_rating": 110.0, "pace": 100.0},
                {"off_rating": 110.0, "def_rating": 110.0, "pace": 100.0},
            ),
            ("MLB", {"off_rating": 4.2, "def_rating": 4.2}, {"off_rating": 4.2, "def_rating": 4.2}),
        ]:
            result = engine.run_fast_game_simulation(
                "Home",
                "Away",
                league=league,
                n_iterations=2000,
                seed=321,
                home_context=home,
                away_context=away,
            )
            assert result["draw_prob"] == 0.0
            assert abs(result["home_win_prob"] + result["away_win_prob"] - 100.0) <= 0.1

    def test_fast_game_simulation_skips_when_context_absent_by_default(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        result = engine.run_fast_game_simulation(
            home_team="Team A",
            away_team="Team B",
            league="NBA",
            n_iterations=100,
        )
        assert result.get("success") is False
        assert result["context_source"] == "missing"
        assert "allow_baseline=True" in result["skip_reason"]

    def test_fast_game_simulation_allows_explicit_baseline(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        result = engine.run_fast_game_simulation(
            home_team="Team A",
            away_team="Team B",
            league="NBA",
            n_iterations=100,
            allow_baseline=True,
        )
        assert result.get("success") is True
        assert result["context_source"] == "league_default"
        assert result["baseline_used"] is True
        assert 0 <= result["home_win_prob"] <= 100
        assert result["simulation_distributions"]

    def test_unknown_league_skips(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        result = engine.run_fast_game_simulation(
            home_team="A",
            away_team="B",
            league="QUIDDITCH",
            n_iterations=10,
        )
        assert result["success"] is False
        assert "No simulation model" in result["skip_reason"]

    def test_soccer_poisson(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        result = engine.run_fast_game_simulation(
            home_team="Liverpool",
            away_team="Man City",
            league="EPL",
            n_iterations=200,
            home_context={"off_rating": 1.8, "def_rating": 0.9},
            away_context={"off_rating": 2.1, "def_rating": 0.7},
        )
        assert result["success"] is True
        assert result.get("draw_prob", 0) > 0  # Soccer should have draws

    def test_soccer_dixon_coles_is_seed_reproducible(self):
        """DC-enabled soccer sim with a fixed seed reruns bit-identically."""
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        kwargs = dict(
            home_team="Liverpool",
            away_team="Man City",
            league="EPL",  # dixon_coles=True in league config
            n_iterations=500,
            home_context={"off_rating": 1.8, "def_rating": 0.9},
            away_context={"off_rating": 2.1, "def_rating": 0.7},
            seed=12345,
        )
        r1 = engine.run_fast_game_simulation(**kwargs)
        r2 = engine.run_fast_game_simulation(**kwargs)
        assert r1["success"] and r2["success"]
        assert r1["draw_prob"] == r2["draw_prob"]
        assert r1["home_win_prob"] == r2["home_win_prob"]
        assert r1["predicted_home_score"] == r2["predicted_home_score"]

    def test_soccer_exposes_exotic_market_probs(self):
        """Soccer sim exposes double-chance / DNB / BTTS / correct-score probs
        that are internally consistent; non-draw sports expose none (Gap 5)."""
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        r = engine.run_fast_game_simulation(
            home_team="Arsenal",
            away_team="Chelsea",
            league="EPL",
            n_iterations=4000,
            home_context={"off_rating": 1.8, "def_rating": 1.0},
            away_context={"off_rating": 1.4, "def_rating": 1.1},
            seed=3,
        )
        # Double chance 1X = home_win + draw.
        home_win = r["home_win_prob"]
        assert abs(r["double_chance_home_draw_prob"] - (home_win + r["draw_prob"])) < 0.2
        # DNB conditional on a decisive result.
        assert abs((r["dnb_home_prob"] + r["dnb_away_prob"]) - 100.0) < 0.2
        # BTTS yes+no ~ 100.
        assert abs((r["btts_yes_prob"] + r["btts_no_prob"]) - 100.0) < 0.2
        assert r["correct_score_probs"]  # non-empty map

        # NBA (non-draw) carries no exotic fields.
        rn = engine.run_fast_game_simulation(
            home_team="A",
            away_team="B",
            league="NBA",
            n_iterations=300,
            home_context={"off_rating": 115, "def_rating": 110, "pace": 100},
            away_context={"off_rating": 112, "def_rating": 112, "pace": 99},
            seed=3,
        )
        assert "btts_yes_prob" not in rn
        assert "correct_score_probs" not in rn

    def test_dixon_coles_increases_draw_mass(self):
        """Negative rho shifts mass onto 0-0/1-1, raising the draw rate vs rho=0
        (independent Poisson)."""
        import numpy as np

        from omega.core.simulation.engine import _dixon_coles_scores

        def draw_rate(rho):
            np.random.seed(7)
            h, a = _dixon_coles_scores(1.3, 1.3, rho, 20000)
            return sum(1 for x, y in zip(h, a) if x == y) / len(h)

        assert draw_rate(-0.13) > draw_rate(0.0)

    def test_ufc_fighting(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        engine = OmegaSimulationEngine()
        result = engine.run_fast_game_simulation(
            home_team="Fighter A",
            away_team="Fighter B",
            league="UFC",
            n_iterations=200,
            home_context={"win_pct": 0.7, "finish_rate": 0.6},
            away_context={"win_pct": 0.5, "finish_rate": 0.4},
        )
        assert result["success"] is True


class TestBettingAnalysis:
    """Test betting math."""

    def test_implied_probability(self):
        from omega.core.betting.odds import implied_probability

        prob = implied_probability(-150)
        assert 0.59 < prob < 0.61  # -150 implies ~60%

    def test_edge_percentage(self):
        from omega.core.betting.odds import edge_percentage, implied_probability

        model_prob = 0.65
        impl_prob = implied_probability(-150)
        edge = edge_percentage(model_prob, impl_prob)
        assert edge > 0  # Model says 65%, market implies ~60%

    def test_expected_value(self):
        from omega.core.betting.odds import expected_value_percent

        ev = expected_value_percent(0.65, -150)
        assert isinstance(ev, (int, float))

    def test_kelly_staking(self):
        from omega.core.betting.kelly import recommend_stake

        stake = recommend_stake(
            true_prob=0.65,
            odds=-150,
            bankroll=1000.0,
            confidence_tier="B",
        )
        assert isinstance(stake, dict)
        assert stake["units"] >= 0
        assert stake["kelly_fraction"] >= 0


class TestLeagueConfig:
    """Test league configuration."""

    def test_nba_config(self):
        from omega.core.config.leagues import get_league_config

        nba = get_league_config("NBA")
        assert nba["periods"] == 4
        assert nba["archetype"] == "basketball"

    def test_nfl_config(self):
        from omega.core.config.leagues import get_league_config

        nfl = get_league_config("NFL")
        assert nfl["archetype"] == "american_football"

    def test_unknown_league_returns_defaults(self):
        from omega.core.config.leagues import get_league_config

        cfg = get_league_config("QUIDDITCH")
        assert isinstance(cfg, dict)


class TestCalibration:
    """Test probability calibration."""

    def test_shrinkage_calibration(self):
        from omega.core.calibration.probability import shrinkage_calibration

        # Extreme probability should shrink toward 0.5
        calibrated = shrinkage_calibration(0.95, shrink_factor=0.7)
        assert 0.5 < calibrated < 0.95

    def test_calibrate_probability(self):
        from omega.core.calibration.probability import calibrate_probability

        result = calibrate_probability(0.85, method="shrinkage")
        assert isinstance(result, dict)
        assert 0 < result["calibrated"] < 1

    def test_apply_calibration_shared_policy(self):
        """apply_calibration is the single source of truth for both paths."""
        from omega.core.calibration.probability import apply_calibration

        # Extreme probability should be calibrated
        cal = apply_calibration(0.95)
        assert 0.5 < cal < 0.95, "Extreme prob must be shrunk"

        # Mild probability (within gate) should pass through unchanged
        cal_mild = apply_calibration(0.65)
        assert cal_mild == 0.65, "Mild prob must not be touched by gate"

    def test_calibration_parity_service_and_backtest(self):
        """Production service and backtest engine use the same calibration."""
        from omega.core.calibration.probability import apply_calibration
        from omega.core.contracts.service import _calibrate

        test_probs = [0.05, 0.15, 0.50, 0.65, 0.85, 0.92, 0.99]
        for p in test_probs:
            assert apply_calibration(p) == _calibrate(p), (
                f"Parity violation at raw_prob={p}: "
                f"apply_calibration={apply_calibration(p)}, "
                f"_calibrate={_calibrate(p)}"
            )

    def test_game_analysis_consumes_normalized_market_quotes(self):
        from omega.core.contracts.schemas import GameAnalysisRequest, MarketQuote, OddsInput
        from omega.core.contracts.service import analyze_game

        req = GameAnalysisRequest(
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            league="NBA",
            n_iterations=1000,
            seed=42,
            home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            away_context={"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
            game_context=_GAME_CONTEXT,
            odds=OddsInput(
                markets=[
                    MarketQuote(
                        market_type="moneyline",
                        selection="Los Angeles Lakers",
                        price=-150,
                        bookmaker="betmgm",
                    ),
                    MarketQuote(
                        market_type="moneyline",
                        selection="Boston Celtics",
                        price=130,
                        bookmaker="betmgm",
                    ),
                ]
            ),
        )

        result = analyze_game(req)

        assert result.status == "success"
        assert [edge.market_odds for edge in result.edges] == [-150.0, 130.0]

    def test_slate_analysis_preserves_normalized_markets(self):
        from omega.core.contracts.schemas import SlateAnalysisRequest
        from omega.core.contracts.service import analyze_slate

        result = analyze_slate(
            SlateAnalysisRequest(
                league="NBA",
                games=[
                    {
                        "home_team": "Los Angeles Lakers",
                        "away_team": "Boston Celtics",
                        "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
                        "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
                        "game_context": _GAME_CONTEXT,
                        "odds": {
                            "markets": [
                                {
                                    "market_type": "moneyline",
                                    "selection": "Los Angeles Lakers",
                                    "price": -150,
                                    "bookmaker": "betmgm",
                                },
                                {
                                    "market_type": "moneyline",
                                    "selection": "Boston Celtics",
                                    "price": 130,
                                    "bookmaker": "betmgm",
                                },
                            ]
                        },
                    }
                ],
            )
        )

        assert result.analyses[0].edges

    def test_slate_analysis_preserves_per_game_seed_and_iterations(self):
        from omega.core.contracts.schemas import SlateAnalysisRequest
        from omega.core.contracts.service import analyze_slate

        request = SlateAnalysisRequest(
            league="NBA",
            games=[
                {
                    "home_team": "Los Angeles Lakers",
                    "away_team": "Boston Celtics",
                    "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
                    "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
                    "n_iterations": 100,
                    "seed": 123,
                    "game_context": _GAME_CONTEXT,
                }
            ],
        )

        first = analyze_slate(request)
        second = analyze_slate(request)

        assert first.analyses[0].simulation == second.analyses[0].simulation
        assert first.analyses[0].simulation.iterations == 100


class TestLegacyMarkovMethods:
    def test_run_game_simulation_uses_markov_module(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        result = OmegaSimulationEngine().run_game_simulation(
            home_team="Home",
            away_team="Away",
            league="NBA",
            home_context={"off_rating": 110, "def_rating": 105, "pace": 99},
            away_context={"off_rating": 108, "def_rating": 107, "pace": 98},
        )

        assert result["success"] is True
        assert result["simulation_backend"] == "markov_state"
        assert result["context_source"] == "provided"
        assert result["simulation_distributions"]

    def test_run_player_prop_simulation_uses_markov_module(self):
        from omega.core.simulation.engine import OmegaSimulationEngine

        result = OmegaSimulationEngine().run_player_prop_simulation(
            player_name="Test Player",
            team="Home",
            opponent="Away",
            league="NBA",
            prop_type="pts",
            game_context={"home_context": {}, "away_context": {}, "home_players": []},
            player_context={"pts_mean": 20.0},
        )

        assert result["success"] is True
        assert result["projected_value"] > 0
        assert 0 <= result["over_prob"] <= 100
