"""
Tests for omega.core.contracts.service — the primary LLM↔engine integration seam.

Covers:
  - _pick_best_bet(): always returns BetSlip (not None) for actionable edges
  - analyze_game(): happy path, no-odds path, skipped path
  - analyze_player_prop(): missing-mean skip, success path
  - _resolve_game_market_odds(): normalized markets vs. legacy fields
"""

from __future__ import annotations

import pytest

from omega.core.contracts.evidence import EvidenceSignal
from omega.core.contracts.schemas import (
    BetSlip,
    EdgeDetail,
    GameAnalysisRequest,
    MarketQuote,
    OddsInput,
    PlayerPropRequest,
)
from omega.core.contracts.service import (
    _apply_game_context,
    _game_evidence_plan_for,
    _pick_best_bet,
    _player_evidence_plan_for,
    _resolve_game_market_odds,
    _stable_input_hash,
    analyze,
    analyze_game,
    analyze_player_prop,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _edge(side="home", team="Lakers", edge_pct=8.0, ev_pct=5.0, tier="A", odds=-130.0):
    return EdgeDetail(
        side=side,
        team=team,
        true_prob=0.60,
        calibrated_prob=0.58,
        market_implied=0.45,
        edge_pct=edge_pct,
        ev_pct=ev_pct,
        market_odds=odds,
        confidence_tier=tier,
    )


_NBA_HOME_CTX = {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0}
_NBA_AWAY_CTX = {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0}
_GAME_CONTEXT = {"is_playoff": False, "rest_days": 2}
_B2B_GAME_CONTEXT = {"is_playoff": False, "rest_days": 0}


# ---------------------------------------------------------------------------
# _pick_best_bet
# ---------------------------------------------------------------------------


class TestPickBestBet:
    def test_returns_betslip_for_actionable_edge(self):
        result = _pick_best_bet([_edge(tier="A")], bankroll=1000.0)
        assert isinstance(result, BetSlip)

    def test_returns_betslip_for_tier_b(self):
        result = _pick_best_bet([_edge(tier="B")], bankroll=1000.0)
        assert isinstance(result, BetSlip)

    def test_returns_none_when_all_pass(self):
        result = _pick_best_bet(
            [_edge(tier="Pass"), _edge(tier="Pass", side="away")], bankroll=1000.0
        )
        assert result is None

    def test_returns_none_for_empty_edges(self):
        assert _pick_best_bet([], bankroll=1000.0) is None

    def test_selects_highest_positive_ev_edge(self):
        weak = _edge(team="TeamA", edge_pct=4.0, ev_pct=2.5, tier="B")
        strong = _edge(team="TeamB", edge_pct=15.0, ev_pct=9.0, tier="A", side="away")
        result = _pick_best_bet([weak, strong], bankroll=1000.0)
        assert result is not None
        assert "TeamB" in result.selection

    def test_betslip_fields_are_populated(self):
        e = _edge(edge_pct=10.0, ev_pct=6.0, tier="A", odds=-120.0)
        result = _pick_best_bet([e], bankroll=1000.0)
        assert result is not None
        assert result.edge_pct == pytest.approx(10.0)
        assert result.ev_pct == pytest.approx(6.0)
        assert result.confidence_tier == "A"
        assert result.odds == pytest.approx(-120.0)
        assert isinstance(result.recommended_units, float)
        assert isinstance(result.kelly_fraction, float)
        assert result.kelly_fraction >= 0.0

    # Issue #3 regression: best_bet must never select a negative-EV side
    def test_ignores_negative_ev_side(self):
        neg = _edge(team="HomeTeam", side="home", edge_pct=-15.27, ev_pct=-33.29, tier="A")
        pos = _edge(team="AwayTeam", side="away", edge_pct=8.0, ev_pct=5.0, tier="A")
        result = _pick_best_bet([neg, pos], bankroll=1000.0)
        assert result is not None
        assert "AwayTeam" in result.selection

    def test_returns_none_when_all_negative_ev(self):
        neg_a = _edge(team="HomeTeam", edge_pct=-10.0, ev_pct=-20.0, tier="A")
        neg_b = _edge(team="AwayTeam", side="away", edge_pct=-5.0, ev_pct=-8.0, tier="B")
        assert _pick_best_bet([neg_a, neg_b], bankroll=1000.0) is None


# ---------------------------------------------------------------------------
# analyze_game
# ---------------------------------------------------------------------------


class TestAnalyzeGame:
    def test_returns_success_with_valid_context(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_game(req)
        assert resp.status == "success"
        assert resp.simulation is not None
        assert 0 <= resp.simulation.home_win_prob <= 100
        assert 0 <= resp.simulation.away_win_prob <= 100

    def test_best_bet_is_none_without_odds(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_game(req)
        assert resp.best_bet is None
        assert resp.edges == []

    def test_best_bet_returned_when_large_market_edge(self):
        # Supply only home moneyline so only one EdgeDetail is produced.
        # Simulation puts home at ~56% while market implies ~25% (+300) —
        # a ~31% edge that is guaranteed to produce an A-tier BetSlip.
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=1000,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
            odds=OddsInput(moneyline_home=300),
        )
        resp = analyze_game(req)
        assert resp.status == "success"
        assert resp.best_bet is not None
        assert isinstance(resp.best_bet, BetSlip)
        assert resp.best_bet.edge_pct > 3.0

    def test_edges_populated_when_odds_supplied(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
            odds=OddsInput(moneyline_home=-150, moneyline_away=130),
        )
        resp = analyze_game(req)
        assert len(resp.edges) == 2
        assert all(isinstance(e, EdgeDetail) for e in resp.edges)

    def test_skips_when_context_absent_by_default(self):
        req = GameAnalysisRequest(
            home_team="Team A",
            away_team="Team B",
            league="NBA",
            n_iterations=100,
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_game(req)
        assert resp.status == "skipped"
        assert resp.simulation is None
        assert resp.context_source is None

    def test_succeeds_with_explicit_baseline_but_marks_provenance(self):
        req = GameAnalysisRequest(
            home_team="Team A",
            away_team="Team B",
            league="NBA",
            n_iterations=100,
            allow_baseline=True,
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_game(req)
        assert resp.status == "success"
        assert resp.simulation is not None
        assert resp.context_source == "league_default"
        assert resp.baseline_used is True
        assert 0 < resp.simulation.home_win_prob < 100

    def test_markov_backend_can_be_selected_explicitly(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            seed=123,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context={"is_playoff": False, "rest_days": 2},
            simulation_backend="markov_state",
        )

        resp = analyze_game(req)

        assert resp.status == "success"
        assert resp.simulation_backend == "markov_state"
        assert resp.component_version == "markov_state_v1"
        assert resp.simulation_distributions
        assert resp.simulation_distributions[0]["distribution_type"] == "empirical_markov"

    def test_unsupported_simulation_backend_skips(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context={"is_playoff": False, "rest_days": 2},
            simulation_backend="unknown_backend",
        )

        resp = analyze_game(req)

        assert resp.status == "skipped"
        assert "simulation_backend" in resp.missing_requirements

    def test_never_raises(self):
        # Garbage input should return error/skipped, not raise
        req = GameAnalysisRequest(
            home_team="",
            away_team="",
            league="NBA",
            n_iterations=100,
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_game(req)
        assert resp.status in ("success", "skipped", "error")

    def test_response_matchup_format(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_game(req)
        assert resp.matchup == "Indiana Pacers @ Boston Celtics"
        assert resp.league == "NBA"


class TestAnalyzeTraceEnvelope:
    def test_analyze_returns_core_trace_envelope(self):
        trace = analyze(
            {
                "home_team": "Boston Celtics",
                "away_team": "Indiana Pacers",
                "league": "NBA",
                "n_iterations": 100,
                "seed": 42,
                "home_context": _NBA_HOME_CTX,
                "away_context": _NBA_AWAY_CTX,
                "game_context": _GAME_CONTEXT,
                "odds": {"moneyline_home": -150, "moneyline_away": 130},
            },
            session_id="sess-20260518-core",
            bankroll=2500.0,
        )

        assert trace["trace_id"].startswith("sandbox-")
        assert trace["model_version"] == "omega-core-phase6h"
        assert trace["kind"] == "game"
        assert trace["session_id"] == "sess-20260518-core"
        assert trace["bankroll"] == 2500.0
        assert trace["input_snapshot"]["seed"] == 42
        assert trace["result"]["status"] == "success"
        assert trace["trace_quality"]["calibration_eligible"] is True
        assert trace["trace_quality"]["context_source"] == "provided"

    def test_analyze_requires_explicit_session_id_and_bankroll(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
        )

        with pytest.raises(ValueError):
            analyze(req, session_id="", bankroll=1000.0)
        with pytest.raises(ValueError):
            analyze(req, session_id="sess-20260518-core", bankroll=0.0)

    def test_analyze_marks_baseline_trace_ineligible(self):
        trace = analyze(
            {
                "home_team": "Team A",
                "away_team": "Team B",
                "league": "NBA",
                "n_iterations": 100,
                "allow_baseline": True,
                "game_context": _GAME_CONTEXT,
            },
            session_id="sess-baseline",
            bankroll=1000.0,
        )
        assert trace["result"]["status"] == "success"
        assert trace["result"]["context_source"] == "league_default"
        assert trace["trace_quality"]["calibration_eligible"] is False
        assert "baseline_default_context" in trace["trace_quality"]["calibration_exclusion_reasons"]

    def test_analyze_skips_invalid_basketball_fractional_ratings_before_dispatch(self):
        trace = analyze(
            {
                "home_team": "Dallas Wings",
                "away_team": "Las Vegas Aces",
                "league": "WNBA",
                "n_iterations": 100,
                "seed": 42,
                "home_context": {"off_rating": 0.462, "def_rating": 0.504, "pace": 69.6},
                "away_context": {"off_rating": 0.504, "def_rating": 0.462, "pace": 69.8},
                "game_context": {"is_playoff": False, "rest_days": 1},
                "odds": {"moneyline_home": 154, "moneyline_away": -185},
            },
            session_id="sess-invalid-context",
            bankroll=1000.0,
        )

        assert trace["result"]["status"] == "skipped"
        assert "Invalid simulation context" in trace["result"]["skip_reason"]
        assert trace["trace_quality"]["calibration_eligible"] is False
        assert "engine_skipped" in trace["trace_quality"]["calibration_exclusion_reasons"]

    def test_analyze_accepts_structured_mlb_context_fields(self):
        trace = analyze(
            {
                "home_team": "Texas Rangers",
                "away_team": "Houston Astros",
                "league": "MLB",
                "n_iterations": 200,
                "seed": 42,
                "home_context": {
                    "off_rating": 4.5,
                    "def_rating": 4.2,
                    "starter_era": 3.45,
                    "park_factor": 1.02,
                    "weather_wind_mph": 8.0,
                },
                "away_context": {
                    "off_rating": 4.1,
                    "def_rating": 4.4,
                    "starter_era": 4.10,
                },
                "game_context": {"is_playoff": False, "rest_days": 4},
                "odds": {"moneyline_home": -120, "moneyline_away": 105},
            },
            session_id="sess-mlb-context",
            bankroll=1000.0,
        )

        assert trace["result"]["status"] == "success"
        assert trace["input_snapshot"]["home_context"]["starter_era"] == pytest.approx(3.45)
        assert trace["input_snapshot"]["home_context"]["weather_wind_mph"] == pytest.approx(8.0)

    def test_markov_live_evidence_uses_transition_path_only(self, monkeypatch):
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "live")
        signal = EvidenceSignal(
            signal_type="b2b_fatigue",
            category="situational",
            plane="game",
            value=True,
            source="agent_reasoning",
            confidence=0.8,
            window="matchup",
            direction="home",
        )
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context={"is_playoff": True, "rest_days": 0},
            simulation_backend="markov_state",
            evidence=[signal],
        )

        plan = _game_evidence_plan_for(req)
        trace = analyze(req, session_id="sess-markov-evidence", bankroll=1000.0)

        assert plan.adjustment is None
        assert plan.transition_modifiers == {"home_score_rate_scalar": pytest.approx(0.94)}
        app = trace["evidence_application"][0]
        assert app["target"] == "markov_transition"
        assert app["applied"] is True
        assert app["reason"] == "mapped_to_markov_transition_modifiers"
        params = trace["result"]["simulation_distributions"][0]["distribution_params"]
        assert params["transition_modifiers"]["home_score_rate_scalar"] == pytest.approx(0.94)

    def test_markov_dual_plane_duplicate_suppresses_player_plane(self, monkeypatch):
        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "live")
        game_signal = EvidenceSignal(
            signal_type="b2b_fatigue",
            category="situational",
            plane="game",
            value=True,
            source="agent_reasoning",
            confidence=0.8,
            window="matchup",
            direction="home",
        )
        player_signal = game_signal.model_copy(update={"plane": "player"})
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=100,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context={"is_playoff": True, "rest_days": 0},
            simulation_backend="markov_state",
            evidence=[game_signal, player_signal],
        )

        trace = analyze(req, session_id="sess-markov-dedup", bankroll=1000.0)

        apps = trace["evidence_application"]
        assert apps[0]["target"] == "markov_transition"
        assert apps[0]["applied"] is True
        assert apps[1]["target"] == "skip"
        assert apps[1]["applied"] is False
        assert apps[1]["reason"] == "suppressed_by_game_plane_dedup"
        params = trace["result"]["simulation_distributions"][0]["distribution_params"]
        assert params["transition_modifiers"]["home_score_rate_scalar"] == pytest.approx(0.94)

    def test_prop_dual_plane_duplicate_suppresses_player_in_shadow_and_live(self, monkeypatch):
        game_signal = EvidenceSignal(
            signal_type="b2b_fatigue",
            category="situational",
            plane="game",
            value=True,
            source="agent_reasoning",
            confidence=0.8,
            window="matchup",
            direction="home",
        )
        player_signal = game_signal.model_copy(update={"plane": "player"})
        req = PlayerPropRequest(
            player_name="Test Player",
            league="NBA",
            prop_type="pts",
            line=20.5,
            odds_over=-110,
            odds_under=-110,
            player_context={"pts_mean": 24.0, "pts_std": 5.0},
            game_context={"is_playoff": False, "rest_days": 0},
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            game_date="2026-05-25",
            n_iterations=1000,
            seed=77,
            evidence=[game_signal, player_signal],
        )

        monkeypatch.delenv("OMEGA_EVIDENCE_MODE", raising=False)
        shadow_plan = _player_evidence_plan_for(req)
        assert shadow_plan.evidence_application[1]["reason"] == "suppressed_by_game_plane_dedup"
        assert shadow_plan.evidence_application[1]["applied"] is False

        monkeypatch.setenv("OMEGA_EVIDENCE_MODE", "live")
        live_plan = _player_evidence_plan_for(req)
        assert live_plan.adjustment is not None
        assert live_plan.adjustment.mean_factor == 1.0
        assert live_plan.evidence_application[1]["reason"] == "suppressed_by_game_plane_dedup"
        assert live_plan.evidence_application[1]["applied"] is False

    def test_stable_hash_excludes_volatile_odds_structures(self):
        base = {
            "home_team": "Boston Celtics",
            "away_team": "Indiana Pacers",
            "league": "NBA",
            "seed": 42,
            "home_context": _NBA_HOME_CTX,
            "away_context": _NBA_AWAY_CTX,
            "odds": {"moneyline_home": -150, "markets": [{"price": -150}]},
            "market_snapshots": [{"book": "betmgm", "price": -150}],
        }
        moved_market = {
            **base,
            "odds": {"moneyline_home": -180, "markets": [{"price": -180}]},
            "market_snapshots": [{"book": "betmgm", "price": -180}],
        }
        changed_context = {
            **base,
            "home_context": {**_NBA_HOME_CTX, "pace": 104.0},
        }

        assert _stable_input_hash(base) == _stable_input_hash(moved_market)
        assert _stable_input_hash(base) != _stable_input_hash(changed_context)


# ---------------------------------------------------------------------------
# analyze_player_prop
# ---------------------------------------------------------------------------


class TestAnalyzePlayerProp:
    def test_skipped_when_missing_mean(self):
        req = PlayerPropRequest(
            player_name="LeBron James",
            league="NBA",
            prop_type="pts",
            line=25.5,
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            game_date="2026-05-17",
            game_context=_GAME_CONTEXT,
            player_context={},
        )
        resp = analyze_player_prop(req)
        assert resp.status == "skipped"
        assert "pts_mean" in (resp.skip_reason or "")
        assert resp.missing_requirements is not None

    def test_success_with_valid_context(self):
        req = PlayerPropRequest(
            player_name="LeBron James",
            league="NBA",
            prop_type="pts",
            line=25.5,
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            game_date="2026-05-17",
            odds_over=-110,
            odds_under=-110,
            n_iterations=500,
            seed=42,
            player_context={"pts_mean": 27.0, "pts_std": 5.5},
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_player_prop(req)
        assert resp.status == "success"
        assert 0 < resp.over_prob < 1
        assert 0 < resp.under_prob < 1
        assert resp.over_prob + resp.under_prob == pytest.approx(1.0, abs=0.01)
        assert resp.projection_mean is not None
        assert resp.projection_std is not None
        assert resp.distribution_type in {"normal", "poisson"}
        assert resp.simulation_distributions

    def test_recommended_prop_populates_deterministic_stake_fields(self):
        req = PlayerPropRequest(
            player_name="LeBron James",
            league="NBA",
            prop_type="pts",
            line=22.5,
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            game_date="2026-05-17",
            odds_over=150,
            odds_under=-110,
            n_iterations=1000,
            seed=42,
            player_context={"pts_mean": 32.0, "pts_std": 5.0},
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_player_prop(req, bankroll=1000.0)

        assert resp.status == "success"
        assert resp.recommendation == "over"
        assert resp.bet_side_odds == pytest.approx(150.0)
        assert isinstance(resp.kelly_fraction, float)
        assert isinstance(resp.recommended_units, float)
        assert resp.recommended_units > 0.0

    def test_prop_pass_leaves_stake_fields_null(self):
        req = PlayerPropRequest(
            player_name="LeBron James",
            league="NBA",
            prop_type="pts",
            line=25.5,
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            game_date="2026-05-17",
            odds_over=-1000,
            odds_under=-1000,
            n_iterations=1000,
            seed=42,
            player_context={"pts_mean": 25.5, "pts_std": 5.0},
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_player_prop(req, bankroll=1000.0)

        assert resp.status == "success"
        assert resp.recommendation == "pass"
        assert resp.kelly_fraction is None
        assert resp.recommended_units is None
        assert resp.bet_side_odds is None

    def test_never_raises_on_bad_context(self):
        req = PlayerPropRequest(
            player_name="Test Player",
            league="NBA",
            prop_type="pts",
            line=20.0,
            home_team="Home Team",
            away_team="Away Team",
            game_date="2026-05-17",
            player_context={"pts_mean": "not_a_number"},
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_player_prop(req)
        assert resp.status in ("error", "skipped")

    def test_imputation_caps_tier(self):
        req = PlayerPropRequest(
            player_name="Test Player",
            league="NBA",
            prop_type="pts",
            line=20.0,
            home_team="Home Team",
            away_team="Away Team",
            game_date="2026-05-17",
            odds_over=-110,
            odds_under=-110,
            n_iterations=1000,
            seed=42,
            player_context={
                "pts_mean": 25.0,
                "pts_std": 5.0,
                "imputed_keys": ["pts_mean", "pts_std", "sample_size"],
                "sample_size": 5,
            },
            game_context=_GAME_CONTEXT,
        )
        resp = analyze_player_prop(req)
        assert resp.status == "success"
        assert resp.confidence_tier is None
        assert resp.recommendation == "pass"
        assert "insufficient_real_observations" in (resp.notes or [])

    def test_mlb_pitcher_prop_requires_std_sample_and_season(self):
        req = PlayerPropRequest(
            player_name="George Kirby",
            league="MLB",
            prop_type="strikeouts_pitched",
            line=5.5,
            home_team="Seattle Mariners",
            away_team="Texas Rangers",
            game_date="2026-05-29",
            odds_over=-110,
            odds_under=-110,
            n_iterations=1000,
            seed=42,
            player_context={"strikeouts_pitched_mean": 6.1, "sample_size": 4},
            game_context={"is_playoff": False, "rest_days": 5, "park_factor": 1.0},
        )

        resp = analyze_player_prop(req)

        assert resp.status == "skipped"
        assert resp.over_prob is None
        assert resp.edge_over is None
        assert resp.confidence_tier is None
        assert resp.recommended_units is None
        assert resp.missing_requirements == [
            "player_context.strikeouts_pitched_std",
            "player_context.sample_size>=5",
            "player_context.sample_season=2026",
        ]

    def test_mlb_pitcher_prop_season_mismatch_skips(self):
        req = PlayerPropRequest(
            player_name="George Kirby",
            league="MLB",
            prop_type="outs_recorded",
            line=17.5,
            home_team="Seattle Mariners",
            away_team="Texas Rangers",
            game_date="2026-05-29",
            n_iterations=1000,
            seed=42,
            player_context={
                "outs_recorded_mean": 18.4,
                "outs_recorded_std": 3.2,
                "sample_size": 8,
                "sample_season": 2025,
            },
            game_context={"is_playoff": False, "rest_days": 5, "park_factor": 1.0},
        )

        resp = analyze_player_prop(req)

        assert resp.status == "skipped"
        assert resp.missing_requirements == ["player_context.sample_season=2026"]

    def test_mlb_pitcher_prop_valid_distribution_succeeds(self):
        req = PlayerPropRequest(
            player_name="George Kirby",
            league="MLB",
            prop_type="strikeouts_pitched",
            line=5.5,
            home_team="Seattle Mariners",
            away_team="Texas Rangers",
            game_date="2026-05-29",
            odds_over=-110,
            odds_under=-110,
            n_iterations=500,
            seed=42,
            player_context={
                "strikeouts_pitched_mean": 6.1,
                "strikeouts_pitched_std": 2.0,
                "sample_size": 8,
                "sample_season": 2026,
            },
            game_context={"is_playoff": False, "rest_days": 5, "park_factor": 1.0},
        )

        resp = analyze_player_prop(req)

        assert resp.status == "success"
        assert resp.over_prob is not None
        assert resp.projection_std is not None

    def test_mlb_pitcher_prop_missing_distribution_marks_trace_ineligible(self):
        trace = analyze(
            {
                "player_name": "George Kirby",
                "league": "MLB",
                "prop_type": "strikeouts_pitched",
                "line": 5.5,
                "home_team": "Seattle Mariners",
                "away_team": "Texas Rangers",
                "game_date": "2026-05-29",
                "n_iterations": 500,
                "seed": 42,
                "player_context": {"strikeouts_pitched_mean": 6.1, "sample_size": 4},
                "game_context": {"is_playoff": False, "rest_days": 5, "park_factor": 1.0},
            },
            session_id="sess-test",
            bankroll=1000.0,
            trace_quality={
                "downgrades": ["mcp_exploratory_iterations"],
                "calibration_exclusion_reasons": ["mcp_exploratory_iterations"],
            },
        )

        tq = trace["trace_quality"]
        assert trace["result"]["status"] == "skipped"
        assert tq["calibration_eligible"] is False
        assert "engine_skipped" in tq["calibration_exclusion_reasons"]
        assert "player_context.strikeouts_pitched_std" in tq["calibration_exclusion_reasons"]
        assert "player_context.sample_size>=5" in tq["calibration_exclusion_reasons"]
        assert "player_context.sample_season=2026" in tq["calibration_exclusion_reasons"]
        assert "mcp_exploratory_iterations" in tq["calibration_exclusion_reasons"]


# ---------------------------------------------------------------------------
# _resolve_game_market_odds
# ---------------------------------------------------------------------------


class TestResolveGameMarketOdds:
    def test_uses_normalized_markets_over_legacy(self):
        odds = OddsInput(
            moneyline_home=-200,  # legacy (should be ignored when markets present)
            moneyline_away=170,
            markets=[
                MarketQuote(market_type="moneyline", selection="Home", price=-150),
                MarketQuote(market_type="moneyline", selection="Away", price=130),
            ],
        )
        home_odds, away_odds = _resolve_game_market_odds(odds, "Home Team", "Away Team")
        assert home_odds == pytest.approx(-150)
        assert away_odds == pytest.approx(130)

    def test_falls_back_to_legacy_moneyline(self):
        odds = OddsInput(moneyline_home=-160, moneyline_away=140)
        home_odds, away_odds = _resolve_game_market_odds(odds, "Home Team", "Away Team")
        assert home_odds == pytest.approx(-160)
        assert away_odds == pytest.approx(140)

    def test_moneyline_resolver_does_not_conflate_spread(self):
        odds = OddsInput(
            moneyline_home=-200,
            markets=[
                MarketQuote(market_type="spread", selection="Home", price=-110, line=-4.5),
                MarketQuote(market_type="moneyline", selection="Home", price=-200),
                MarketQuote(market_type="moneyline", selection="Away", price=170),
            ],
        )
        home_odds, away_odds = _resolve_game_market_odds(odds, "Home", "Away")
        assert home_odds == pytest.approx(-200)

    def test_returns_none_when_no_odds(self):
        odds = OddsInput()
        home_odds, away_odds = _resolve_game_market_odds(odds, "Home", "Away")
        assert home_odds is None
        assert away_odds is None


class TestGameMarkets:
    def _request(self, backend: str) -> GameAnalysisRequest:
        return GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=300,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
            simulation_backend=backend,
            odds=OddsInput(
                moneyline_home=-150,
                moneyline_away=130,
                spread_home=-2.5,
                spread_home_price=-110,
                spread_away_price=-110,
                over_under=218.5,
                total_over_price=-110,
                total_under_price=-110,
            ),
        )

    def test_fast_score_and_markov_emit_same_market_set(self):
        fast = analyze_game(self._request("fast_score"))
        markov = analyze_game(self._request("markov_state"))

        assert fast.status == "success"
        assert markov.status == "success"
        fast_markets = {(edge.market, edge.side) for edge in fast.edges}
        markov_markets = {(edge.market, edge.side) for edge in markov.edges}
        assert fast_markets == markov_markets
        assert fast_markets == {
            ("moneyline", "home"),
            ("moneyline", "away"),
            ("spread", "home"),
            ("spread", "away"),
            ("total", "over"),
            ("total", "under"),
        }

    def test_edge_detail_serializes_market_and_line(self):
        resp = analyze_game(self._request("fast_score"))

        spread = next(edge for edge in resp.edges if edge.market == "spread" and edge.side == "home")
        total = next(edge for edge in resp.edges if edge.market == "total" and edge.side == "over")
        assert spread.line == pytest.approx(-2.5)
        assert total.line == pytest.approx(218.5)

    def test_wnba_total_market_is_suppressed_without_best_bet_exception(self):
        req = GameAnalysisRequest(
            home_team="Dallas Wings",
            away_team="Las Vegas Aces",
            league="WNBA",
            n_iterations=300,
            seed=42,
            home_context={"off_rating": 105.0, "def_rating": 102.0, "pace": 82.0},
            away_context={"off_rating": 108.0, "def_rating": 101.0, "pace": 82.0},
            game_context={"is_playoff": False, "rest_days": 1},
            odds=OddsInput(
                over_under=166.5,
                total_over_price=-110,
                total_under_price=-110,
            ),
        )

        resp = analyze_game(req)

        assert resp.status == "success"
        assert resp.edges == []
        assert resp.best_bet is None
        assert resp.metadata.suppressed_markets == ["WNBA:total"]


# ---------------------------------------------------------------------------
# Issue #4 regression: EdgeDetail must serialize recommended_units
# ---------------------------------------------------------------------------


class TestEdgeDetailRecommendedUnits:
    def test_edge_detail_has_recommended_units(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=1000,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
            odds=OddsInput(moneyline_home=300, moneyline_away=-350),
        )
        resp = analyze_game(req, bankroll=1000.0)
        assert resp.status == "success"
        assert len(resp.edges) > 0
        for edge in resp.edges:
            assert isinstance(edge.recommended_units, float)

    def test_edge_recommended_units_nonzero_for_actionable_edge(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=1000,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
            # +300 home moneyline guarantees a large positive edge for home
            odds=OddsInput(moneyline_home=300),
        )
        resp = analyze_game(req, bankroll=1000.0)
        home_edge = next((e for e in resp.edges if e.side == "home"), None)
        assert home_edge is not None
        assert home_edge.confidence_tier in ("A", "B")
        assert home_edge.recommended_units > 0.0


# ---------------------------------------------------------------------------
# Issue #5 regression: run-line edge must use coverage prob, not win prob
# ---------------------------------------------------------------------------


class TestRunLineCoverageProb:
    _MLB_HOME_CTX = {"off_rating": 4.5, "def_rating": 3.5}
    _MLB_AWAY_CTX = {"off_rating": 3.0, "def_rating": 4.5}

    def test_spread_edge_true_prob_is_coverage_not_win_prob(self):
        req = GameAnalysisRequest(
            home_team="TB",
            away_team="BAL",
            league="MLB",
            n_iterations=5000,
            seed=42,
            home_context=self._MLB_HOME_CTX,
            away_context=self._MLB_AWAY_CTX,
            game_context=_GAME_CONTEXT,
            odds=OddsInput(spread_home=-1.5, spread_home_price=140),
        )
        resp = analyze_game(req, bankroll=1000.0)
        assert resp.status == "success"
        home_edge = next((e for e in resp.edges if e.side == "home"), None)
        assert home_edge is not None
        assert home_edge.spread_coverage_prob is not None
        # Coverage prob must be meaningfully below outright win prob
        assert home_edge.true_prob < resp.simulation.home_win_prob / 100.0 - 0.05

    def test_moneyline_edge_has_no_spread_coverage_prob(self):
        req = GameAnalysisRequest(
            home_team="Boston Celtics",
            away_team="Indiana Pacers",
            league="NBA",
            n_iterations=500,
            seed=42,
            home_context=_NBA_HOME_CTX,
            away_context=_NBA_AWAY_CTX,
            game_context=_GAME_CONTEXT,
            odds=OddsInput(moneyline_home=-150, moneyline_away=130),
        )
        resp = analyze_game(req, bankroll=1000.0)
        for edge in resp.edges:
            assert edge.spread_coverage_prob is None


# ---------------------------------------------------------------------------
# _apply_game_context — context-adjusted input assembly
# ---------------------------------------------------------------------------


class TestApplyGameContext:
    def test_no_game_context_returns_original_dict(self):
        pc = {"pts_mean": 20.0, "pts_std": 5.0}
        result = _apply_game_context(pc, {}, "pts", "NBA")
        # No context signals → factor=1.0; mean unchanged
        assert result["pts_mean"] == pytest.approx(20.0)
        assert result["_context_factor_applied"] == pytest.approx(1.0)

    def test_playoff_compresses_nba_pts_mean(self):
        pc = {"pts_mean": 20.0, "pts_std": 4.0}
        result = _apply_game_context(pc, {"is_playoff": True}, "pts", "NBA")
        assert result["pts_mean"] < 20.0
        assert result["pts_std"] < 4.0  # CV preserved
        assert result["_context_factor_applied"] < 1.0

    def test_playoff_compresses_nhl_goals(self):
        pc = {"goals_mean": 0.5, "goals_std": 0.4}
        result = _apply_game_context(pc, {"is_playoff": True}, "goals", "NHL")
        assert result["goals_mean"] < 0.5
        assert result["_context_factor_applied"] < 1.0

    def test_mlb_park_factor_boosts_hr(self):
        pc = {"hr_mean": 0.4, "hr_std": 0.3}
        result = _apply_game_context(pc, {"park_factor": 1.15}, "hr", "MLB")
        assert result["hr_mean"] > 0.4
        assert result["_context_factor_applied"] > 1.0

    def test_mlb_park_factor_ignored_for_non_power_stats(self):
        pc = {"strikeouts_mean": 6.0}
        result = _apply_game_context(pc, {"park_factor": 1.20}, "strikeouts", "MLB")
        # park_factor not applied to strikeouts
        assert result["strikeouts_mean"] == pytest.approx(6.0)

    def test_b2b_fatigue_applied_nba(self):
        pc = {"pts_mean": 20.0}
        result = _apply_game_context(pc, {"rest_days": 0}, "pts", "NBA")
        assert result["pts_mean"] < 20.0
        assert result["_context_factor_applied"] < 1.0

    def test_b2b_not_applied_for_mlb(self):
        pc = {"hits_mean": 1.2}
        result = _apply_game_context(pc, {"rest_days": 0}, "hits", "MLB")
        # MLB has no B2B fatigue
        assert result["hits_mean"] == pytest.approx(1.2)

    def test_pace_adjustment_scales_mean(self):
        pc = {"pts_mean": 20.0, "pts_std": 4.0}
        result = _apply_game_context(pc, {"pace_adjustment_factor": 0.92}, "pts", "NBA")
        assert result["pts_mean"] == pytest.approx(20.0 * 0.92)
        assert result["pts_std"] == pytest.approx(4.0 * 0.92)

    def test_unknown_league_falls_back_gracefully(self):
        pc = {"goals_mean": 0.4}
        result = _apply_game_context(pc, {"is_playoff": True}, "goals", "LIGUE1")
        # Default playoff factor (0.97) applied; no KeyError
        assert result["goals_mean"] < 0.4
        assert "_context_factor_applied" in result

    def test_missing_mean_key_returns_original(self):
        pc = {"reb_mean": 8.0}
        result = _apply_game_context(pc, {"is_playoff": True}, "pts", "NBA")
        # pts_mean absent → returned unchanged
        assert result is pc

    def test_non_numeric_mean_returns_copy_for_structured_error_path(self):
        pc = {"pts_mean": "not_a_number"}
        result = _apply_game_context(pc, _GAME_CONTEXT, "pts", "NBA")
        assert result is not pc
        assert result["pts_mean"] == "not_a_number"

    def test_does_not_mutate_original(self):
        pc = {"pts_mean": 20.0, "pts_std": 4.0}
        _ = _apply_game_context(pc, {"is_playoff": True}, "pts", "NBA")
        assert pc["pts_mean"] == pytest.approx(20.0)  # original untouched

    def test_factors_compose_playoff_and_pace(self):
        pc = {"pts_mean": 20.0}
        result = _apply_game_context(
            pc,
            {"is_playoff": True, "pace_adjustment_factor": 0.90},
            "pts",
            "NBA",
        )
        # Should be 20.0 * playoff_factor * 0.90 — both applied
        factor = result["_context_factor_applied"]
        assert result["pts_mean"] == pytest.approx(20.0 * factor)
        assert factor < 0.90  # combined is less than pace alone

    def test_analyze_prop_uses_game_context(self):
        """End-to-end: prop with playoff game_context produces lower over_prob on a high line."""
        base_req = PlayerPropRequest(
            player_name="Paul Reed Jr",
            league="NBA",
            prop_type="pts",
            line=14.5,
            home_team="Philadelphia 76ers",
            away_team="New York Knicks",
            game_date="2026-05-15",
            odds_over=-110,
            odds_under=-110,
            n_iterations=2000,
            seed=42,
            player_context={"pts_mean": 12.0, "pts_std": 5.0},
            game_context=_GAME_CONTEXT,
        )
        playoff_req = base_req.model_copy(update={"game_context": {"is_playoff": True, "rest_days": 2}})
        resp_base = analyze_player_prop(base_req)
        resp_playoff = analyze_player_prop(playoff_req)
        assert resp_base.status == "success"
        assert resp_playoff.status == "success"
        # Playoff context must reduce the over probability
        assert resp_playoff.over_prob < resp_base.over_prob

    def test_context_labels_in_analyze_trace(self):
        """analyze() populates context_labels from game_context in the trace envelope."""
        trace = analyze(
            {
                "player_name": "Test Player",
                "league": "NBA",
                "prop_type": "pts",
                "line": 20.0,
                "home_team": "Home",
                "away_team": "Away",
                "game_date": "2026-05-15",
                "n_iterations": 200,
                "seed": 1,
                "player_context": {"pts_mean": 18.0, "pts_std": 4.0},
                "game_context": {"is_playoff": True, "rest_days": 2},
            },
            session_id="sess-test",
            bankroll=1000.0,
        )
        labels = trace.get("context_labels", {})
        assert labels.get("is_playoff") is True
        assert labels.get("rest_days") == 2
