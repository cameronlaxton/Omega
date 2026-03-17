"""
Backtest Engine — replay strategies against historical game data.

The engine takes:
- A strategy configuration (edge threshold, leagues, markets, params)
- Historical games with known outcomes and closing lines

It runs each game through the simulation engine, computes edges,
applies the strategy's filters, and tracks P&L.

Design principles:
- Deterministic: same inputs → same outputs (seeded RNG)
- No network calls: all data must be pre-loaded
- Outcome-blind simulation: the engine does NOT see the actual result
  until after it decides whether to "bet"
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from omega.core.simulation.engine import OmegaSimulationEngine
from omega.core.betting.odds import (
    american_to_decimal,
    edge_percentage,
    implied_probability,
)
from omega.core.betting.kelly import recommend_stake
from omega.core.calibration.probability import calibrate_probability
from omega.strategy.models import BacktestResult, StrategyEntry

logger = logging.getLogger("omega.strategy.backtest")


class HistoricalGame(dict):
    """A historical game with context, odds, and known outcome.

    Expected keys:
        home_team: str
        away_team: str
        league: str
        home_context: dict (off_rating, def_rating, pace, etc.)
        away_context: dict
        odds: dict (moneyline_home, moneyline_away, spread_home, over_under)
        outcome: dict (home_score, away_score)
        date: str (YYYY-MM-DD)
        closing_odds: dict (optional — for CLV calculation)
    """
    pass


class BacktestEngine:
    """Replay a strategy against historical data."""

    def __init__(
        self,
        n_iterations: int = 1000,
        seed: int = 42,
    ) -> None:
        self._sim = OmegaSimulationEngine()
        self._n_iterations = n_iterations
        self._seed = seed

    def run(
        self,
        strategy: StrategyEntry,
        games: List[HistoricalGame],
    ) -> BacktestResult:
        """Run a full backtest.

        For each game:
        1. Simulate (outcome-blind)
        2. Compute edges vs market odds
        3. Apply strategy filters (edge threshold, confidence tier)
        4. If bet placed, compare to actual outcome
        5. Track P&L
        """
        run_id = f"bt-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(timezone.utc).isoformat()

        bets: List[Dict[str, Any]] = []
        by_league: Dict[str, Dict[str, Any]] = {}
        by_market: Dict[str, Dict[str, Any]] = {}

        for game in games:
            game_bets = self._process_game(strategy, game)
            bets.extend(game_bets)

            # Track by league
            league = game.get("league", "UNKNOWN")
            if league not in by_league:
                by_league[league] = {"bets": 0, "wins": 0, "units": 0.0}
            for bet in game_bets:
                by_league[league]["bets"] += 1
                by_league[league]["wins"] += 1 if bet["won"] else 0
                by_league[league]["units"] += bet["net_units"]

            # Track by market
            for bet in game_bets:
                mkt = bet.get("market_type", "moneyline")
                if mkt not in by_market:
                    by_market[mkt] = {"bets": 0, "wins": 0, "units": 0.0}
                by_market[mkt]["bets"] += 1
                by_market[mkt]["wins"] += 1 if bet["won"] else 0
                by_market[mkt]["units"] += bet["net_units"]

        # Aggregate
        wins = sum(1 for b in bets if b["won"])
        losses = sum(1 for b in bets if not b["won"] and not b.get("push", False))
        pushes = sum(1 for b in bets if b.get("push", False))
        total_wagered = sum(b["stake"] for b in bets)
        units_won = sum(b["net_units"] for b in bets if b["net_units"] > 0)
        units_lost = sum(b["net_units"] for b in bets if b["net_units"] < 0)
        net_units = sum(b["net_units"] for b in bets)

        # Max drawdown
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        for b in bets:
            running += b["net_units"]
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        # CLV
        clv_values = [b["clv"] for b in bets if b.get("clv") is not None]
        avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0.0

        # Brier score
        brier_values = [b["brier"] for b in bets if b.get("brier") is not None]
        brier_score = sum(brier_values) / len(brier_values) if brier_values else None

        avg_edge = (
            sum(b["edge_pct"] for b in bets) / len(bets) if bets else 0.0
        )

        roi = (net_units / total_wagered * 100) if total_wagered > 0 else 0.0
        win_rate = wins / len(bets) if bets else 0.0

        # Pass/fail criteria
        rejection_reasons: List[str] = []
        if len(bets) < 20:
            rejection_reasons.append(f"Insufficient sample: {len(bets)} bets (need 20+)")
        if roi < -5.0:
            rejection_reasons.append(f"Negative ROI: {roi:.1f}%")
        if win_rate < 0.40 and len(bets) >= 20:
            rejection_reasons.append(f"Low win rate: {win_rate:.1%}")
        if max_dd > 15.0:
            rejection_reasons.append(f"Max drawdown: {max_dd:.1f} units")
        if avg_clv < -2.0:
            rejection_reasons.append(f"Negative CLV: {avg_clv:.1f}%")

        passed = len(rejection_reasons) == 0 and len(bets) >= 20

        return BacktestResult(
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.version,
            run_id=run_id,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_games=len(games),
            games_with_edge=sum(1 for b in bets if b["edge_pct"] > 0),
            total_bets_placed=len(bets),
            win_count=wins,
            loss_count=losses,
            push_count=pushes,
            win_rate=round(win_rate, 4),
            roi_pct=round(roi, 2),
            units_won=round(units_won, 2),
            units_lost=round(units_lost, 2),
            net_units=round(net_units, 2),
            max_drawdown_units=round(max_dd, 2),
            avg_edge_pct=round(avg_edge, 2),
            avg_closing_line_value=round(avg_clv, 2),
            brier_score=round(brier_score, 4) if brier_score is not None else None,
            results_by_league=by_league,
            results_by_market=by_market,
            passed=passed,
            rejection_reasons=rejection_reasons,
        )

    def _process_game(
        self,
        strategy: StrategyEntry,
        game: HistoricalGame,
    ) -> List[Dict[str, Any]]:
        """Process one game: simulate, evaluate, decide, grade."""
        home_team = game.get("home_team", "Home")
        away_team = game.get("away_team", "Away")
        league = game.get("league", "NBA")
        home_ctx = game.get("home_context", {})
        away_ctx = game.get("away_context", {})
        odds = game.get("odds", {})
        outcome = game.get("outcome", {})
        closing = game.get("closing_odds", odds)

        # Check league filter
        if strategy.leagues and league.upper() not in [l.upper() for l in strategy.leagues]:
            return []

        # Simulate
        sim_result = self._sim.run_fast_game_simulation(
            home_team=home_team,
            away_team=away_team,
            league=league,
            n_iterations=self._n_iterations,
            home_context=home_ctx or None,
            away_context=away_ctx or None,
        )

        if not sim_result.get("success"):
            return []

        bets = []

        # Evaluate moneyline edges
        ml_home = odds.get("moneyline_home")
        ml_away = odds.get("moneyline_away")
        home_prob = sim_result["home_win_prob"] / 100.0
        away_prob = sim_result["away_win_prob"] / 100.0

        # Calibrate
        cal_home = calibrate_probability(home_prob)
        cal_away = calibrate_probability(away_prob)
        if isinstance(cal_home, dict):
            cal_home = cal_home.get("calibrated", home_prob)
        if isinstance(cal_away, dict):
            cal_away = cal_away.get("calibrated", away_prob)

        # Home ML
        if ml_home is not None:
            bet = self._evaluate_side(
                side="home",
                team=home_team,
                model_prob=cal_home,
                market_odds=ml_home,
                strategy=strategy,
                outcome=outcome,
                closing_odds=closing.get("moneyline_home", ml_home),
                market_type="moneyline",
            )
            if bet is not None:
                bets.append(bet)

        # Away ML
        if ml_away is not None:
            bet = self._evaluate_side(
                side="away",
                team=away_team,
                model_prob=cal_away,
                market_odds=ml_away,
                strategy=strategy,
                outcome=outcome,
                closing_odds=closing.get("moneyline_away", ml_away),
                market_type="moneyline",
            )
            if bet is not None:
                bets.append(bet)

        return bets

    def _evaluate_side(
        self,
        side: str,
        team: str,
        model_prob: float,
        market_odds: float,
        strategy: StrategyEntry,
        outcome: Dict[str, Any],
        closing_odds: Optional[float],
        market_type: str = "moneyline",
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single side for edge, decide bet, grade against outcome."""
        impl_prob = implied_probability(market_odds)
        edge = edge_percentage(model_prob, impl_prob)

        if edge < strategy.edge_threshold * 100:
            return None

        # Confidence tier
        if edge >= 8.0:
            tier = "A"
        elif edge >= 4.0:
            tier = "B"
        else:
            tier = "C"

        if tier not in strategy.confidence_tiers:
            return None

        # Kelly stake
        stake_result = recommend_stake(
            true_prob=model_prob,
            odds=market_odds,
            bankroll=1000.0,
            confidence_tier=tier,
        )
        stake = stake_result["units"]
        if stake <= 0:
            return None

        # Grade against outcome
        home_score = outcome.get("home_score")
        away_score = outcome.get("away_score")

        if home_score is None or away_score is None:
            # No outcome available
            return None

        if side == "home":
            won = home_score > away_score
        else:
            won = away_score > home_score

        push = home_score == away_score

        # Net units
        if push:
            net_units = 0.0
        elif won:
            decimal = american_to_decimal(market_odds)
            net_units = stake * (decimal - 1)
        else:
            net_units = -stake

        # CLV
        clv = None
        if closing_odds is not None:
            closing_impl = implied_probability(closing_odds)
            clv = (model_prob - closing_impl) * 100

        # Brier score component
        actual = 1.0 if won else 0.0
        brier = (model_prob - actual) ** 2

        return {
            "side": side,
            "team": team,
            "market_type": market_type,
            "model_prob": round(model_prob, 4),
            "market_odds": market_odds,
            "edge_pct": round(edge, 2),
            "tier": tier,
            "stake": round(stake, 2),
            "won": won,
            "push": push,
            "net_units": round(net_units, 2),
            "clv": round(clv, 2) if clv is not None else None,
            "brier": round(brier, 4),
        }
