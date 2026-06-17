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
from typing import Any

from omega.core.betting.kelly import recommend_stake
from omega.core.betting.odds import (
    american_to_decimal,
    edge_percentage,
    implied_probability,
)
from omega.core.calibration.probability import apply_calibration
from omega.core.simulation.engine import OmegaSimulationEngine
from omega.strategy.artifacts import FrozenArtifact, compat_dict_to_artifact
from omega.strategy.models import BacktestResult, StrategyEntry

UTC = timezone.utc

logger = logging.getLogger("omega.strategy.backtest")


def _grade_selection(
    market_type: str,
    side: str,
    home_score: float,
    away_score: float,
    line: float | None = None,
) -> tuple[bool, bool]:
    """Grade one bet selection against a final score. Returns (won, push).

    Covers moneyline (incl. 3-way draw), point spread, game total, and the exotic
    soccer markets. A draw moneyline wins on a tie (not a push); straight
    home/away push on a tie; draw-no-bet voids (pushes) on a tie; double chance /
    BTTS / correct score never push. Spread and total push on the exact number
    (``line`` is signed the way it was bet — home's spread for side='home',
    its negation for side='away', the total line for over/under).
    """
    is_tie = home_score == away_score

    if market_type == "spread":
        if line is None:
            return False, False
        margin = (home_score - away_score) if side == "home" else (away_score - home_score)
        value = margin + line
        return value > 0, value == 0

    if market_type == "total":
        if line is None:
            return False, False
        total = home_score + away_score
        if side == "over":
            return total > line, total == line
        return total < line, total == line

    if market_type == "draw_no_bet":
        if is_tie:
            return False, True  # void / push
        if side == "home":
            return home_score > away_score, False
        return away_score > home_score, False

    if market_type == "double_chance":
        if side == "home_draw":
            return home_score >= away_score, False
        if side == "away_draw":
            return away_score >= home_score, False
        if side == "home_away":
            return not is_tie, False
        return False, False

    if market_type == "both_teams_to_score":
        both = home_score > 0 and away_score > 0
        if side == "yes":
            return both, False
        return not both, False

    if market_type == "correct_score":
        # side is the scoreline "home-away".
        try:
            h, a = (int(x) for x in side.split("-"))
        except (ValueError, AttributeError):
            return False, False
        return home_score == h and away_score == a, False

    # Moneyline (2-way and 3-way).
    if side == "draw":
        return is_tie, False
    if side == "home":
        return home_score > away_score, is_tie
    return away_score > home_score, is_tie


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
        exact_eval: bool = False,
    ) -> None:
        self._sim = OmegaSimulationEngine()
        self._n_iterations = n_iterations
        self._seed = seed
        # When True, parametric backends (integer-Poisson archetypes) evaluate
        # market probabilities exactly instead of Monte-Carlo sampling. Removes
        # the MC sampling noise the edge filter selects on (optimizer's curse),
        # making backtest probabilities — and the calibration pairs derived from
        # them — noise-free. Non-parametric backends ignore the flag.
        self._exact_eval = exact_eval

    def run(
        self,
        strategy: StrategyEntry,
        games: list[FrozenArtifact] | list[HistoricalGame] | list[dict[str, Any]],
    ) -> BacktestResult:
        """Run a full backtest.

        Accepts FrozenArtifact objects (preferred) or legacy HistoricalGame dicts.
        Legacy dicts are auto-converted via compat_dict_to_artifact().

        For each game:
        1. Simulate (outcome-blind)
        2. Compute edges vs market odds
        3. Apply strategy filters (edge threshold, confidence tier)
        4. If bet placed, compare to actual outcome
        5. Track P&L
        """
        run_id = f"bt-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(UTC).isoformat()

        # Normalize inputs to FrozenArtifact
        artifacts = self._normalize_inputs(games)

        bets: list[dict[str, Any]] = []
        by_league: dict[str, dict[str, Any]] = {}
        by_market: dict[str, dict[str, Any]] = {}
        trace_ids: list[str] = []

        for artifact in artifacts:
            game_bets = self._process_artifact(strategy, artifact)
            bets.extend(game_bets)

            if artifact.source_trace_id:
                trace_ids.append(artifact.source_trace_id)

            # Track by league
            league = artifact.league or "UNKNOWN"
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

        avg_edge = sum(b["edge_pct"] for b in bets) / len(bets) if bets else 0.0

        roi = (net_units / total_wagered * 100) if total_wagered > 0 else 0.0
        win_rate = wins / len(bets) if bets else 0.0

        # Pass/fail criteria
        rejection_reasons: list[str] = []
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
            completed_at=datetime.now(UTC).isoformat(),
            total_games=len(artifacts),
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
            trace_ids=trace_ids,
            passed=passed,
            rejection_reasons=rejection_reasons,
        )

    @staticmethod
    def _normalize_inputs(
        games: list[FrozenArtifact] | list[HistoricalGame] | list[dict[str, Any]],
    ) -> list[FrozenArtifact]:
        """Convert mixed inputs to a uniform list of FrozenArtifacts."""
        if not games:
            return []
        if isinstance(games[0], FrozenArtifact):
            return games  # type: ignore[return-value]
        return [compat_dict_to_artifact(dict(g)) for g in games]

    def _process_game(
        self,
        strategy: StrategyEntry,
        game: HistoricalGame,
    ) -> list[dict[str, Any]]:
        """Legacy entry point — delegates to _process_artifact via shim."""
        artifact = compat_dict_to_artifact(dict(game))
        return self._process_artifact(strategy, artifact)

    def _process_artifact(
        self,
        strategy: StrategyEntry,
        artifact: FrozenArtifact,
    ) -> list[dict[str, Any]]:
        """Process one artifact: simulate, evaluate, decide, grade."""
        home_team = artifact.home_team
        away_team = artifact.away_team
        league = artifact.league
        home_ctx = artifact.home_context
        away_ctx = artifact.away_context
        game_ctx = artifact.game_context
        odds = artifact.odds
        outcome = artifact.outcome or {}
        closing = artifact.closing_odds or odds

        # Check league filter
        if strategy.leagues and league.upper() not in [
            strategy_league.upper() for strategy_league in strategy.leagues
        ]:
            return []

        # Spread/total lines, passed to the sim so it computes cover/over
        # probabilities (the same inputs the production path supplies).
        spread_line = odds.get("spread_home")
        total_line = odds.get("over_under")

        # Simulate
        sim_result = self._sim.run_fast_game_simulation(
            home_team=home_team,
            away_team=away_team,
            league=league,
            n_iterations=self._n_iterations,
            home_context=home_ctx or None,
            away_context=away_ctx or None,
            seed=artifact.simulation_seed if artifact.simulation_seed is not None else self._seed,
            spread_home=spread_line,
            over_under=total_line,
            exact=self._exact_eval,
        )

        if not sim_result.get("success"):
            return []

        bets = []

        # Evaluate moneyline edges
        ml_home = odds.get("moneyline_home")
        ml_away = odds.get("moneyline_away")
        home_prob = sim_result["home_win_prob"] / 100.0
        away_prob = sim_result["away_win_prob"] / 100.0

        # Calibrate via shared policy (must match production path)
        cal_home = apply_calibration(home_prob, league=league, context_hints=game_ctx or None)
        cal_away = apply_calibration(away_prob, league=league, context_hints=game_ctx or None)

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

        # Draw ML (3-way markets: soccer, hockey regulation). Only evaluated when
        # a draw price exists and the simulator reports real draw mass.
        ml_draw = odds.get("moneyline_draw")
        draw_prob = (sim_result.get("draw_prob") or 0.0) / 100.0
        if ml_draw is not None and draw_prob > 0:
            cal_draw = apply_calibration(
                draw_prob, league=league, context_hints=game_ctx or None, market="draw"
            )
            bet = self._evaluate_side(
                side="draw",
                team="Draw",
                model_prob=cal_draw,
                market_odds=ml_draw,
                strategy=strategy,
                outcome=outcome,
                closing_odds=closing.get("moneyline_draw", ml_draw),
                market_type="moneyline",
            )
            if bet is not None:
                bets.append(bet)

        # Exotic soccer markets: double chance, draw-no-bet, BTTS, correct score.
        # Each is evaluated only when both a price (in the artifact odds) and the
        # corresponding simulated probability are present. Probabilities reuse
        # the game calibration plane (cold-start) until exotic profiles exist.
        # (price_key, sim_prob_key, side, team_label, market_type)
        exotic_specs = [
            ("dc_home_draw", "double_chance_home_draw_prob", "home_draw", "Home or Draw", "double_chance"),
            ("dc_home_away", "double_chance_home_away_prob", "home_away", "Home or Away", "double_chance"),
            ("dc_away_draw", "double_chance_away_draw_prob", "away_draw", "Away or Draw", "double_chance"),
            ("dnb_home", "dnb_home_prob", "home", home_team, "draw_no_bet"),
            ("dnb_away", "dnb_away_prob", "away", away_team, "draw_no_bet"),
            ("btts_yes", "btts_yes_prob", "yes", "BTTS Yes", "both_teams_to_score"),
            ("btts_no", "btts_no_prob", "no", "BTTS No", "both_teams_to_score"),
        ]
        for price_key, prob_key, side, team_label, market_type in exotic_specs:
            price = odds.get(price_key)
            sim_prob = sim_result.get(prob_key)
            if price is None or sim_prob is None:
                continue
            prob = sim_prob / 100.0
            if prob <= 0:
                continue
            cal = apply_calibration(prob, league=league, context_hints=game_ctx or None)
            bet = self._evaluate_side(
                side=side,
                team=team_label,
                model_prob=cal,
                market_odds=price,
                strategy=strategy,
                outcome=outcome,
                closing_odds=closing.get(price_key, price),
                market_type=market_type,
            )
            if bet is not None:
                bets.append(bet)

        # Correct score: a map of "home-away" scoreline -> price.
        cs_prices = odds.get("correct_score") or {}
        cs_probs = sim_result.get("correct_score_probs") or {}
        for scoreline, price in cs_prices.items():
            sim_prob = cs_probs.get(scoreline)
            if price is None or sim_prob is None:
                continue
            prob = sim_prob / 100.0
            if prob <= 0:
                continue
            cal = apply_calibration(prob, league=league, context_hints=game_ctx or None)
            bet = self._evaluate_side(
                side=scoreline,
                team=f"Correct Score {scoreline}",
                model_prob=cal,
                market_odds=price,
                strategy=strategy,
                outcome=outcome,
                closing_odds=(closing.get("correct_score") or {}).get(scoreline, price),
                market_type="correct_score",
            )
            if bet is not None:
                bets.append(bet)

        # Point spread. Cover probabilities use the dedicated 'cover' calibration
        # plane (matching production); the home spread is bet at spread_line, the
        # away side at its negation.
        if spread_line is not None and "home_cover_prob" in sim_result:
            spread_specs = [
                ("home", home_team, "home_cover_prob", "spread_home_price", spread_line),
                ("away", away_team, "away_cover_prob", "spread_away_price", -spread_line),
            ]
            for side, team, prob_key, price_key, bet_line in spread_specs:
                price = odds.get(price_key) if price_key in odds else None
                sim_prob = sim_result.get(prob_key)
                if price is None or sim_prob is None:
                    continue
                cal = apply_calibration(
                    sim_prob / 100.0, league=league, context_hints=game_ctx or None, market="cover"
                )
                bet = self._evaluate_side(
                    side=side,
                    team=team,
                    model_prob=cal,
                    market_odds=price,
                    strategy=strategy,
                    outcome=outcome,
                    closing_odds=closing.get(price_key, price),
                    market_type="spread",
                    line=bet_line,
                )
                if bet is not None:
                    bets.append(bet)

        # Game total. Over/under use their own calibration planes (matching
        # production); both are bet at total_line.
        if total_line is not None and "over_prob" in sim_result and "under_prob" in sim_result:
            total_specs = [
                ("over", f"Over {total_line:g}", "over_prob", "total_over_price", "over"),
                ("under", f"Under {total_line:g}", "under_prob", "total_under_price", "under"),
            ]
            for side, team, prob_key, price_key, cal_market in total_specs:
                price = odds.get(price_key) if price_key in odds else None
                sim_prob = sim_result.get(prob_key)
                if price is None or sim_prob is None:
                    continue
                cal = apply_calibration(
                    sim_prob / 100.0, league=league, context_hints=game_ctx or None, market=cal_market
                )
                bet = self._evaluate_side(
                    side=side,
                    team=team,
                    model_prob=cal,
                    market_odds=price,
                    strategy=strategy,
                    outcome=outcome,
                    closing_odds=closing.get(price_key, price),
                    market_type="total",
                    line=total_line,
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
        outcome: dict[str, Any],
        closing_odds: float | None,
        market_type: str = "moneyline",
        line: float | None = None,
    ) -> dict[str, Any] | None:
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

        # Selection-aware grading across moneyline (incl. 3-way draw) and the
        # exotic soccer markets (double chance, draw-no-bet, BTTS, correct score).
        won, push = _grade_selection(market_type, side, home_score, away_score, line)

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
            "line": line,
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
