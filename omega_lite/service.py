"""
Stable service interface wrapping Omega core internals.

JSON-in/JSON-out service layer. Caller supplies all context;
no data fetching, no config loading, no network calls.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from omega_lite.odds import (
    american_to_decimal,
    edge_percentage,
    expected_value_percent,
    implied_probability,
)
from omega_lite.kelly import recommend_stake
from omega_lite.schemas import (
    AnalysisMetadata,
    BetSlip,
    EdgeDetail,
    GameAnalysisRequest,
    GameAnalysisResponse,
    OddsInput,
    PlayerPropRequest,
    PlayerPropResponse,
    SimulationResult,
    SlateAnalysisRequest,
    SlateAnalysisResponse,
)
from omega_lite.engine import OmegaSimulationEngine, run_player_simulation
from omega_lite.archetypes import get_archetype, get_archetype_name
from omega_lite.calibration import apply_calibration

logger = logging.getLogger("omega_lite.service")

_engine = OmegaSimulationEngine()


def _calibrate(raw_prob: float, league: str | None = None) -> float:
    """Apply probability calibration via shared policy.

    Delegates to apply_calibration() — the single source of truth for
    calibration parameters. Do not add local overrides here.
    """
    return apply_calibration(raw_prob, league=league)


def _build_edge(
    side: str,
    team: str,
    true_prob: float,
    calibrated_prob: float,
    market_odds: float,
    bankroll: float,
    n_iterations: int,
) -> EdgeDetail:
    """Compute edge detail for one side of a matchup."""
    market_prob = implied_probability(market_odds)
    edge_pct = edge_percentage(calibrated_prob, market_prob)
    ev_pct = expected_value_percent(calibrated_prob, market_odds)
    tier = "A" if n_iterations >= 1000 else "B"
    if abs(edge_pct) < 3.0:
        tier = "Pass"

    return EdgeDetail(
        side=side,
        team=team,
        true_prob=round(true_prob, 4),
        calibrated_prob=round(calibrated_prob, 4),
        market_implied=round(market_prob, 4),
        edge_pct=round(edge_pct, 2),
        ev_pct=round(ev_pct, 2),
        market_odds=market_odds,
        confidence_tier=tier,
    )


def _pick_best_bet(
    edges: List[EdgeDetail],
    bankroll: float,
) -> Optional[BetSlip]:
    """Select the strongest edge and build a BetSlip, if any edge qualifies."""
    actionable = [e for e in edges if e.confidence_tier in ("A", "B")]
    if not actionable:
        return None
    best = max(actionable, key=lambda e: abs(e.edge_pct))
    stake = recommend_stake(
        true_prob=best.calibrated_prob,
        odds=best.market_odds,
        bankroll=bankroll,
        confidence_tier=best.confidence_tier,
    )
    return BetSlip(
        selection=f"{best.team} {best.side}",
        odds=best.market_odds,
        edge_pct=best.edge_pct,
        ev_pct=best.ev_pct,
        confidence_tier=best.confidence_tier,
        recommended_units=stake["units"],
        kelly_fraction=stake["kelly_fraction"],
    )


# ---------------------------------------------------------------------------
# analyze_game  — primary entry point
# ---------------------------------------------------------------------------

def analyze_game(
    request: GameAnalysisRequest,
    bankroll: float = 1000.0,
) -> GameAnalysisResponse:
    """Analyze a single game matchup. Never raises — returns structured response."""
    now = datetime.now().isoformat()
    matchup = f"{request.away_team} @ {request.home_team}"
    archetype_name = get_archetype_name(request.league)

    try:
        sim_result = _engine.run_fast_game_simulation(
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league,
            n_iterations=request.n_iterations,
            home_context=request.home_context,
            away_context=request.away_context,
            seed=request.seed,
        )
    except Exception as exc:
        logger.warning("Simulation error for %s: %s", matchup, exc)
        return GameAnalysisResponse(
            matchup=matchup,
            league=request.league,
            analyzed_at=now,
            status="error",
            skip_reason=f"Simulation error: {exc}",
        )

    # Skipped — propagate missing_requirements
    if not sim_result.get("success"):
        return GameAnalysisResponse(
            matchup=matchup,
            league=request.league,
            analyzed_at=now,
            status="skipped",
            skip_reason=sim_result.get("skip_reason", "Unknown skip"),
            missing_requirements=sim_result.get("missing_requirements"),
        )

    # Build simulation result
    simulation = SimulationResult(
        iterations=sim_result.get("iterations", request.n_iterations),
        home_win_prob=sim_result["home_win_prob"],
        away_win_prob=sim_result["away_win_prob"],
        draw_prob=sim_result.get("draw_prob"),
        predicted_spread=sim_result["predicted_spread"],
        predicted_total=sim_result["predicted_total"],
        predicted_home_score=sim_result.get("predicted_home_score", 0),
        predicted_away_score=sim_result.get("predicted_away_score", 0),
    )

    # Edge analysis — requires odds
    edges: List[EdgeDetail] = []
    data_sources = ["simulation"]

    if request.odds:
        data_sources.append("user_provided")
        home_prob = sim_result["home_win_prob"] / 100.0
        away_prob = sim_result["away_win_prob"] / 100.0
        draw_prob_raw = (sim_result.get("draw_prob") or 0.0) / 100.0

        # Normalize probabilities if draw exists
        total_prob = home_prob + away_prob + draw_prob_raw
        if total_prob > 0 and draw_prob_raw > 0:
            home_prob /= total_prob
            away_prob /= total_prob
            draw_prob_raw /= total_prob

        cal_home = _calibrate(home_prob, league=request.league)
        cal_away = _calibrate(away_prob, league=request.league)

        # Use spread price if available, else moneyline
        home_odds = request.odds.spread_home_price or request.odds.moneyline_home
        away_odds = request.odds.moneyline_away

        if home_odds is not None:
            edges.append(
                _build_edge("home", request.home_team, home_prob, cal_home, home_odds, bankroll, request.n_iterations)
            )
        if away_odds is not None:
            edges.append(
                _build_edge("away", request.away_team, away_prob, cal_away, away_odds, bankroll, request.n_iterations)
            )

        # 3-way moneyline (hockey regulation, soccer)
        if request.odds.moneyline_draw is not None and draw_prob_raw > 0:
            cal_draw = _calibrate(draw_prob_raw, league=request.league)
            edges.append(
                _build_edge("draw", "Draw", draw_prob_raw, cal_draw, request.odds.moneyline_draw, bankroll, request.n_iterations)
            )

    best_bet = _pick_best_bet(edges, bankroll) if edges else None

    return GameAnalysisResponse(
        matchup=matchup,
        league=request.league,
        analyzed_at=now,
        status="success",
        simulation=simulation,
        edges=edges,
        best_bet=best_bet,
        missing_requirements=[],
        metadata=AnalysisMetadata(
            data_sources=data_sources,
            archetype=archetype_name,
        ),
    )


# ---------------------------------------------------------------------------
# analyze_player_prop
# ---------------------------------------------------------------------------

def analyze_player_prop(
    request: PlayerPropRequest,
    bankroll: float = 1000.0,
) -> PlayerPropResponse:
    """Analyze a single player prop. Never raises.

    Uses run_player_simulation: archetype-aware Poisson/Normal sampling over
    a player rolling-stat distribution (mean, std) against the prop line.
    Caller must supply player_context with `{stat}_mean` and optionally
    `{stat}_std` keys (e.g. for pts: pts_mean=24.3, pts_std=6.1).
    """
    player_ctx = request.player_context or {}
    stat_key = request.prop_type
    mean_key = f"{stat_key}_mean"
    std_key = f"{stat_key}_std"

    mean = player_ctx.get(mean_key)
    if mean is None:
        return PlayerPropResponse(
            player_name=request.player_name,
            league=request.league,
            prop_type=request.prop_type,
            line=request.line,
            status="skipped",
            skip_reason=f"Missing required input player_context.{mean_key}",
            missing_requirements=[f"player_context.{mean_key}"],
        )

    std_val = player_ctx.get(std_key, max(1.0, float(mean) * 0.25))
    try:
        mean_f = float(mean)
        std_f = float(std_val)
    except (TypeError, ValueError) as exc:
        return PlayerPropResponse(
            player_name=request.player_name,
            league=request.league,
            prop_type=request.prop_type,
            line=request.line,
            status="error",
            skip_reason=f"Non-numeric player_context values: {exc}",
        )

    player_proj = {
        "league": request.league,
        "stat_key": stat_key,
        "mean": mean_f,
        "variance": std_f ** 2,
        "market_line": request.line,
    }

    try:
        sim_result = run_player_simulation(
            player_proj,
            n_iter=request.n_iterations,
            seed=request.seed,
        )
    except Exception as exc:
        logger.warning("Prop simulation error for %s: %s", request.player_name, exc)
        return PlayerPropResponse(
            player_name=request.player_name,
            league=request.league,
            prop_type=request.prop_type,
            line=request.line,
            status="error",
            skip_reason=f"Simulation error: {exc}",
        )

    # run_player_simulation returns probabilities as decimals (0-1)
    over_prob = sim_result.get("over_prob", 0.0)
    under_prob = sim_result.get("under_prob", 0.0)

    # Compute edges if odds provided
    edge_over = None
    edge_under = None
    recommendation = "pass"
    tier = None

    if request.odds_over is not None:
        market_over = implied_probability(request.odds_over)
        cal_over = _calibrate(over_prob, league=request.league)
        edge_over = round(edge_percentage(cal_over, market_over), 2)

    if request.odds_under is not None:
        market_under = implied_probability(request.odds_under)
        cal_under = _calibrate(under_prob, league=request.league)
        edge_under = round(edge_percentage(cal_under, market_under), 2)

    if edge_over is not None and edge_under is not None:
        if edge_over > 3.0 and edge_over > edge_under:
            recommendation = "over"
            tier = "A" if request.n_iterations >= 1000 else "B"
        elif edge_under > 3.0:
            recommendation = "under"
            tier = "A" if request.n_iterations >= 1000 else "B"

    return PlayerPropResponse(
        player_name=request.player_name,
        league=request.league,
        prop_type=request.prop_type,
        line=request.line,
        status="success",
        over_prob=round(over_prob, 4),
        under_prob=round(under_prob, 4),
        edge_over=edge_over,
        edge_under=edge_under,
        recommendation=recommendation,
        confidence_tier=tier,
        missing_requirements=[],
    )


# ---------------------------------------------------------------------------
# analyze_slate
# ---------------------------------------------------------------------------

def analyze_slate(
    request: SlateAnalysisRequest,
    games: Optional[List[Dict[str, Any]]] = None,
) -> SlateAnalysisResponse:
    """Analyze a slate of games. Loops analyze_game per game; catches errors per-game.
    Does not fetch games; caller must supply request.games or games argument."""
    date_str = request.date or datetime.now().strftime("%Y-%m-%d")

    games = games if games is not None else request.games
    if not games:
        games = []

    analyses: List[GameAnalysisResponse] = []
    for game in games:
        home = _extract_team(game, "home_team") or _extract_team(game, "home")
        away = _extract_team(game, "away_team") or _extract_team(game, "away")
        if not home or not away:
            continue

        odds_dict = game.get("odds") or game.get("markets") or {}
        odds_input = None
        if odds_dict:
            odds_input = OddsInput(
                spread_home=odds_dict.get("spread_home"),
                spread_home_price=odds_dict.get("spread_home_price", -110),
                moneyline_home=odds_dict.get("moneyline_home"),
                moneyline_away=odds_dict.get("moneyline_away"),
                over_under=odds_dict.get("over_under"),
                moneyline_draw=odds_dict.get("moneyline_draw"),
            )

        game_request = GameAnalysisRequest(
            home_team=home,
            away_team=away,
            league=request.league,
            odds=odds_input,
            home_context=game.get("home_context"),
            away_context=game.get("away_context"),
        )
        result = analyze_game(game_request, bankroll=request.bankroll)
        analyses.append(result)

    games_with_edge = sum(1 for a in analyses if a.best_bet is not None)

    return SlateAnalysisResponse(
        league=request.league,
        date=date_str,
        total_games=len(games),
        games_analyzed=len(analyses),
        games_with_edge=games_with_edge,
        analyses=analyses,
    )


def _extract_team(game: Dict[str, Any], key: str) -> Optional[str]:
    """Safely extract team name from various game dict shapes."""
    val = game.get(key)
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("name") or val.get("team") or val.get("full_name")
    return None
