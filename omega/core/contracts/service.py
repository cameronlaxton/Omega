"""
Stable service interface wrapping Omega core internals.

JSON-in/JSON-out service layer. Caller supplies all context;
no data fetching, no config loading, no network calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
UTC = timezone.utc
from typing import Any

from omega.core.betting.kelly import recommend_stake
from omega.core.betting.odds import (
    edge_percentage,
    expected_value_percent,
    implied_probability,
)
from omega.core.calibration.probability import apply_calibration, apply_calibration_audited
from omega.core.contracts.schemas import (
    AnalysisMetadata,
    BetSlip,
    CalibrationAudit,
    EdgeDetail,
    GameAnalysisRequest,
    GameAnalysisResponse,
    MarketQuote,
    OddsInput,
    PlayerPropRequest,
    PlayerPropResponse,
    SimulationResult,
    SlateAnalysisRequest,
    SlateAnalysisResponse,
)
from omega.core.simulation.archetypes import get_archetype_name
from omega.core.simulation.engine import (
    OmegaSimulationEngine,
    run_player_simulation,
    select_distribution,
)

logger = logging.getLogger("omega.service")

_engine = OmegaSimulationEngine()
MODEL_VERSION = "omega-core-phase6h"
_TRACE_HASH_EXCLUDE = {
    "odds",
    "odds_over",
    "odds_under",
    "markets",
    "odds_snapshot",
    "market_snapshots",
    "closing_line",
    "closing_lines",
}


def _sanitize_for_trace_hash(value: Any) -> Any:
    """Drop volatile market fields before deriving trace hash identity."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(k): _sanitize_for_trace_hash(v)
            for k, v in value.items()
            if str(k) not in _TRACE_HASH_EXCLUDE
        }
    if isinstance(value, list):
        return [_sanitize_for_trace_hash(item) for item in value]
    return value


def _stable_input_hash(request: Any) -> str | None:
    """Stable 8-char content hash, excluding volatile odds and close snapshots."""
    try:
        payload = _sanitize_for_trace_hash(request)
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return None


def _new_trace_id(request: Any = None) -> str:
    """Python-minted trace id for VM/MCP runs.

    The stable hash prefix identifies the simulation context while the nonce
    keeps repeated runs separately persistable. Volatile odds structures are
    intentionally excluded from the hash to avoid cache fracturing.
    """
    if request is not None:
        h = _stable_input_hash(request)
        if h is not None:
            return f"sandbox-{h}-{uuid.uuid4().hex[:4]}"
    return f"sandbox-{uuid.uuid4().hex[:12]}"


def _coerce_request(
    request: dict[str, Any] | GameAnalysisRequest | PlayerPropRequest | SlateAnalysisRequest,
) -> GameAnalysisRequest | PlayerPropRequest | SlateAnalysisRequest:
    """Accept a typed request or a dict and return the canonical typed request."""
    if isinstance(request, (GameAnalysisRequest, PlayerPropRequest, SlateAnalysisRequest)):
        return request
    if not isinstance(request, dict):
        raise TypeError(f"Expected dict or request object, got {type(request).__name__}")

    if "player_name" in request and "prop_type" in request:
        return PlayerPropRequest(**request)
    if "games" in request or (
        "league" in request and "home_team" not in request and "player_name" not in request
    ):
        return SlateAnalysisRequest(**request)
    return GameAnalysisRequest(**request)


def _safe_dump(obj: Any) -> dict[str, Any]:
    """Convert Pydantic models and dicts to JSON-friendly dictionaries."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def _result_downgrades(result: Any) -> list[str]:
    """Expose minimal trace-level downgrades without reviving the lite gate."""
    status = getattr(result, "status", None)
    if status == "skipped":
        return ["engine_skipped"]
    if status == "error":
        return ["engine_error"]
    return []


def analyze(
    request: dict[str, Any] | GameAnalysisRequest | PlayerPropRequest | SlateAnalysisRequest,
    *,
    session_id: str,
    bankroll: float,
) -> dict[str, Any]:
    """Run a canonical Omega analysis and return an auditable trace envelope.

    This wrapper owns only request typing, trace identity, and audit snapshots.
    Simulation, calibration, edge, EV, Kelly, staking, and confidence tiers stay
    in the deterministic analyzers below.
    """
    if not session_id:
        raise ValueError("session_id is required for canonical analyze()")
    if bankroll is None or bankroll <= 0:
        raise ValueError("bankroll must be a positive explicit value")

    typed_req = _coerce_request(request)
    trace_id = _new_trace_id(typed_req)
    ran_at = datetime.now(UTC).isoformat()

    result: GameAnalysisResponse | PlayerPropResponse | SlateAnalysisResponse
    if isinstance(typed_req, GameAnalysisRequest):
        result = analyze_game(typed_req, bankroll=bankroll)
        kind = "game"
    elif isinstance(typed_req, PlayerPropRequest):
        result = analyze_player_prop(typed_req, bankroll=bankroll)
        kind = "prop"
    elif isinstance(typed_req, SlateAnalysisRequest):
        typed_req = typed_req.model_copy(update={"bankroll": bankroll})
        result = analyze_slate(typed_req)
        kind = "slate"
    else:
        raise TypeError(
            f"Unsupported request type {type(typed_req).__name__}. "
            "Expected GameAnalysisRequest, PlayerPropRequest, or SlateAnalysisRequest."
        )

    # Populate context_labels for calibration slice fitting.
    # Covers both GameAnalysisRequest and PlayerPropRequest via _build_context_labels().
    context_labels: dict[str, Any] = {}
    if isinstance(typed_req, (GameAnalysisRequest, PlayerPropRequest)):
        context_labels = _build_context_labels(typed_req)

    return {
        "trace_id": trace_id,
        "model_version": MODEL_VERSION,
        "ran_at": ran_at,
        "kind": kind,
        "session_id": session_id,
        "bankroll": bankroll,
        "input_snapshot": _safe_dump(typed_req),
        "result": _safe_dump(result),
        "downgrades": _result_downgrades(result),
        "context_labels": context_labels,
    }


def _calibrate(
    raw_prob: float,
    league: str | None = None,
    context_hints: dict[str, Any] | None = None,
) -> float:
    """Apply calibration. Delegates to apply_calibration() — the single source of truth."""
    return apply_calibration(raw_prob, league=league, context_hints=context_hints)


def _calibrate_audited(
    raw_prob: float,
    league: str | None = None,
    context_hints: dict[str, Any] | None = None,
    plane: str = "game",
    market: str = "home",
) -> tuple[float, CalibrationAudit]:
    """Like _calibrate() but also returns a CalibrationAudit recording the path taken."""
    calibrated, d = apply_calibration_audited(raw_prob, league=league, context_hints=context_hints)
    audit = CalibrationAudit(
        raw_prob=d["raw_prob"],
        calibrated_prob=d["calibrated_prob"],
        league=league,
        plane=plane,
        market=market,
        method_resolved=d.get("method_resolved"),
        profile_id=d.get("profile_id"),
        context_slice=d.get("context_slice"),
        resolved_slice=d.get("resolved_slice"),
        path=d["path"],
    )
    return calibrated, audit


def _build_context_labels(
    request: GameAnalysisRequest | PlayerPropRequest,
) -> dict[str, Any]:
    """Extract calibration context labels from any request type.

    Passes through all game_context keys so the fitter can use any signal
    the agent supplies, not just the known set at coding time.
    """
    gc: dict[str, Any] | None = getattr(request, "game_context", None)
    if not gc:
        return {}
    labels: dict[str, Any] = {}
    # Type-coerce the known calibration slice keys for safety
    if gc.get("is_playoff") is not None:
        labels["is_playoff"] = bool(gc["is_playoff"])
    if gc.get("rest_days") is not None:
        labels["rest_days"] = int(gc["rest_days"])
    if gc.get("blowout_risk") is not None:
        labels["blowout_risk"] = float(gc["blowout_risk"])
    # Pass through any additional context keys (matchup weaknesses, scheme signals, etc.)
    for k, v in gc.items():
        if k not in labels:
            labels[k] = v
    return labels


def _build_edge(
    side: str,
    team: str,
    true_prob: float,
    calibrated_prob: float,
    market_odds: float,
    bankroll: float,
    n_iterations: int,
    calibration_audit: CalibrationAudit | None = None,
) -> EdgeDetail:
    """Compute edge detail for one side of a matchup."""
    market_prob = implied_probability(market_odds)
    edge_pct = edge_percentage(calibrated_prob, market_prob)
    ev_pct = expected_value_percent(calibrated_prob, market_odds)
    tier = "A" if n_iterations >= 1000 else "B"
    if abs(edge_pct) < 3.0:
        tier = "Pass"

    stake = recommend_stake(
        true_prob=calibrated_prob,
        odds=market_odds,
        bankroll=bankroll,
        confidence_tier=tier,
    )

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
        recommended_units=stake["units"],
        calibration_audit=calibration_audit,
    )


def _pick_best_bet(
    edges: list[EdgeDetail],
    bankroll: float,
) -> BetSlip | None:
    """Select the strongest edge and build a BetSlip, if any edge qualifies."""
    actionable = [e for e in edges if e.confidence_tier in ("A", "B") and e.ev_pct > 0]
    if not actionable:
        return None
    best = max(actionable, key=lambda e: e.ev_pct)
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


def _quote_matches_selection(quote: MarketQuote, *labels: str) -> bool:
    selection = quote.selection.strip().casefold()
    return any(selection == label.strip().casefold() for label in labels if label)


def _market_quote(
    odds: OddsInput,
    market_type: str,
    *labels: str,
) -> MarketQuote | None:
    for quote in odds.markets or []:
        if quote.market_type != market_type:
            continue
        if _quote_matches_selection(quote, *labels):
            return quote
    return None


def _resolve_game_market_odds(
    odds: OddsInput,
    home_team: str,
    away_team: str,
) -> tuple[float | None, float | None]:
    """Resolve home/away odds from normalized markets first, legacy fields second."""
    home_spread = _market_quote(odds, "spread", home_team, "Home")
    home_ml = _market_quote(odds, "moneyline", home_team, "Home")
    away_ml = _market_quote(odds, "moneyline", away_team, "Away")

    # Only use spread_home_price when spread_home was explicitly supplied;
    # otherwise fall through to moneyline_home. The spread_home_price field
    # defaults to -110 which would otherwise shadow a real moneyline_home value.
    legacy_home = (
        odds.spread_home_price if odds.spread_home is not None else odds.moneyline_home
    )
    home_odds = (
        home_spread.price
        if home_spread is not None
        else (home_ml.price if home_ml is not None else legacy_home)
    )
    away_odds = away_ml.price if away_ml is not None else odds.moneyline_away
    return home_odds, away_odds


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

    # Extract spread value so the engine can compute coverage probabilities.
    # Done before the engine call; odds may be absent.
    spread_value: float | None = None
    if request.odds:
        _hsq = _market_quote(request.odds, "spread", request.home_team, "Home")
        if _hsq and _hsq.line is not None:
            spread_value = _hsq.line
        elif request.odds.spread_home is not None:
            spread_value = request.odds.spread_home

    try:
        sim_result = _engine.run_fast_game_simulation(
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league,
            n_iterations=request.n_iterations,
            home_context=request.home_context,
            away_context=request.away_context,
            seed=request.seed,
            spread_home=spread_value,
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
    edges: list[EdgeDetail] = []
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

        home_odds, away_odds = _resolve_game_market_odds(
            request.odds,
            request.home_team,
            request.away_team,
        )

        # Detect whether the home market is a spread/run-line so we can
        # substitute coverage probability for outright win probability.
        home_spread_q = _market_quote(request.odds, "spread", request.home_team, "Home")
        home_ml_q = _market_quote(request.odds, "moneyline", request.home_team, "Home")
        is_home_spread_market = (home_spread_q is not None) or (
            request.odds.spread_home is not None
            and home_ml_q is None
            and request.odds.moneyline_home is None
        )

        gc = request.game_context
        if home_odds is not None:
            if is_home_spread_market and "home_cover_prob" in sim_result:
                cover_prob = sim_result["home_cover_prob"] / 100.0
                cal_cover, cover_audit = _calibrate_audited(cover_prob, league=request.league, context_hints=gc, plane="game", market="cover")
                home_edge = _build_edge(
                    "home", request.home_team, cover_prob, cal_cover,
                    home_odds, bankroll, request.n_iterations,
                    calibration_audit=cover_audit,
                )
                home_edge = home_edge.model_copy(update={"spread_coverage_prob": cover_prob})
            else:
                cal_home, home_audit = _calibrate_audited(home_prob, league=request.league, context_hints=gc, plane="game", market="home")
                home_edge = _build_edge(
                    "home", request.home_team, home_prob, cal_home,
                    home_odds, bankroll, request.n_iterations,
                    calibration_audit=home_audit,
                )
            edges.append(home_edge)

        if away_odds is not None:
            cal_away, away_audit = _calibrate_audited(away_prob, league=request.league, context_hints=gc, plane="game", market="away")
            edges.append(
                _build_edge("away", request.away_team, away_prob, cal_away, away_odds, bankroll, request.n_iterations, calibration_audit=away_audit)
            )

        # 3-way moneyline (hockey regulation, soccer)
        if request.odds.moneyline_draw is not None and draw_prob_raw > 0:
            cal_draw, draw_audit = _calibrate_audited(draw_prob_raw, league=request.league, context_hints=gc, plane="game", market="draw")
            edges.append(
                _build_edge("draw", "Draw", draw_prob_raw, cal_draw, request.odds.moneyline_draw, bankroll, request.n_iterations, calibration_audit=draw_audit)
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
# ---------------------------------------------------------------------------
# Context-adjusted input assembly
# ---------------------------------------------------------------------------

# Per-league, per-stat playoff suppression factors (conservative empirical defaults).
# Missing league falls back to _DEFAULT_PLAYOFF_FACTOR; missing stat within a
# known league also falls back to the default.
_PLAYOFF_STAT_FACTORS: dict[str, dict[str, float]] = {
    "NBA":  {"pts": 0.96, "reb": 0.97, "ast": 0.96, "3pm": 0.94, "stl": 0.95, "blk": 0.95},
    "NHL":  {"goals": 0.94, "assists": 0.95, "shots": 0.94, "saves": 0.97},
    "MLB":  {"hits": 0.97, "hr": 0.93, "rbis": 0.96, "strikeouts": 1.02, "total_bases": 0.95},
    "NFL":  {"pass_yds": 0.97, "rush_yds": 0.97, "rec_yds": 0.97, "receptions": 0.97},
    "EPL":  {"goals": 0.96, "assists": 0.97, "shots": 0.95},
    "MLS":  {"goals": 0.96, "assists": 0.97, "shots": 0.95},
}
_DEFAULT_PLAYOFF_FACTOR = 0.97
# Back-to-back fatigue — only meaningful in sports with consecutive-night scheduling.
_B2B_FATIGUE: dict[str, float] = {"NBA": 0.94, "NHL": 0.95}
# MLB park factor applies to power/extra-base counting stats only.
_MLB_PARK_FACTOR_STATS = frozenset({"hr", "total_bases", "rbis"})


def _apply_game_context(
    player_context: dict[str, Any],
    game_context: dict[str, Any],
    prop_type: str,
    league: str,
) -> dict[str, Any]:
    """Return an adjusted copy of player_context using game_context signals.

    Applies sport-appropriate factors to {prop_type}_mean and {prop_type}_std.
    Never mutates the input dict. Stores _context_factor_applied for audit.
    Returns the original dict unchanged if mean_key is absent.
    """
    mean_key = f"{prop_type}_mean"
    std_key = f"{prop_type}_std"
    if mean_key not in player_context:
        return player_context

    ctx = dict(player_context)
    league_uc = league.upper()
    factor = 1.0

    if game_context.get("is_playoff"):
        league_factors = _PLAYOFF_STAT_FACTORS.get(league_uc, {})
        factor *= league_factors.get(prop_type, _DEFAULT_PLAYOFF_FACTOR)

    # B2B fatigue: rest_days=0 means the team played the previous night.
    # Only applied for sports with consecutive-night schedules (NBA, NHL).
    _rest_days = game_context.get("rest_days")
    if _rest_days is not None and int(_rest_days) == 0 and league_uc in _B2B_FATIGUE:
        factor *= _B2B_FATIGUE[league_uc]

    # Pace scales counting stats proportionally (NBA, NHL, soccer).
    if (paf := game_context.get("pace_adjustment_factor")) is not None:
        factor *= float(paf)

    # MLB park factor boosts/suppresses power stats in hitter/pitcher-friendly parks.
    if league_uc == "MLB" and prop_type in _MLB_PARK_FACTOR_STATS:
        if (park := game_context.get("park_factor")) is not None:
            factor *= float(park)

    ctx[mean_key] = ctx[mean_key] * factor
    if std_key in ctx:
        ctx[std_key] = ctx[std_key] * factor  # preserve coefficient of variation
    ctx["_context_factor_applied"] = round(factor, 4)
    return ctx


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

    if request.game_context:
        player_ctx = _apply_game_context(
            player_ctx, request.game_context, stat_key, request.league
        )

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

    try:
        mean_f = float(mean)
        std_raw = player_ctx.get(std_key)
        std_f = float(std_raw) if std_raw is not None else max(1.0, mean_f * 0.25)
    except (TypeError, ValueError) as exc:
        return PlayerPropResponse(
            player_name=request.player_name,
            league=request.league,
            prop_type=request.prop_type,
            line=request.line,
            status="error",
            skip_reason=f"Non-numeric player_context values: {exc}",
        )

    # B1: surface optional distribution override from caller. select_distribution()
    # also auto-routes low-mean basketball count stats (blk/stl/3pm/oreb/dreb/to
    # with mean < 3.0) to Poisson where Normal would understate right-tail mass.
    distribution_override = player_ctx.get("distribution")

    player_proj = {
        "league": request.league,
        "stat_key": stat_key,
        "mean": mean_f,
        "variance": std_f ** 2,
        "market_line": request.line,
        "distribution": distribution_override,
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

    notes: list[str] = []
    resolved_dist = select_distribution(
        stat_key, request.league, mean=mean_f, override=distribution_override,
    )
    if distribution_override in {"normal", "poisson"}:
        notes.append(f"distribution_override:{resolved_dist}")

    # B2: imputation provenance — LLM declares which observation slots are
    # imputed and (ideally) the underlying sample size. We cap tiers based on
    # the imputed fraction so a 4-of-5-imputed std cannot ride an A-tier edge.
    raw_imputed = player_ctx.get("imputed_keys") or []
    imputed_keys = [str(k) for k in raw_imputed if k is not None]
    sample_size_raw = player_ctx.get("sample_size")
    try:
        sample_size = int(sample_size_raw) if sample_size_raw is not None else None
    except (TypeError, ValueError):
        sample_size = None

    imputed_fraction: float | None
    if imputed_keys and (sample_size is None or sample_size <= 0):
        imputed_fraction = 1.0
        notes.append("imputed_keys_provided_without_sample_size")
    elif imputed_keys:
        assert sample_size is not None
        imputed_fraction = min(1.0, len(imputed_keys) / float(sample_size))
    else:
        imputed_fraction = 0.0

    # B4: single-side odds — "implied opposite" is forbidden. If only one
    # side is sourced, compute that side's edge only and annotate the other.
    edge_over: float | None = None
    edge_under: float | None = None
    recommendation = "pass"
    tier: str | None = None
    over_audit: CalibrationAudit | None = None
    under_audit: CalibrationAudit | None = None

    _ctx_hints = request.game_context or None
    if request.odds_over is not None:
        market_over = implied_probability(request.odds_over)
        cal_over, over_audit = _calibrate_audited(over_prob, league=request.league, context_hints=_ctx_hints, plane="prop", market="over")
        edge_over = round(edge_percentage(cal_over, market_over), 2)
    else:
        notes.append("odds_unsourced_over")

    if request.odds_under is not None:
        market_under = implied_probability(request.odds_under)
        cal_under, under_audit = _calibrate_audited(under_prob, league=request.league, context_hints=_ctx_hints, plane="prop", market="under")
        edge_under = round(edge_percentage(cal_under, market_under), 2)
    else:
        notes.append("odds_unsourced_under")

    if edge_over is not None and edge_under is not None:
        if edge_over > 3.0 and edge_over > edge_under:
            recommendation = "over"
            tier = "A" if request.n_iterations >= 1000 else "B"
        elif edge_under > 3.0:
            recommendation = "under"
            tier = "A" if request.n_iterations >= 1000 else "B"
    elif edge_over is not None and edge_over > 3.0:
        recommendation = "over"
        tier = "A" if request.n_iterations >= 1000 else "B"
    elif edge_under is not None and edge_under > 3.0:
        recommendation = "under"
        tier = "A" if request.n_iterations >= 1000 else "B"

    # B2 cap: imputation discipline supersedes engine tier
    if imputed_fraction is not None and imputed_fraction > 0.4:
        tier = None
        recommendation = "pass"
        notes.append("insufficient_real_observations")
    elif imputed_fraction is not None and imputed_fraction > 0.2 and tier == "A":
        tier = "B"
        notes.append("tier_capped_imputation")

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
        notes=notes,
        imputed_fraction=imputed_fraction,
        over_calibration_audit=over_audit,
        under_calibration_audit=under_audit,
    )


# ---------------------------------------------------------------------------
# analyze_slate
# ---------------------------------------------------------------------------

def analyze_slate(
    request: SlateAnalysisRequest,
    games: list[dict[str, Any]] | None = None,
) -> SlateAnalysisResponse:
    """Analyze a slate of games. Loops analyze_game per game; catches errors per-game.
    Does not fetch games; caller must supply request.games or games argument."""
    date_str = request.date or datetime.now().strftime("%Y-%m-%d")

    games = games if games is not None else request.games
    if not games:
        games = []

    analyses: list[GameAnalysisResponse] = []
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
                markets=odds_dict.get("markets"),
            )

        game_kwargs: dict[str, Any] = {
            "home_team": home,
            "away_team": away,
            "league": request.league,
            "odds": odds_input,
            "home_context": game.get("home_context"),
            "away_context": game.get("away_context"),
            "game_context": game.get("game_context"),
        }
        if game.get("n_iterations") is not None:
            game_kwargs["n_iterations"] = game.get("n_iterations")
        if game.get("seed") is not None:
            game_kwargs["seed"] = game.get("seed")

        game_request = GameAnalysisRequest(**game_kwargs)
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


def _extract_team(game: dict[str, Any], key: str) -> str | None:
    """Safely extract team name from various game dict shapes."""
    val = game.get(key)
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("name") or val.get("team") or val.get("full_name")
    return None
