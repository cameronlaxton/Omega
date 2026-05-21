"""
Anchor parlay scanner.

Scans a slate of games to find high-probability player prop anchors,
then builds optimal 2-4 leg same-game parlays. Deterministic: same inputs
always produce same outputs.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from omega.core.betting.kelly import recommend_stake
from omega.core.betting.odds import american_to_decimal
from omega.core.betting.parlay import ParlayLeg, build_parlay
from omega.strategy.anchor.detector import AnchorLeg, scan_player

UTC = timezone.utc

logger = logging.getLogger("omega.strategy.anchor.scanner")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AnchorParlayConfig:
    """Configuration for the anchor parlay scanner."""

    league: str = "NBA"
    min_hit_rate: float = 0.70
    games_lookback: int = 10
    min_legs: int = 2
    max_legs: int = 4
    target_min_odds: float = 1.80  # decimal
    target_max_odds: float = 3.00  # decimal
    min_leg_edge: float = -5.0  # allow slightly negative individual edge
    min_parlay_ev: float = 0.0  # minimum combined EV% to include
    correlation_discount: float = 0.95  # multiply combined prob when correlated
    bankroll: float = 1000.0
    confidence_tier: str = "B"
    max_results: int = 20


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


@dataclass
class AnchorParlay:
    """A single recommended parlay."""

    game: str  # e.g. "Thunder @ Clippers"
    legs: list[AnchorLeg]
    combined_decimal_odds: float
    combined_hit_rate: float  # empirical
    implied_probability: float  # from combined odds
    combined_edge_pct: float
    ev_pct: float
    correlation_warnings: list[str]
    recommended_units: float
    kelly_fraction: float
    confidence_tier: str


@dataclass
class ScanResult:
    """Complete output from a slate scan."""

    league: str
    date: str
    games_scanned: int
    players_scanned: int
    anchors_found: int
    parlays_built: int
    parlays: list[AnchorParlay]
    scan_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def _anchor_to_parlay_leg(leg: AnchorLeg) -> ParlayLeg | None:
    """Convert an AnchorLeg (with odds) to a ParlayLeg for parlay math."""
    if leg.odds_over is None:
        return None
    return ParlayLeg(
        selection=f"{leg.player_name} {leg.threshold}+ {leg.stat_key}",
        decimal_odds=american_to_decimal(leg.odds_over),
        win_probability=leg.hit_rate,
        player=leg.player_name,
        stat_key=leg.stat_key,
        team=leg.team,
    )


def build_parlays_for_game(
    anchor_legs: list[AnchorLeg],
    game_label: str,
    config: AnchorParlayConfig,
) -> list[AnchorParlay]:
    """Build all valid parlay combinations from anchor legs within one game.

    Args:
        anchor_legs: All anchor legs found for players in this game.
        game_label: Display string e.g. "Thunder @ Clippers".
        config: Scanner configuration.

    Returns:
        List of AnchorParlay objects, sorted by EV descending.
    """
    # Filter to legs with odds and convert
    parlay_ready = []
    for aleg in anchor_legs:
        pleg = _anchor_to_parlay_leg(aleg)
        if pleg is not None:
            parlay_ready.append((aleg, pleg))

    if len(parlay_ready) < config.min_legs:
        return []

    results: list[AnchorParlay] = []

    for n_legs in range(config.min_legs, min(config.max_legs, len(parlay_ready)) + 1):
        for combo in itertools.combinations(parlay_ready, n_legs):
            alegs = [c[0] for c in combo]
            plegs = [c[1] for c in combo]

            # Build the parlay slip
            slip = build_parlay(plegs)

            # Apply correlation discount if warnings exist
            discount = 1.0
            if slip.correlation_warnings:
                discount = config.correlation_discount ** len(slip.correlation_warnings)
                slip = build_parlay(plegs, independence_discount=discount)

            # Filter: odds in target range
            if slip.combined_decimal_odds < config.target_min_odds:
                continue
            if slip.combined_decimal_odds > config.target_max_odds:
                continue

            # Filter: minimum EV
            if slip.ev_pct < config.min_parlay_ev:
                continue

            # Compute stake recommendation
            # Convert combined decimal odds to American for Kelly
            if slip.combined_decimal_odds >= 2.0:
                american_equiv = (slip.combined_decimal_odds - 1) * 100
            else:
                american_equiv = -100 / (slip.combined_decimal_odds - 1)

            stake = recommend_stake(
                true_prob=slip.combined_win_probability,
                odds=american_equiv,
                bankroll=config.bankroll,
                confidence_tier=config.confidence_tier,
            )

            results.append(
                AnchorParlay(
                    game=game_label,
                    legs=alegs,
                    combined_decimal_odds=slip.combined_decimal_odds,
                    combined_hit_rate=slip.combined_win_probability,
                    implied_probability=slip.implied_probability,
                    combined_edge_pct=slip.combined_edge_pct,
                    ev_pct=slip.ev_pct,
                    correlation_warnings=slip.correlation_warnings,
                    recommended_units=stake["units"],
                    kelly_fraction=stake["kelly_fraction"],
                    confidence_tier=config.confidence_tier,
                )
            )

    # Sort by EV descending
    results.sort(key=lambda p: p.ev_pct, reverse=True)
    return results


def run_full_scan(
    league: str = "NBA",
    config: AnchorParlayConfig | None = None,
) -> ScanResult:
    """Run the full anchor parlay pipeline: gather evidence → detect → build → rank.

    This is the single entry point that orchestrates data collection and scanning.
    It calls gather_slate_data() for live evidence, then scan_slate() for analysis.

    Args:
        league: League to scan (default "NBA").
        config: Scanner configuration. Uses defaults if None.

    Returns:
        ScanResult with ranked parlays, or empty result if no games found.
    """
    from omega.strategy.anchor.gather import gather_slate_data

    if config is None:
        config = AnchorParlayConfig(league=league)

    logger.info("Starting full anchor scan for %s", league)

    # Phase 1: Gather evidence (schedule, game logs, prop odds)
    games_data = gather_slate_data(league=league)

    if not games_data:
        logger.warning("No games or data found for %s — returning empty result", league)
        return ScanResult(
            league=league,
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            games_scanned=0,
            players_scanned=0,
            anchors_found=0,
            parlays_built=0,
            parlays=[],
            scan_metadata={"error": f"No games found for {league} today"},
        )

    # Phase 2: Scan the gathered data
    result = scan_slate(games_data, config=config)

    logger.info(
        "Scan complete: %d games, %d players, %d anchors, %d parlays",
        result.games_scanned,
        result.players_scanned,
        result.anchors_found,
        result.parlays_built,
    )
    return result


def scan_slate(
    games: list[dict[str, Any]],
    config: AnchorParlayConfig | None = None,
) -> ScanResult:
    """Scan a full slate of games for anchor parlay opportunities.

    Args:
        games: List of game dicts, each containing:
            - "game_label": str (e.g. "Thunder @ Clippers")
            - "players": list of player dicts, each containing:
                - "name": str
                - "team": str
                - "game_logs": {stat_key: [values]}
                - "prop_odds": {stat_key: {threshold: american_odds}} (optional)
        config: Scanner configuration. Uses defaults if None.

    Returns:
        ScanResult with ranked parlays.
    """
    if config is None:
        config = AnchorParlayConfig()

    all_parlays: list[AnchorParlay] = []
    total_players = 0
    total_anchors = 0

    for game in games:
        game_label = game.get("game_label", "Unknown Game")
        players = game.get("players", [])
        game_anchor_legs: list[AnchorLeg] = []

        for player_data in players:
            total_players += 1
            name = player_data["name"]
            team = player_data["team"]
            game_logs = player_data.get("game_logs", {})
            prop_odds = player_data.get("prop_odds")

            legs = scan_player(
                player_name=name,
                team=team,
                game_logs=game_logs,
                prop_odds=prop_odds,
                min_hit_rate=config.min_hit_rate,
                min_games=max(1, config.games_lookback // 2),  # require at least half the lookback
            )
            total_anchors += len(legs)
            game_anchor_legs.extend(legs)

        # Build parlays for this game
        game_parlays = build_parlays_for_game(game_anchor_legs, game_label, config)
        all_parlays.extend(game_parlays)

    # Sort all parlays by EV and limit
    all_parlays.sort(key=lambda p: p.ev_pct, reverse=True)
    all_parlays = all_parlays[: config.max_results]

    return ScanResult(
        league=config.league,
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        games_scanned=len(games),
        players_scanned=total_players,
        anchors_found=total_anchors,
        parlays_built=len(all_parlays),
        parlays=all_parlays,
        scan_metadata={
            "config": {
                "min_hit_rate": config.min_hit_rate,
                "games_lookback": config.games_lookback,
                "min_legs": config.min_legs,
                "max_legs": config.max_legs,
                "target_odds_range": [config.target_min_odds, config.target_max_odds],
            },
        },
    )
