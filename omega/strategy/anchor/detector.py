"""
Anchor bet detection engine.

Pure deterministic functions that take player game logs and identify
stat thresholds with high empirical hit rates. No network calls,
no randomness — just math on historical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default thresholds per stat category
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, List[float]] = {
    "pts": [10, 15, 20, 25, 30],
    "reb": [3, 5, 7, 10],
    "ast": [3, 5, 7, 10],
    "3pm": [1, 2, 3, 4],
    "stl": [1, 2],
    "blk": [1, 2],
}

# Minimum games required for a meaningful hit-rate sample
MIN_GAMES_DEFAULT = 5


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorThreshold:
    """A single stat threshold that qualifies as an anchor."""

    player_name: str
    stat_key: str
    threshold: float
    games_checked: int
    games_hit: int
    hit_rate: float       # games_hit / games_checked


@dataclass(frozen=True)
class AnchorLeg:
    """An anchor threshold matched against available prop odds."""

    player_name: str
    team: str
    stat_key: str
    threshold: float
    hit_rate: float
    games_checked: int
    odds_over: Optional[float]    # American odds for over this threshold
    implied_prob: float            # from odds (0-1)
    empirical_prob: float          # = hit_rate (0-1)
    edge_pct: float               # (empirical - implied) * 100


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def compute_hit_rate(
    values: List[float],
    threshold: float,
) -> Tuple[int, int, float]:
    """Compute how often values meet or exceed a threshold.

    Args:
        values: Recent game stat values (most recent first).
        threshold: The threshold to check (e.g., 20 for "20+ points").

    Returns:
        (games_checked, games_hit, hit_rate)
    """
    if not values:
        return (0, 0, 0.0)
    games_hit = sum(1 for v in values if v >= threshold)
    games_checked = len(values)
    hit_rate = games_hit / games_checked
    return (games_checked, games_hit, hit_rate)


def detect_anchors(
    player_name: str,
    stat_key: str,
    values: List[float],
    thresholds: Optional[List[float]] = None,
    min_hit_rate: float = 0.70,
    min_games: int = MIN_GAMES_DEFAULT,
) -> List[AnchorThreshold]:
    """Detect stat thresholds that qualify as anchors for a player.

    Args:
        player_name: Player's name.
        stat_key: Stat category (pts, reb, ast, 3pm, stl, blk).
        values: Recent game values for this stat (most recent first).
        thresholds: Thresholds to check. Defaults to DEFAULT_THRESHOLDS[stat_key].
        min_hit_rate: Minimum hit rate to qualify (default 0.70).
        min_games: Minimum games in sample (default 5).

    Returns:
        List of AnchorThreshold objects that pass the filter, sorted by
        threshold descending (highest valuable threshold first).
    """
    if len(values) < min_games:
        return []

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.get(stat_key, [])

    anchors: List[AnchorThreshold] = []
    for threshold in thresholds:
        games_checked, games_hit, hit_rate = compute_hit_rate(values, threshold)
        if hit_rate >= min_hit_rate:
            anchors.append(AnchorThreshold(
                player_name=player_name,
                stat_key=stat_key,
                threshold=threshold,
                games_checked=games_checked,
                games_hit=games_hit,
                hit_rate=hit_rate,
            ))

    # Sort by threshold descending — higher thresholds are more valuable
    anchors.sort(key=lambda a: a.threshold, reverse=True)
    return anchors


def match_anchor_to_odds(
    anchor: AnchorThreshold,
    team: str,
    odds_over: float,
) -> AnchorLeg:
    """Match an anchor threshold against available prop odds to compute edge.

    Args:
        anchor: The detected anchor threshold.
        team: Player's team.
        odds_over: American odds for the "over" side of this prop.

    Returns:
        AnchorLeg with edge calculation.
    """
    from omega.core.betting.odds import implied_probability

    impl_prob = implied_probability(odds_over)
    edge = (anchor.hit_rate - impl_prob) * 100.0

    return AnchorLeg(
        player_name=anchor.player_name,
        team=team,
        stat_key=anchor.stat_key,
        threshold=anchor.threshold,
        hit_rate=anchor.hit_rate,
        games_checked=anchor.games_checked,
        odds_over=odds_over,
        implied_prob=round(impl_prob, 4),
        empirical_prob=anchor.hit_rate,
        edge_pct=round(edge, 2),
    )


def scan_player(
    player_name: str,
    team: str,
    game_logs: Dict[str, List[float]],
    prop_odds: Optional[Dict[str, Dict[float, float]]] = None,
    min_hit_rate: float = 0.70,
    min_games: int = MIN_GAMES_DEFAULT,
) -> List[AnchorLeg]:
    """Scan a single player for all anchor opportunities.

    Args:
        player_name: Player's name.
        team: Player's team.
        game_logs: {stat_key: [recent values]} e.g. {"pts": [25, 30, 22, ...]}
        prop_odds: {stat_key: {threshold: american_odds}} e.g. {"pts": {20: -250, 25: +110}}
            If None, returns anchors without odds/edge (odds_over=None).
        min_hit_rate: Minimum hit rate to qualify.
        min_games: Minimum games in sample.

    Returns:
        List of AnchorLeg objects across all stat categories.
    """
    all_legs: List[AnchorLeg] = []

    for stat_key, values in game_logs.items():
        anchors = detect_anchors(
            player_name=player_name,
            stat_key=stat_key,
            values=values,
            min_hit_rate=min_hit_rate,
            min_games=min_games,
        )

        for anchor in anchors:
            # Try to match against available odds
            if prop_odds and stat_key in prop_odds:
                odds_map = prop_odds[stat_key]
                # Look for exact threshold match or closest lower threshold
                if anchor.threshold in odds_map:
                    leg = match_anchor_to_odds(
                        anchor, team, odds_map[anchor.threshold]
                    )
                    all_legs.append(leg)
                # Sportsbook thresholds use X+ format matching our thresholds
                # If no exact match, check for threshold - 0.5 (e.g., 19.5 for 20+)
                elif (anchor.threshold - 0.5) in odds_map:
                    leg = match_anchor_to_odds(
                        anchor, team, odds_map[anchor.threshold - 0.5]
                    )
                    all_legs.append(leg)
            else:
                # No odds available — still report the anchor without edge
                all_legs.append(AnchorLeg(
                    player_name=player_name,
                    team=team,
                    stat_key=stat_key,
                    threshold=anchor.threshold,
                    hit_rate=anchor.hit_rate,
                    games_checked=anchor.games_checked,
                    odds_over=None,
                    implied_prob=0.0,
                    empirical_prob=anchor.hit_rate,
                    edge_pct=0.0,
                ))

    return all_legs
