"""
Evidence gathering for the anchor parlay scanner.

Orchestrates data collection using the existing collector registry.
Primary path is LLM-powered web search (Perplexity/Anthropic) for both
player game logs and prop odds. ESPN is used as a tier-1 accelerator
for schedules and game logs when available.

No OddsAPI dependency — all prop odds come through web search.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from omega.evidence.collectors.base import CollectorResult
from omega.evidence.registry import get_default_registry

logger = logging.getLogger("omega.strategy.anchor.gather")

# Stat keys we care about for anchor detection
ANCHOR_STAT_KEYS = ["pts", "reb", "ast", "3pm", "stl", "blk"]


def gather_schedule(league: str) -> List[Dict[str, Any]]:
    """Fetch today's game schedule via the collector registry.

    Returns list of game dicts with home_team, away_team, game_id.
    """
    registry = get_default_registry()
    collectors = registry.get_collectors_for("schedule", league)

    for collector in collectors:
        result = collector.collect(entity=league, league=league, data_type="schedule")
        if result and result.data:
            games = result.data.get("games", [])
            if games:
                return games

    logger.warning("No schedule data available for %s", league)
    return []


def gather_player_game_log(
    player_name: str,
    league: str,
) -> Optional[Dict[str, List[float]]]:
    """Fetch recent game logs for a player and extract stat arrays.

    Uses the collector registry (ESPN tier-1, web search tier-2).

    Returns:
        Dict mapping stat_key -> list of recent values, e.g.:
        {"pts": [28, 25, 32, ...], "reb": [5, 7, 4, ...], "ast": [6, 8, 5, ...]}
        Returns None if no data found.
    """
    registry = get_default_registry()
    collectors = registry.get_collectors_for("player_game_log", league)

    for collector in collectors:
        result = collector.collect(
            entity=player_name, league=league, data_type="player_game_log"
        )
        if result and result.data:
            return _extract_stat_arrays(result.data)

    logger.debug("No game log data for %s (%s)", player_name, league)
    return None


def gather_player_prop_odds(
    game_label: str,
    league: str,
) -> Dict[str, Dict[str, Dict[float, float]]]:
    """Fetch player prop odds for a game via web search.

    Returns:
        Nested dict: {player_name: {stat_key: {threshold: american_odds}}}
        e.g.: {"SGA": {"ast": {5: -200, 7: +150}}}
    """
    registry = get_default_registry()
    collectors = registry.get_collectors_for("player_prop_odds", league)

    for collector in collectors:
        result = collector.collect(
            entity=game_label, league=league, data_type="player_prop_odds"
        )
        if result and result.data:
            return _extract_prop_odds(result.data)

    logger.debug("No prop odds data for %s (%s)", game_label, league)
    return {}


def gather_slate_data(
    league: str = "NBA",
) -> List[Dict[str, Any]]:
    """Gather all data needed for an anchor parlay scan.

    Orchestrates:
    1. Fetch today's schedule
    2. For each game, identify players (from roster or search)
    3. Fetch game logs for key players
    4. Fetch prop odds per game

    Returns:
        List of game dicts ready for scan_slate(), each containing:
        - game_label: str
        - players: list of {name, team, game_logs, prop_odds}
    """
    schedule = gather_schedule(league)
    if not schedule:
        logger.warning("No games found for %s today", league)
        return []

    games_data: List[Dict[str, Any]] = []

    for game in schedule:
        home = game.get("home_team", {})
        away = game.get("away_team", {})

        home_name = home.get("name", "") if isinstance(home, dict) else str(home)
        away_name = away.get("name", "") if isinstance(away, dict) else str(away)
        game_label = f"{away_name} @ {home_name}"

        # Get prop odds for this game
        prop_odds_by_player = gather_player_prop_odds(game_label, league)

        # Gather game logs for players we have prop odds for
        players_data: List[Dict[str, Any]] = []
        for player_name in prop_odds_by_player:
            game_logs = gather_player_game_log(player_name, league)
            if game_logs:
                players_data.append({
                    "name": player_name,
                    "team": _guess_team(player_name, home_name, away_name),
                    "game_logs": game_logs,
                    "prop_odds": prop_odds_by_player.get(player_name),
                })

        games_data.append({
            "game_label": game_label,
            "players": players_data,
        })

    return games_data


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_stat_arrays(data: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    """Extract stat arrays from collector result data.

    Handles both structured ESPN game log format and LLM-extracted prose.
    """
    # Structured format: {"games": [{"pts": 28, "reb": 5, ...}, ...]}
    games = data.get("games", [])
    if games and isinstance(games, list):
        stat_arrays: Dict[str, List[float]] = {}
        for stat_key in ANCHOR_STAT_KEYS:
            values = []
            for game in games:
                val = game.get(stat_key)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass
            if values:
                stat_arrays[stat_key] = values
        if stat_arrays:
            return stat_arrays

    # LLM prose format: try to parse from _raw_text
    raw_text = data.get("_raw_text", "")
    if raw_text:
        return _parse_game_log_from_prose(raw_text)

    # Flat format: data might directly contain stat arrays
    stat_arrays = {}
    for stat_key in ANCHOR_STAT_KEYS:
        if stat_key in data and isinstance(data[stat_key], list):
            stat_arrays[stat_key] = [float(v) for v in data[stat_key] if v is not None]
    if stat_arrays:
        return stat_arrays

    return None


def _parse_game_log_from_prose(text: str) -> Optional[Dict[str, List[float]]]:
    """Best-effort extraction of game log stats from LLM prose.

    Looks for patterns like "25 points, 8 rebounds, 6 assists" across
    multiple game entries in the text.
    """
    stat_arrays: Dict[str, List[float]] = {k: [] for k in ANCHOR_STAT_KEYS}

    # Pattern: "N points" or "N pts" etc.
    stat_patterns = {
        "pts": r"(\d+)\s*(?:points|pts|pt)",
        "reb": r"(\d+)\s*(?:rebounds|reb|boards)",
        "ast": r"(\d+)\s*(?:assists|ast)",
        "3pm": r"(\d+)\s*(?:three[- ]pointers?|3pm|3pt|threes|3-pointers?)",
        "stl": r"(\d+)\s*(?:steals?|stl)",
        "blk": r"(\d+)\s*(?:blocks?|blk)",
    }

    for stat_key, pattern in stat_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        stat_arrays[stat_key] = [float(m) for m in matches]

    # Only return if we got meaningful data
    non_empty = {k: v for k, v in stat_arrays.items() if v}
    return non_empty if non_empty else None


def _extract_prop_odds(data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[float, float]]]:
    """Extract player prop odds from collector result data.

    Handles both structured format and LLM-extracted data.

    Returns:
        {player_name: {stat_key: {threshold: american_odds}}}
    """
    result: Dict[str, Dict[str, Dict[float, float]]] = {}

    # Structured format: list of prop entries
    props = data.get("props", data.get("player_props", []))
    if isinstance(props, list):
        for prop in props:
            player = prop.get("player", "")
            stat_key = prop.get("stat_key", "")
            line = prop.get("line")
            odds_over = prop.get("odds_over")
            if player and stat_key and line is not None and odds_over is not None:
                result.setdefault(player, {}).setdefault(stat_key, {})[float(line)] = float(odds_over)
        if result:
            return result

    # Try parsing from raw text
    raw_text = data.get("_raw_text", "")
    if raw_text:
        return _parse_prop_odds_from_prose(raw_text)

    return result


def _parse_prop_odds_from_prose(
    text: str,
) -> Dict[str, Dict[str, Dict[float, float]]]:
    """Best-effort extraction of prop odds from LLM prose.

    Looks for patterns like "Player Name: Over 20.5 points (-115)"
    """
    result: Dict[str, Dict[str, Dict[float, float]]] = {}

    # Common patterns for prop lines in prose
    # "PlayerName Over X.5 stat (-110)" or "PlayerName X+ stat -110"
    patterns = [
        # "Player Name Over 20.5 points (-115)"
        r"([A-Z][a-z]+(?: [A-Z][a-z'-]+)+)\s+[Oo]ver\s+(\d+\.?\d*)\s+"
        r"(points?|rebounds?|assists?|three[- ]pointers?|steals?|blocks?)"
        r"\s*\(?\s*([+-]\d+)\s*\)?",
        # "Player Name 20+ points -250"
        r"([A-Z][a-z]+(?: [A-Z][a-z'-]+)+)\s+(\d+)\+\s+"
        r"(points?|rebounds?|assists?|three[- ]pointers?|steals?|blocks?)"
        r"\s*[:\s]*\(?\s*([+-]\d+)\s*\)?",
    ]

    stat_name_map = {
        "points": "pts", "point": "pts", "pts": "pts",
        "rebounds": "reb", "rebound": "reb", "reb": "reb",
        "assists": "ast", "assist": "ast", "ast": "ast",
        "three-pointers": "3pm", "three pointers": "3pm", "threes": "3pm",
        "steals": "stl", "steal": "stl",
        "blocks": "blk", "block": "blk",
    }

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            player = match.group(1).strip()
            threshold = float(match.group(2))
            stat_raw = match.group(3).lower().strip()
            odds = float(match.group(4))

            stat_key = stat_name_map.get(stat_raw, "")
            if stat_key and player:
                result.setdefault(player, {}).setdefault(stat_key, {})[threshold] = odds

    return result


def _guess_team(
    player_name: str,
    home_team: str,
    away_team: str,
) -> str:
    """Best-effort team assignment. Returns home_team as fallback."""
    # In a real implementation, we'd look up the player's team from roster data.
    # For now, return a placeholder that the scanner can work with.
    return home_team
