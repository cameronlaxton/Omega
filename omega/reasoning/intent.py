"""
Intent Understanding — multi-dimensional classification of user prompts.

Produces a QueryUnderstanding that captures Subject, Goal, entities,
constraints, and tone along independent dimensions. Uses LLM function-calling
when available, with a heuristic fallback that works without any LLM.

The heuristic parser defaults wants_betting_advice=False unless the prompt
contains strong betting signals. Betting advice requires an affirmative
signal, not a default assumption.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from omega.core.models import (
    Entity,
    EntityRole,
    QueryUnderstanding,
    Subject,
    UserGoal,
)

if TYPE_CHECKING:
    from omega.reasoning.llm.client import LLMClient


def _unwrap_scalar(value: Any) -> Any:
    """Unwrap a single-element list to its scalar value.

    LLMs sometimes return ["analyze"] instead of "analyze" for enum fields.
    """
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value

logger = logging.getLogger("omega.agent.intent")

# ---------------------------------------------------------------------------
# League keyword mapping
# ---------------------------------------------------------------------------

LEAGUE_KEYWORDS = {
    "nba": "NBA", "basketball": "NBA", "hoops": "NBA",
    "wnba": "WNBA",
    "ncaab": "NCAAB", "college basketball": "NCAAB", "march madness": "NCAAB",
    "nfl": "NFL", "football": "NFL",
    "ncaaf": "NCAAF", "college football": "NCAAF", "cfb": "NCAAF",
    "mlb": "MLB", "baseball": "MLB",
    "nhl": "NHL", "hockey": "NHL",
    "mls": "MLS", "soccer": "EPL", "premier league": "EPL",
    "epl": "EPL", "la liga": "LA_LIGA", "bundesliga": "BUNDESLIGA",
    "serie a": "SERIE_A", "ligue 1": "LIGUE_1", "champions league": "CHAMPIONS_LEAGUE",
    "ufc": "UFC", "mma": "UFC", "boxing": "BOXING", "fight": "UFC",
    "atp": "ATP", "wta": "WTA", "tennis": "ATP",
    "pga": "PGA", "golf": "PGA", "lpga": "LPGA",
    "cs2": "CS2", "csgo": "CS2", "esports": "ESPORTS",
    "valorant": "VALORANT", "league of legends": "LOL", "lol": "LOL",
    "dota": "DOTA2",
}

# Betting intent signals — presence of these implies wants_betting_advice=True
BETTING_SIGNALS = {
    "should i bet", "is that a play", "is this a play", "worth a bet",
    "edge", "units", "stake", "wager", "bet on", "play this",
    "sharp", "value", "ev", "expected value", "kelly",
    "hammer", "lock", "parlay", "sgp", "same game",
    "how should i play", "what's the play", "any plays",
}

# Goal signals
GOAL_SIGNALS = {
    "decide": ["should i", "is it worth", "play this", "bet this", "is that a play"],
    "analyze": ["analyze", "break down", "breakdown", "deep dive"],
    "explain": ["why", "how come", "what happened", "explain"],
    "compare": ["compare", "versus", "which is better", "difference between"],
    "discuss": ["what do you think", "thoughts on", "opinion", "take on"],
    "summarize": ["catch me up", "summary", "what's happening", "recap", "rundown"],
    "learn": ["how does", "what is", "teach me", "explain how", "definition of"],
    "monitor": ["any value", "any edges", "what's on the board", "scan", "find me"],
}

# Subject signals
SUBJECT_SIGNALS_BANKROLL = {"bankroll", "aggressive", "conservative", "risk", "sizing", "how much", "units"}
SUBJECT_SIGNALS_PROP = {"over", "under", "points", "rebounds", "assists", "yards", "prop", "o/u"}
SUBJECT_SIGNALS_SLATE = {"slate", "all games", "tonight's games", "today's games", "full card", "board"}


# ---------------------------------------------------------------------------
# Heuristic parser (no LLM required)
# ---------------------------------------------------------------------------

def _detect_league(prompt_lower: str) -> Optional[str]:
    """Detect league from keywords in the prompt.

    Checks longer (more specific) keywords first so that "college football"
    matches NCAAF before "football" matches NFL.
    """
    # Sort by keyword length descending — longer matches are more specific
    for keyword, league in sorted(LEAGUE_KEYWORDS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if keyword in prompt_lower:
            return league
    return None


def _extract_teams(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract two team names from 'X vs Y' or 'X at Y' patterns."""
    patterns = [
        r"(.+?)\s+(?:vs\.?|versus|v\.?)\s+(.+)",
        r"(.+?)\s+(?:at|@)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            away = match.group(1).strip().rstrip("?.,!")
            home = match.group(2).strip().rstrip("?.,!")
            # Clean up: remove trailing league keywords
            for kw in LEAGUE_KEYWORDS:
                away = re.sub(rf"\s+{re.escape(kw)}$", "", away, flags=re.IGNORECASE)
                home = re.sub(rf"\s+{re.escape(kw)}$", "", home, flags=re.IGNORECASE)
            return home, away
    return None, None


def _detect_goal(prompt_lower: str) -> UserGoal:
    """Detect the user's goal from signal phrases."""
    for goal, signals in GOAL_SIGNALS.items():
        for signal in signals:
            if signal in prompt_lower:
                return UserGoal(goal)
    return UserGoal.ANALYZE  # default


def _detect_betting_intent(prompt_lower: str) -> bool:
    """Returns True only if there are strong betting signals in the prompt."""
    return any(signal in prompt_lower for signal in BETTING_SIGNALS)


def _detect_subjects(prompt_lower: str, home: Optional[str], away: Optional[str],
                     has_prop_signals: bool, has_slate_signals: bool,
                     has_bankroll_signals: bool, league: Optional[str],
                     has_player_entity: bool = False) -> List[Subject]:
    """Determine which subjects the prompt covers."""
    subjects = []

    if has_bankroll_signals:
        subjects.append(Subject.BANKROLL)

    if has_prop_signals:
        subjects.append(Subject.PLAYER_PROP)
    elif has_slate_signals:
        subjects.append(Subject.SLATE)
    elif home and away:
        subjects.append(Subject.GAME)
    elif has_player_entity:
        # Player mentioned without vs/at pattern — still a game context query
        subjects.append(Subject.GAME)
    elif league:
        # Single team mentioned or just a league reference
        subjects.append(Subject.GAME)

    # If no specific subject detected, infer from context
    if not subjects:
        if league:
            subjects.append(Subject.GENERAL_SPORTS)
        else:
            subjects.append(Subject.GENERAL_SPORTS)

    return subjects


def _extract_prop_details(prompt_lower: str) -> Tuple[Optional[str], Optional[float]]:
    """Extract prop type and line from prompt."""
    # Pattern: "over 25.5 points" or "under 7.5 rebounds"
    prop_pattern = r"(?:over|under)\s+([\d.]+)\s+(\w+)"
    match = re.search(prop_pattern, prompt_lower)
    if match:
        line = float(match.group(1))
        stat = match.group(2).rstrip("s")  # "points" → "point"
        stat_map = {
            "point": "pts", "rebound": "reb", "assist": "ast",
            "three": "3pm", "steal": "stl", "block": "blk",
            "yard": "pass_yds", "reception": "receptions",
            "touchdown": "td", "goal": "goals", "ace": "aces",
            "kill": "kills", "save": "saves", "shot": "shots",
        }
        return stat_map.get(stat, stat), line

    # Pattern: "LeBron 25.5 points" or "player_name o25.5 pts"
    line_pattern = r"[uo]?([\d.]+)\s*(?:pts|points|reb|rebounds|ast|assists|yds|yards)"
    match = re.search(line_pattern, prompt_lower)
    if match:
        return None, float(match.group(1))

    return None, None


# Common words that look like player names but aren't
_NON_PLAYER_WORDS = {
    "the", "how", "what", "who", "which", "when", "where", "why",
    "this", "that", "these", "those", "any", "some", "all",
    "monte carlo", "kelly criterion", "game log", "super bowl",
    "world series", "stanley cup", "march madness", "final four",
}


def _extract_player_entities(prompt: str, league: Optional[str],
                              existing_entities: List[Entity]) -> List[Entity]:
    """Detect player name mentions not caught by the vs/at team extraction.

    Only fires when no entities were found by _extract_teams(). Looks for
    2-3 consecutive capitalized words that aren't league keywords or team names.
    """
    if existing_entities:
        return []  # vs/at pattern already handled entity extraction

    # Find sequences of 2-3 words starting with uppercase (proper nouns / player names)
    # Handles names like "LeBron James", "De'Aaron Fox", etc.
    matches = re.findall(r"\b([A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+){1,2})\b", prompt)
    if not matches:
        return []

    # Filter out league keywords and common non-player phrases
    league_names = set(LEAGUE_KEYWORDS.values())
    league_kw_title = {kw.title() for kw in LEAGUE_KEYWORDS}

    players: List[Entity] = []
    for name in matches:
        name_lower = name.lower()
        if name_lower in _NON_PLAYER_WORDS:
            continue
        if name in league_names or name in league_kw_title:
            continue
        # Skip if it matches a league keyword substring
        if any(name_lower == kw for kw in LEAGUE_KEYWORDS):
            continue
        players.append(Entity(
            name=name,
            role=EntityRole.SUBJECT,
            entity_type="player",
        ))
        break  # Take only the first player match to avoid false positives

    return players


def _detect_tone(prompt_lower: str) -> str:
    """Infer desired response tone."""
    if any(w in prompt_lower for w in ("quick", "brief", "short", "tldr", "tl;dr")):
        return "brief"
    if any(w in prompt_lower for w in ("think", "opinion", "feel", "vibe")):
        return "conversational"
    return "analytical"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_heuristic(prompt: str) -> QueryUnderstanding:
    """Parse a user prompt into a QueryUnderstanding using heuristics only.

    This is the fallback path when no LLM is available. It works well for
    common patterns but misses nuance on ambiguous or complex prompts.
    """
    prompt_lower = prompt.lower().strip()

    league = _detect_league(prompt_lower)
    home, away = _extract_teams(prompt)
    goal = _detect_goal(prompt_lower)
    wants_betting = _detect_betting_intent(prompt_lower)
    tone = _detect_tone(prompt_lower)

    has_prop_signals = any(s in prompt_lower for s in SUBJECT_SIGNALS_PROP)
    has_slate_signals = any(s in prompt_lower for s in SUBJECT_SIGNALS_SLATE)
    has_bankroll_signals = any(s in prompt_lower for s in SUBJECT_SIGNALS_BANKROLL)

    # Build entities
    entities: List[Entity] = []
    if home:
        entities.append(Entity(name=home, role=EntityRole.HOME))
    if away:
        entities.append(Entity(name=away, role=EntityRole.AWAY))

    # Try to extract player entities if no teams were found
    player_entities = _extract_player_entities(prompt, league, entities)
    entities.extend(player_entities)

    subjects = _detect_subjects(
        prompt_lower, home, away, has_prop_signals,
        has_slate_signals, has_bankroll_signals, league,
        has_player_entity=bool(player_entities),
    )

    # Extract prop details
    prop_type, prop_line = None, None
    if has_prop_signals:
        prop_type, prop_line = _extract_prop_details(prompt_lower)

    # Detect explicit constraints
    constraints: List[str] = []
    if "no bet" in prompt_lower or "not looking for bet" in prompt_lower:
        constraints.append("no_bets")
        wants_betting = False
    if "just analysis" in prompt_lower or "just key factors" in prompt_lower:
        constraints.append("analysis_only")
        wants_betting = False

    wants_explanation = goal in (UserGoal.EXPLAIN, UserGoal.LEARN)
    wants_alternatives = "alternative" in prompt_lower or "other" in prompt_lower

    return QueryUnderstanding(
        subjects=subjects,
        league=league,
        entities=entities,
        markets=[],
        prop_type=prop_type,
        prop_line=prop_line,
        date=date.today().isoformat(),
        goal=goal,
        wants_betting_advice=wants_betting,
        wants_explanation=wants_explanation,
        wants_alternatives=wants_alternatives,
        tone=tone,
        explicit_constraints=constraints,
        raw_prompt=prompt,
    )


# ---------------------------------------------------------------------------
# LLM tool schema for intent classification
# ---------------------------------------------------------------------------

CLASSIFY_QUERY_TOOL: Dict[str, Any] = {
    "name": "classify_query",
    "description": (
        "Classify a sports betting query into structured dimensions. "
        "Extract subjects, entities, league, goal, and betting intent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subjects": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [s.value for s in Subject],
                },
                "description": "What the prompt is about: game, player_prop, slate, comparison, bankroll, news_context, general_sports",
            },
            "league": {
                "type": "string",
                "nullable": True,
                "description": "League code (NBA, NFL, MLB, NHL, EPL, UFC, ATP, PGA, CS2, etc.) or null if ambiguous",
            },
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Team or player name"},
                        "role": {"type": "string", "enum": [r.value for r in EntityRole]},
                        "entity_type": {"type": "string", "enum": ["team", "player"]},
                    },
                    "required": ["name", "role", "entity_type"],
                },
                "description": "Teams or players mentioned in the query",
            },
            "goal": {
                "type": "string",
                "enum": [g.value for g in UserGoal],
                "description": "User's goal: decide, analyze, compare, explain, discuss, summarize, learn, monitor",
            },
            "wants_betting_advice": {
                "type": "boolean",
                "description": "True only if the user explicitly wants betting recommendations (edges, units, plays)",
            },
            "wants_explanation": {"type": "boolean"},
            "wants_alternatives": {"type": "boolean"},
            "tone": {
                "type": "string",
                "enum": ["analytical", "conversational", "brief"],
            },
            "prop_type": {
                "type": "string",
                "nullable": True,
                "description": "Stat type for player props: pts, reb, ast, 3pm, pass_yds, rush_yds, etc.",
            },
            "prop_line": {
                "type": "number",
                "nullable": True,
                "description": "The over/under line for a player prop",
            },
            "explicit_constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit constraints like 'no_bets', 'analysis_only'",
            },
        },
        "required": ["subjects", "goal", "wants_betting_advice", "tone"],
    },
}

INTENT_SYSTEM_PROMPT = """You are a sports betting query classifier for OmegaSportsAgent, a quantitative sports analytics engine.

Classify the user's query into structured dimensions:
- **subjects**: What is this about? (game matchup, player prop, full slate, bankroll management, etc.)
- **league**: Which league? Use standard codes: NBA, NFL, MLB, NHL, EPL, MLS, UFC, ATP, PGA, CS2, NCAAB, NCAAF, etc.
- **entities**: Extract team/player names with roles (home, away, subject).
  - For "X vs Y" or "X at Y": second team is home, first is away.
  - For player props: player is "subject", their team opponent is "opponent".
- **goal**: What does the user want? (decide=should I bet, analyze=break it down, compare, explain, discuss, summarize, learn, monitor)
- **wants_betting_advice**: Only true if they explicitly want bet recommendations (edges, units, plays). Analysis alone is NOT betting advice.
- **tone**: analytical (default), conversational (casual/opinion), brief (quick answer)
- **prop_type/prop_line**: For player props, extract the stat type and line.

Today's date: """ + date.today().isoformat()


def _llm_classify(prompt: str, llm_client: "LLMClient") -> Optional[QueryUnderstanding]:
    """Attempt LLM-based intent classification via tool use."""
    result = llm_client.call_with_tools(
        system=INTENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[CLASSIFY_QUERY_TOOL],
    )
    if result is None:
        return None

    try:
        # Build entities from LLM output
        entities = []
        for ent in result.get("entities", []):
            entities.append(Entity(
                name=ent["name"],
                role=EntityRole(ent["role"]),
                entity_type=ent.get("entity_type", "team"),
            ))

        return QueryUnderstanding(
            subjects=[Subject(s) for s in result.get("subjects", ["general_sports"])],
            league=result.get("league"),
            entities=entities,
            markets=[],
            prop_type=result.get("prop_type"),
            prop_line=result.get("prop_line"),
            date=date.today().isoformat(),
            goal=UserGoal(_unwrap_scalar(result.get("goal", "analyze"))),
            wants_betting_advice=result.get("wants_betting_advice", False),
            wants_explanation=result.get("wants_explanation", False),
            wants_alternatives=result.get("wants_alternatives", False),
            tone=result.get("tone", "analytical"),
            explicit_constraints=result.get("explicit_constraints", []),
            raw_prompt=prompt,
        )
    except (ValueError, KeyError):
        logger.warning("Failed to parse LLM intent classification result", exc_info=True)
        return None


def understand(prompt: str, llm_client: Optional["LLMClient"] = None) -> QueryUnderstanding:
    """Top-level entry point for intent understanding.

    Uses LLM function-calling when available, otherwise falls back to
    heuristic parsing. Both paths produce the same QueryUnderstanding type.

    Args:
        prompt: Raw user input string.
        llm_client: Optional LLMClient instance for LLM-based classification.

    Returns:
        QueryUnderstanding with all four dimensions classified.
    """
    if llm_client is not None and llm_client.is_available():
        result = _llm_classify(prompt, llm_client)
        if result is not None:
            logger.info("Intent classified via LLM")
            return result
        logger.info("LLM classification failed, falling back to heuristic")

    return parse_heuristic(prompt)
