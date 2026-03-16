"""
Core data models for the agent execution lifecycle.

These models define the contracts between lifecycle stages:
    Intent Understanding → Answer Strategy → Requirement Planning →
    Capability Routing → Fact Gathering → Normalization → Quality Gate →
    Execution → Response Composition

All models are Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dimension A: Subject — what is the prompt about?
# ---------------------------------------------------------------------------

class Subject(str, Enum):
    GAME = "game"
    PLAYER_PROP = "player_prop"
    SLATE = "slate"
    COMPARISON = "comparison"
    BANKROLL = "bankroll"
    NEWS_CONTEXT = "news_context"
    UNSUPPORTED_SPORT = "unsupported_sport"
    GENERAL_SPORTS = "general_sports"


# ---------------------------------------------------------------------------
# Dimension B: User Goal — what does the user want?
# ---------------------------------------------------------------------------

class UserGoal(str, Enum):
    DECIDE = "decide"       # "should I bet this?"
    ANALYZE = "analyze"     # "break this game down"
    COMPARE = "compare"     # "Lakers vs Warriors vs Celtics"
    EXPLAIN = "explain"     # "why is this line moving?"
    DISCUSS = "discuss"     # "what do you think about golf?"
    SUMMARIZE = "summarize" # "catch me up on tonight"
    LEARN = "learn"         # "how does Kelly criterion work?"
    MONITOR = "monitor"     # "any value on the board right now?"


# ---------------------------------------------------------------------------
# Dimension C: Execution Mode — how should the system produce the answer?
# ---------------------------------------------------------------------------

class ExecutionMode(str, Enum):
    NATIVE_SIM = "native_sim"       # full Monte Carlo + calibration + edges
    RESEARCH = "research"           # fact gathering + LLM synthesis
    BANKROLL_CALC = "bankroll_calc"  # Kelly/staking math, no simulation
    MIXED = "mixed"                 # sim for directional context + narrative
    NARRATIVE = "narrative"         # pure discussion/explanation


# ---------------------------------------------------------------------------
# Dimension D: Output Package — what artifacts should the response contain?
# ---------------------------------------------------------------------------

class OutputPackage(str, Enum):
    BET_CARD = "bet_card"                 # edges, staking, confidence tiers
    GAME_BREAKDOWN = "game_breakdown"     # sim results + narrative analysis
    SCENARIO_ANALYSIS = "scenario_analysis"
    KEY_FACTORS = "key_factors"           # top drivers, matchup advantages
    ALTERNATIVE_BETS = "alternative_bets"
    BANKROLL_GUIDANCE = "bankroll_guidance"
    NEWS_DIGEST = "news_digest"
    RESEARCH_REPORT = "research_report"   # structured analysis without formal edges
    PLAIN_EXPLANATION = "plain_explanation"
    COMPACT_SUMMARY = "compact_summary"
    LIMITED_CONTEXT_ANSWER = "limited_context_answer"  # ultra-low-data fallback


# ---------------------------------------------------------------------------
# Input importance tiers for quality gating
# ---------------------------------------------------------------------------

class InputImportance(str, Enum):
    CRITICAL = "critical"     # missing → no formal edges
    IMPORTANT = "important"   # missing → confidence capped at C
    OPTIONAL = "optional"     # missing → still valid, less nuanced


# ---------------------------------------------------------------------------
# Entity reference within a query
# ---------------------------------------------------------------------------

class EntityRole(str, Enum):
    HOME = "home"
    AWAY = "away"
    SUBJECT = "subject"       # player for props, team for research
    OPPONENT = "opponent"


class Entity(BaseModel):
    """A team, player, or participant referenced in the user query."""

    name: str
    role: EntityRole
    entity_type: str = "team"  # "team", "player"
    canonical_id: Optional[UUID] = None  # resolved identity


# ---------------------------------------------------------------------------
# QueryUnderstanding — output of the intent understanding stage
# ---------------------------------------------------------------------------

class QueryUnderstanding(BaseModel):
    """Multi-dimensional classification of a user prompt.

    Produced by the intent understanding stage (LLM or heuristic).
    Consumed by the answer strategist.
    """

    # Dimension A
    subjects: List[Subject]
    league: Optional[str] = None
    entities: List[Entity] = Field(default_factory=list)
    markets: List[str] = Field(default_factory=list)
    prop_type: Optional[str] = None
    prop_line: Optional[float] = None
    date: Optional[str] = None  # ISO date, defaults to today

    # Dimension B
    goal: UserGoal = UserGoal.ANALYZE
    wants_betting_advice: bool = False
    wants_explanation: bool = False
    wants_alternatives: bool = False
    tone: str = "analytical"  # "analytical", "conversational", "brief"

    # User constraints
    explicit_constraints: List[str] = Field(default_factory=list)
    raw_prompt: str = ""


# ---------------------------------------------------------------------------
# AnswerPlan — output of the answer strategist
# ---------------------------------------------------------------------------

class AnswerPlan(BaseModel):
    """Decides what execution modes and output packages the response requires.

    Produced by the answer strategist. May be revised by the quality gate
    if data quality does not meet thresholds.
    """

    execution_modes: List[ExecutionMode]
    output_packages: List[OutputPackage]
    simulation_required: bool = False
    betting_recommendations_included: bool = False

    # Min data quality per output package that requires it
    quality_thresholds: Dict[str, float] = Field(default_factory=dict)

    clarification_needed: bool = False
    clarification_question: Optional[str] = None


# ---------------------------------------------------------------------------
# GatherSlot — a single data value that needs to be gathered
# ---------------------------------------------------------------------------

class GatherSlot(BaseModel):
    """A typed slot representing a data value the system must gather.

    Produced by the requirement planner. Consumed by the capability router
    and fact gatherer.
    """

    key: str                           # e.g. "home_team.off_rating"
    data_type: str                     # "team_stat", "odds", "injury", "schedule", "player_stat"
    entity: str                        # team or player name
    league: str
    importance: InputImportance = InputImportance.IMPORTANT
    freshness_max: float = 86400.0     # seconds; default 24h
    providers: List[str] = Field(default_factory=list)  # preferred provider names


# ---------------------------------------------------------------------------
# ProviderResult — what a provider returns for a gather slot
# ---------------------------------------------------------------------------

class ProviderResult(BaseModel):
    """Result from a data provider for a single gather slot.

    Carries provenance metadata for quality scoring and audit.
    """

    data: Dict[str, Any]
    source: str                        # provider name: "espn", "bbref", etc.
    source_url: Optional[str] = None   # attribution URL
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 1.0            # 1.0 = primary API, 0.5 = LLM-extracted, 0.3 = stale cache


# ---------------------------------------------------------------------------
# GatheredFact — a filled slot with provenance
# ---------------------------------------------------------------------------

class GatheredFact(BaseModel):
    """A gather slot that has been filled with a provider result."""

    slot: GatherSlot
    result: Optional[ProviderResult] = None
    filled: bool = False
    quality_score: float = 0.0         # 0.0–1.0 composite quality


# ---------------------------------------------------------------------------
# ExecutionResult — output from the deterministic engine
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    """Wrapper around whatever the execution engine produced.

    May contain simulation results, edge calculations, staking recommendations,
    or just gathered facts for research mode.
    """

    mode: ExecutionMode
    simulation: Optional[Dict[str, Any]] = None
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    best_bet: Optional[Dict[str, Any]] = None
    research_facts: List[GatheredFact] = Field(default_factory=list)
    data_quality_score: float = 0.0
    data_completeness: Dict[str, str] = Field(default_factory=dict)  # key → "real" | "defaulted" | "missing"


# ---------------------------------------------------------------------------
# Freshness rules by data type
# ---------------------------------------------------------------------------

FRESHNESS_RULES: Dict[str, float] = {
    "team_stat": 86400.0,       # 24 hours
    "player_stat": 86400.0,     # 24 hours
    "player_game_log": 86400.0, # 24 hours
    "odds": 900.0,              # 15 minutes
    "injury": 7200.0,           # 2 hours
    "schedule": 3600.0,         # 1 hour
    "environment": 14400.0,     # 4 hours
}
