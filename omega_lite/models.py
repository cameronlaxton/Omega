"""
Minimal subset of omega.core.models needed by the omega_lite quality gate.

Extracted verbatim from omega/core/models.py — keep in sync with canonical
when the enums or schemas change.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------

class ExecutionMode(str, Enum):
    NATIVE_SIM = "native_sim"
    RESEARCH = "research"
    BANKROLL_CALC = "bankroll_calc"
    MIXED = "mixed"
    NARRATIVE = "narrative"


# ---------------------------------------------------------------------------
# Output package types
# ---------------------------------------------------------------------------

class OutputPackage(str, Enum):
    BET_CARD = "bet_card"
    GAME_BREAKDOWN = "game_breakdown"
    SCENARIO_ANALYSIS = "scenario_analysis"
    KEY_FACTORS = "key_factors"
    ALTERNATIVE_BETS = "alternative_bets"
    BANKROLL_GUIDANCE = "bankroll_guidance"
    NEWS_DIGEST = "news_digest"
    RESEARCH_REPORT = "research_report"
    PLAIN_EXPLANATION = "plain_explanation"
    COMPACT_SUMMARY = "compact_summary"
    LIMITED_CONTEXT_ANSWER = "limited_context_answer"


# ---------------------------------------------------------------------------
# Input importance tiers
# ---------------------------------------------------------------------------

class InputImportance(str, Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    OPTIONAL = "optional"


# ---------------------------------------------------------------------------
# AnswerPlan
# ---------------------------------------------------------------------------

class AnswerPlan(BaseModel):
    """Output of the answer strategist. Revised by the quality gate."""

    execution_modes: List[ExecutionMode]
    output_packages: List[OutputPackage]
    simulation_required: bool = False
    betting_recommendations_included: bool = False

    quality_thresholds: Dict[str, float] = Field(default_factory=dict)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    downgrades: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# GatherSlot / ProviderResult / GatheredFact
# ---------------------------------------------------------------------------

class GatherSlot(BaseModel):
    """A typed slot representing a data value to gather."""

    key: str
    data_type: str
    entity: str
    league: str
    importance: InputImportance = InputImportance.IMPORTANT
    freshness_max: float = 86400.0
    providers: List[str] = Field(default_factory=list)
    focus_hint: Optional[str] = None


class ProviderResult(BaseModel):
    """Result from a data provider for a single gather slot."""

    data: Dict[str, Any]
    source: str
    source_url: Optional[str] = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    method: str = "unknown"


class GatheredFact(BaseModel):
    """A gather slot that has been filled (or not)."""

    slot: GatherSlot
    result: Optional[ProviderResult] = None
    filled: bool = False
    quality_score: float = 0.0
