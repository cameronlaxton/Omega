"""
Fact Gatherer — delegates to the search-first data pipeline.

Key behaviors:
    - Delegates slot filling to src/data/orchestration/retrieval_orchestrator.
    - Quality scoring: composite score based on provider confidence and freshness.
    - Graceful degradation: unfilled slots are returned with filled=False.
    - Utility functions for aggregate quality, completeness, and critical input checks.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from omega.core.models import (
    GatherSlot,
    GatheredFact,
    InputImportance,
    ProviderResult,
)

logger = logging.getLogger("omega.agent.fact_gatherer")


def _score_result(slot: GatherSlot, result: ProviderResult) -> float:
    """Compute a quality score for a provider result.

    Factors:
    - Provider confidence (1.0 for primary API, lower for scraped/LLM)
    - Freshness (penalize if close to expiry)
    """
    # Base score is provider confidence
    score = result.confidence

    # Freshness penalty: if data is older than half the freshness window, start penalizing
    age_seconds = (time.time() - result.fetched_at.timestamp())
    if age_seconds < 0:
        age_seconds = 0
    half_life = slot.freshness_max / 2
    if half_life > 0 and age_seconds > half_life:
        freshness_penalty = min(0.3, (age_seconds - half_life) / slot.freshness_max * 0.3)
        score -= freshness_penalty

    return max(0.0, min(1.0, score))


def gather_facts(slots: List[GatherSlot]) -> List[GatheredFact]:
    """Fill gather slots via the search-first data pipeline.

    Delegates to src/data/orchestration/retrieval_orchestrator which handles:
    session cache → DB cache → direct API → web search → extract → normalize →
    validate → fuse → return.

    Returns a list of GatheredFact, one per input slot.
    Unfilled slots have filled=False and result=None.
    """
    from omega.evidence.pipeline.retrieval import retrieve_facts
    return retrieve_facts(slots)


def compute_aggregate_quality(facts: List[GatheredFact]) -> float:
    """Compute an aggregate data quality score across all gathered facts.

    Weights by importance tier:
    - Critical: weight 3
    - Important: weight 2
    - Optional: weight 1
    """
    if not facts:
        return 0.0

    weight_map = {
        InputImportance.CRITICAL: 3.0,
        InputImportance.IMPORTANT: 2.0,
        InputImportance.OPTIONAL: 1.0,
    }

    total_weight = 0.0
    weighted_score = 0.0

    for fact in facts:
        w = weight_map.get(fact.slot.importance, 1.0)
        total_weight += w
        if fact.filled:
            weighted_score += w * fact.quality_score

    if total_weight == 0:
        return 0.0
    return weighted_score / total_weight


def critical_inputs_filled(facts: List[GatheredFact]) -> bool:
    """Check if ALL critical-importance slots were successfully filled.

    Returns False if there are no critical slots (avoids vacuous truth).
    """
    critical_facts = [f for f in facts if f.slot.importance == InputImportance.CRITICAL]
    if not critical_facts:
        return False
    return all(f.filled for f in critical_facts)


def important_inputs_filled(facts: List[GatheredFact]) -> bool:
    """Check if ALL important-importance slots were successfully filled.

    Returns False if there are no important slots (avoids vacuous truth).
    """
    important_facts = [f for f in facts if f.slot.importance == InputImportance.IMPORTANT]
    if not important_facts:
        return False
    return all(f.filled for f in important_facts)


def build_data_completeness(facts: List[GatheredFact]) -> Dict[str, str]:
    """Build a completeness map showing which slots are real/missing.

    Returns: {slot_key: "real" | "missing"}
    """
    completeness: Dict[str, str] = {}
    for fact in facts:
        completeness[fact.slot.key] = "real" if fact.filled else "missing"
    return completeness
