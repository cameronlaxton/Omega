"""
Quality aggregation helpers — pure-logic extract from omega/reasoning/gatherer.py.

The gather pipeline itself is not part of omega_lite (no network in the
sandbox). These helpers operate on already-built GatheredFact lists that
the caller supplies (or that omega_lite.run synthesizes from a request).
"""

from __future__ import annotations

from typing import Dict, List

from omega_lite.models import GatheredFact, InputImportance


def compute_aggregate_quality(facts: List[GatheredFact]) -> float:
    """Weighted average quality across all gathered facts.

    Weights: CRITICAL=3, IMPORTANT=2, OPTIONAL=1.
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
    """True only if every CRITICAL slot was filled.

    Returns False if there are no critical slots at all (avoids vacuous truth).
    """
    critical_facts = [f for f in facts if f.slot.importance == InputImportance.CRITICAL]
    if not critical_facts:
        return False
    return all(f.filled for f in critical_facts)


def important_inputs_filled(facts: List[GatheredFact]) -> bool:
    """True only if every IMPORTANT slot was filled.

    Returns False if there are no important slots at all.
    """
    important_facts = [f for f in facts if f.slot.importance == InputImportance.IMPORTANT]
    if not important_facts:
        return False
    return all(f.filled for f in important_facts)


def build_data_completeness(facts: List[GatheredFact]) -> Dict[str, str]:
    """{slot_key: "real" | "missing"} map."""
    completeness: Dict[str, str] = {}
    for fact in facts:
        completeness[fact.slot.key] = "real" if fact.filled else "missing"
    return completeness
