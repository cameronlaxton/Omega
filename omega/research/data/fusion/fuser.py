"""
Fact fuser -- merge multiple observations for the same slot into a single value.

When multiple sources provide data for the same key, the fuser picks the
best value based on trust tier, freshness, and agreement between sources.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

from omega.research.data.models import FactBundle, SportsFact

logger = logging.getLogger("omega.data.fusion")


def fuse_facts(bundle: FactBundle) -> Dict[str, Any]:
    """Fuse all facts in a bundle into a single dict of key->value.

    Strategy:
    - Group facts by key
    - For each key, pick the value from the highest-trust source
    - If multiple sources at the same trust tier agree, boost confidence
    - If they disagree, use the value from the most recent source

    Args:
        bundle: FactBundle with facts from multiple sources.

    Returns:
        Merged dict of key->value pairs.
    """
    if not bundle.facts:
        return {}

    # Group facts by key
    by_key: Dict[str, List[SportsFact]] = defaultdict(list)
    for fact in bundle.facts:
        by_key[fact.key].append(fact)

    fused: Dict[str, Any] = {}

    for key, facts in by_key.items():
        if len(facts) == 1:
            fused[key] = facts[0].value
        else:
            # Sort by trust tier (ascending = better), then by freshness (desc)
            facts.sort(key=lambda f: (f.attribution.trust_tier, -f.attribution.fetched_at.timestamp()))
            fused[key] = facts[0].value

    return fused


def score_confidence(bundle: FactBundle) -> float:
    """Score overall confidence for a fused bundle.

    Factors:
    - Average trust tier of sources
    - Number of sources
    - Agreement between sources for shared keys

    Returns:
        Confidence score 0.0-1.0.
    """
    if not bundle.facts:
        return 0.0

    # Base confidence from average trust tier
    avg_confidence = sum(f.attribution.confidence for f in bundle.facts) / len(bundle.facts)

    # Bonus for multiple sources
    unique_sources = len({f.attribution.source_name for f in bundle.facts})
    source_bonus = min(0.1, unique_sources * 0.03)

    # Agreement check on shared keys
    by_key: Dict[str, List[Any]] = defaultdict(list)
    for fact in bundle.facts:
        by_key[fact.key].append(fact.value)

    agreement_scores: List[float] = []
    for key, values in by_key.items():
        if len(values) > 1:
            # Check if values agree (within 5% for numerics, exact for strings)
            if _values_agree(values):
                agreement_scores.append(1.0)
            else:
                agreement_scores.append(0.5)

    agreement_bonus = 0.0
    if agreement_scores:
        agreement_bonus = (sum(agreement_scores) / len(agreement_scores)) * 0.1

    return min(1.0, avg_confidence + source_bonus + agreement_bonus)


def _values_agree(values: List[Any]) -> bool:
    """Check if multiple values for the same key are in agreement."""
    if not values:
        return True

    # Filter out None values
    non_none = [v for v in values if v is not None]
    if len(non_none) <= 1:
        return True

    # Try numeric comparison
    try:
        nums = [float(v) for v in non_none]
        if not nums:
            return True
        avg = sum(nums) / len(nums)
        if avg == 0:
            return all(n == 0 for n in nums)
        # Within 5% of each other
        return all(abs(n - avg) / abs(avg) < 0.05 for n in nums)
    except (ValueError, TypeError):
        pass

    # String comparison
    return len(set(str(v).lower().strip() for v in non_none)) == 1
