"""
Freshness validator -- checks if facts are within acceptable age limits.

Facts whose attribution.fetched_at is older than the freshness window
for their data_type are excluded.
"""

from __future__ import annotations

import time
from typing import List

from omega.core.models import FRESHNESS_RULES
from omega.research.data.models import SportsFact


def validate_freshness(facts: List[SportsFact], data_type: str) -> List[SportsFact]:
    """Filter facts to only those within the freshness window.

    Args:
        facts: List of facts to validate.
        data_type: The data type to look up freshness rules.

    Returns:
        List of facts that pass freshness validation.
    """
    max_age = FRESHNESS_RULES.get(data_type, 86400.0)
    now = time.time()
    valid: List[SportsFact] = []

    for fact in facts:
        age = now - fact.attribution.fetched_at.timestamp()
        if age <= max_age:
            fact.validated = True
            valid.append(fact)

    return valid
