"""
Retrieval orchestrator — search-first data pipeline.

Pipeline stages per slot:
    1. Session cache (in-memory, fast)
    2. Direct API (ESPN, stats sites, odds API)
    3. Web search (Perplexity Sonar) → extract → normalize → validate → fuse
    4. Return GatheredFact (filled=True|False)

Phase 3 stub: returns unfilled facts. Will be wired to real data
sources in a follow-up phase.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from omega.core.models import GatherSlot, GatheredFact, ProviderResult

logger = logging.getLogger("omega.research.data.retrieval")


def retrieve_facts(slots: List[GatherSlot]) -> List[GatheredFact]:
    """Fill gather slots via the data pipeline.

    Currently a stub — returns all slots as unfilled.
    Phase 4 will wire in real data sources.
    """
    facts: List[GatheredFact] = []
    for slot in slots:
        # TODO: wire in real pipeline stages
        facts.append(GatheredFact(
            slot=slot,
            result=None,
            filled=False,
            quality_score=0.0,
        ))
    return facts
