"""
Retrieval orchestrator -- the main entry point for the evidence pipeline.

Pipeline per slot:
    1. Resolve entity via EntityResolver
    2. Check session cache (in-memory LRU)
    3. Query CollectorRegistry for matching collectors (ordered by trust tier)
    4. Try collectors in order; first successful result wins
    5. Return GatheredFact (filled=True|False)

The public API (``retrieve_facts``) is unchanged from Phase 1 so all
existing consumers continue to work.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from omega.core.models import GatherSlot, GatheredFact, ProviderResult
from omega.evidence.collectors.base import CollectorResult, validate_collector_numeric_fields

logger = logging.getLogger("omega.evidence.pipeline.retrieval")


# ---------------------------------------------------------------------------
# Simple in-memory session cache (unchanged from Phase 1)
# ---------------------------------------------------------------------------

class _SessionCache:
    """LRU cache for within-session deduplication."""

    def __init__(self, max_size: int = 128):
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size

    def _key(self, data_type: str, entity: str, league: str) -> str:
        return f"{data_type}|{entity.lower()}|{league.upper()}"

    def get(self, data_type: str, entity: str, league: str) -> Optional[Dict[str, Any]]:
        k = self._key(data_type, entity, league)
        if k in self._store:
            self._store.move_to_end(k)
            return self._store[k]
        return None

    def put(self, data_type: str, entity: str, league: str, data: Dict[str, Any]) -> None:
        k = self._key(data_type, entity, league)
        self._store[k] = data
        self._store.move_to_end(k)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_facts(slots: List[GatherSlot]) -> List[GatheredFact]:
    """Fill gather slots via the evidence pipeline.

    For each slot:
    1. Resolve entity name to canonical form
    2. Check session cache
    3. Try registered collectors in trust-tier order
    4. Return as GatheredFact

    Args:
        slots: List of GatherSlots from the requirement planner.

    Returns:
        List of GatheredFacts, one per input slot.
    """
    from omega.evidence.entity.resolver import get_resolver
    from omega.evidence.registry import get_default_registry

    cache = _SessionCache()
    resolver = get_resolver()
    registry = get_default_registry()
    results: List[GatheredFact] = []

    for slot in slots:
        # Step 1: Entity resolution
        resolved = resolver.resolve(slot.entity, slot.league)
        canonical_entity = resolved.canonical

        fact = _fill_slot(slot, canonical_entity, cache, registry)
        results.append(fact)

    filled_count = sum(1 for f in results if f.filled)
    logger.info("Pipeline filled %d/%d slots", filled_count, len(slots))

    return results


# ---------------------------------------------------------------------------
# Per-slot pipeline
# ---------------------------------------------------------------------------

def _fill_slot(
    slot: GatherSlot,
    canonical_entity: str,
    cache: _SessionCache,
    registry,
) -> GatheredFact:
    """Fill a single gather slot through the pipeline stages."""

    # Stage 1: Session cache
    cached = cache.get(slot.data_type, canonical_entity, slot.league)
    if cached is not None:
        logger.debug("Session cache hit for slot %s", slot.key)
        return _make_gathered_fact(slot, cached, "session_cache", 0.90, method="cache_hit")

    # Stage 2: Collector dispatch (trust-tier order)
    collectors = registry.get_collectors_for(slot.data_type, slot.league)

    for collector in collectors:
        try:
            result = collector.collect(canonical_entity, slot.league, slot.data_type)
            if result is not None:
                result.data = validate_collector_numeric_fields(result.data, slot.data_type)
                logger.debug(
                    "Collector %s succeeded for slot %s (tier=%d, method=%s)",
                    collector.name, slot.key, result.trust_tier, result.method,
                )
                cache.put(slot.data_type, canonical_entity, slot.league, result.data)
                return _make_gathered_fact(
                    slot,
                    result.data,
                    result.source,
                    result.confidence,
                    method=result.method,
                )
        except Exception as exc:
            logger.debug("Collector %s failed for slot %s: %s", collector.name, slot.key, exc)

    # All collectors exhausted
    logger.info("All collectors exhausted for slot %s", slot.key)
    return GatheredFact(slot=slot, result=None, filled=False, quality_score=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gathered_fact(
    slot: GatherSlot,
    data: Dict[str, Any],
    source: str,
    confidence: float,
    method: str = "unknown",
) -> GatheredFact:
    """Convert pipeline output to a GatheredFact."""
    return GatheredFact(
        slot=slot,
        result=ProviderResult(
            data=data,
            source=source,
            fetched_at=datetime.now(timezone.utc),
            confidence=confidence,
            method=method,
        ),
        filled=True,
        quality_score=confidence,
    )
