"""
Retrieval orchestrator -- the main entry point for the search-first data pipeline.

Pipeline stages per slot:
    1. Session cache (in-memory LRU, same query)
    2. Direct API (ESPN schedule, Odds API)
    3. Web search (Perplexity Sonar / Anthropic) -> normalize -> validate -> fuse
    4. Return GatheredFact (filled=True|False)

Each stage is tried in order; first hit wins.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from omega.core.models import GatherSlot, GatheredFact, ProviderResult
from omega.research.data.models import (
    FactBundle,
    SearchResult,
    SourceAttribution,
    SportsFact,
)
from omega.research.data.sources.config import (
    DIRECT_API_CAPABILITIES,
    get_confidence_for_tier,
    get_trust_tier,
)

logger = logging.getLogger("omega.research.data.retrieval")


# ---------------------------------------------------------------------------
# Simple in-memory session cache
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
    """Fill gather slots via the search-first data pipeline.

    For each slot:
    1. Check session cache (in-memory LRU)
    2. Try direct API (ESPN schedule, Odds API -- fast path)
    3. Web search -> normalize -> validate -> fuse
    4. Return as GatheredFact

    Args:
        slots: List of GatherSlots from the requirement planner.

    Returns:
        List of GatheredFacts, one per input slot.
    """
    cache = _SessionCache()
    results: List[GatheredFact] = []

    for slot in slots:
        fact = _fill_slot(slot, cache)
        results.append(fact)

    filled_count = sum(1 for f in results if f.filled)
    logger.info("Pipeline filled %d/%d slots", filled_count, len(slots))

    return results


# ---------------------------------------------------------------------------
# Per-slot pipeline
# ---------------------------------------------------------------------------

def _fill_slot(slot: GatherSlot, cache: _SessionCache) -> GatheredFact:
    """Fill a single gather slot through the pipeline stages."""

    # Stage 1: Session cache
    cached = cache.get(slot.data_type, slot.entity, slot.league)
    if cached is not None:
        logger.debug("Session cache hit for slot %s", slot.key)
        return _make_gathered_fact(slot, cached, "session_cache", 0.90)

    # Stage 2: Direct API (fast path)
    direct = _try_direct_api(slot)
    if direct is not None:
        cache.put(slot.data_type, slot.entity, slot.league, direct["data"])
        return _make_gathered_fact(
            slot, direct["data"], direct["source"], direct["confidence"]
        )

    # Stage 3: Web search pipeline
    search = _try_web_search_pipeline(slot)
    if search is not None:
        cache.put(slot.data_type, slot.entity, slot.league, search["data"])
        return _make_gathered_fact(
            slot, search["data"], search["source"], search["confidence"]
        )

    # All sources exhausted
    logger.info("All sources exhausted for slot %s", slot.key)
    return GatheredFact(slot=slot, result=None, filled=False, quality_score=0.0)


# ---------------------------------------------------------------------------
# Stage 2: Direct API
# ---------------------------------------------------------------------------

def _try_direct_api(slot: GatherSlot) -> Optional[Dict[str, Any]]:
    """Try direct API modules as fast path.

    Currently supports:
    - ESPN schedule API (data_type=schedule)
    - The Odds API (data_type=odds)
    """
    league = slot.league.upper() if slot.league else ""

    # ESPN schedule
    if slot.data_type == "schedule" and league in DIRECT_API_CAPABILITIES.get("espn", {}).get("leagues", set()):
        try:
            from omega.research.data.acquisition.espn import get_todays_games

            games = get_todays_games(league)
            if games:
                # Filter to matching entity if possible
                entity_lower = slot.entity.lower()
                matched = [
                    g for g in games
                    if entity_lower in (g.get("home_team", {}) or {}).get("name", "").lower()
                    or entity_lower in (g.get("away_team", {}) or {}).get("name", "").lower()
                    or entity_lower in g.get("name", "").lower()
                ]
                data = matched if matched else games
                return {"data": {"games": data}, "source": "espn", "confidence": 0.95}
        except Exception as exc:
            logger.debug("ESPN direct API failed for slot %s: %s", slot.key, exc)

    # The Odds API
    if slot.data_type == "odds" and league in DIRECT_API_CAPABILITIES.get("odds_api", {}).get("leagues", set()):
        try:
            from omega.research.data.acquisition.odds_api import (
                extract_consensus_odds,
                get_upcoming_odds,
            )

            games = get_upcoming_odds(league)
            if games:
                entity_lower = slot.entity.lower()
                matched = [
                    g for g in games
                    if entity_lower in g.get("home_team", "").lower()
                    or entity_lower in g.get("away_team", "").lower()
                ]
                if matched:
                    consensus = extract_consensus_odds(matched)
                    return {"data": {"odds": consensus, "raw_games": matched}, "source": "odds_api", "confidence": 0.95}
                else:
                    # Return all odds for the league
                    consensus = extract_consensus_odds(games)
                    return {"data": {"odds": consensus}, "source": "odds_api", "confidence": 0.90}
        except Exception as exc:
            logger.debug("Odds API direct failed for slot %s: %s", slot.key, exc)

    return None


# ---------------------------------------------------------------------------
# Stage 3: Web search pipeline
# ---------------------------------------------------------------------------

def _try_web_search_pipeline(slot: GatherSlot) -> Optional[Dict[str, Any]]:
    """Web search -> extract structured results -> normalize -> validate -> fuse."""
    try:
        from omega.research.data.acquisition.search import search_for_slot

        search_results = search_for_slot(slot)
        if not search_results:
            return None

        # Extract facts from search results
        all_facts = _extract_facts_from_results(search_results, slot)
        if not all_facts:
            return None

        # Normalize
        all_facts = _normalize_facts(all_facts, slot)

        # Validate
        all_facts = _validate_facts(all_facts, slot)
        if not all_facts:
            return None

        # Fuse
        bundle = FactBundle(
            slot_key=slot.key,
            data_type=slot.data_type,
            entity=slot.entity,
            league=slot.league,
            facts=all_facts,
        )

        from omega.research.data.fusion.fuser import fuse_facts, score_confidence

        fused_data = fuse_facts(bundle)
        confidence = score_confidence(bundle)

        if not fused_data:
            return None

        # Find best source name
        best = min(all_facts, key=lambda f: f.attribution.trust_tier)

        return {
            "data": fused_data,
            "source": best.attribution.source_name,
            "confidence": confidence,
        }

    except Exception as exc:
        logger.debug("Web search pipeline failed for slot %s: %s", slot.key, exc)
        return None


def _extract_facts_from_results(
    search_results: List[SearchResult], slot: GatherSlot
) -> List[SportsFact]:
    """Convert search results to SportsFacts.

    Handles two paths:
    1. Structured Perplexity results (domain=perplexity.structured) -> direct JSON
    2. Prose results -> stored as single "raw_text" fact for downstream use
    """
    facts: List[SportsFact] = []

    for sr in search_results:
        if sr.domain == "perplexity.structured":
            # Structured JSON path -- bypass extraction
            try:
                data = json.loads(sr.snippet)
            except (ValueError, TypeError):
                continue

            if not isinstance(data, dict):
                continue

            attribution = SourceAttribution(
                source_name="perplexity.structured",
                source_url=None,
                fetched_at=datetime.now(timezone.utc),
                trust_tier=2,
                confidence=0.80,
            )

            for key, value in data.items():
                if value is None:
                    continue
                facts.append(SportsFact(
                    key=key,
                    value=value,
                    data_type=slot.data_type,
                    entity=slot.entity,
                    league=slot.league,
                    attribution=attribution,
                ))

        elif sr.snippet and sr.domain in ("anthropic.web_search", "perplexity.ai"):
            # Prose results -- store as raw text fact
            trust_tier = get_trust_tier(sr.domain)
            attribution = SourceAttribution(
                source_name=f"web_search:{sr.domain}",
                source_url=sr.url if not sr.url.startswith(("perplexity://", "anthropic://")) else None,
                fetched_at=datetime.now(timezone.utc),
                trust_tier=trust_tier,
                confidence=get_confidence_for_tier(trust_tier),
            )
            facts.append(SportsFact(
                key="raw_text",
                value=sr.snippet,
                data_type=slot.data_type,
                entity=slot.entity,
                league=slot.league,
                attribution=attribution,
            ))

    return facts


def _normalize_facts(facts: List[SportsFact], slot: GatherSlot) -> List[SportsFact]:
    """Run normalization on extracted facts."""
    from omega.research.data.normalizers.stats import normalize_stat_value

    for fact in facts:
        if not fact.normalized and fact.key != "raw_text":
            fact.value = normalize_stat_value(fact.key, fact.value, slot.league)
            fact.normalized = True

    return facts


def _validate_facts(facts: List[SportsFact], slot: GatherSlot) -> List[SportsFact]:
    """Run validation pipeline on facts."""
    from omega.research.data.validators.freshness import validate_freshness
    from omega.research.data.validators.sanity import validate_sanity

    facts = validate_freshness(facts, slot.data_type)
    facts = validate_sanity(facts)

    return facts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gathered_fact(
    slot: GatherSlot,
    data: Dict[str, Any],
    source: str,
    confidence: float,
) -> GatheredFact:
    """Convert pipeline output to a GatheredFact."""
    return GatheredFact(
        slot=slot,
        result=ProviderResult(
            data=data,
            source=source,
            fetched_at=datetime.now(timezone.utc),
            confidence=confidence,
        ),
        filled=True,
        quality_score=confidence,
    )
