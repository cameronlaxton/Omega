"""
Collector protocol and result types for the evidence layer.

Every evidence-class collector implements the Collector protocol so the
pipeline can dispatch to them uniformly.  CollectorResult carries full
provenance metadata (source, method, trust tier, confidence).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, Set, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger("omega.evidence.collectors.base")


# ---------------------------------------------------------------------------
# Result type returned by every collector
# ---------------------------------------------------------------------------

class CollectorResult(BaseModel):
    """Typed result from a single evidence collection attempt."""

    data: Dict[str, Any]
    source: str                          # "espn", "odds_api", "perplexity", ...
    source_url: Optional[str] = None
    method: str                          # "structured_api", "llm_extraction", "web_scrape", "cache_hit"
    trust_tier: int                      # 1-4  (mirrors sources/config tiers)
    confidence: float                    # 0.0-1.0
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entity_matched: str = ""             # canonical entity name after resolution


# ---------------------------------------------------------------------------
# Collector protocol – the contract every collector must satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class Collector(Protocol):
    """Evidence-class collector interface.

    Implementations serve one or more evidence types (data_type values)
    for one or more leagues.  The pipeline queries ``evidence_types`` and
    ``supported_leagues`` to decide which collectors to try for a given
    GatherSlot, then calls ``collect()`` in trust-tier order.
    """

    @property
    def name(self) -> str:
        """Human-readable collector name (e.g. 'espn', 'odds_api')."""
        ...

    @property
    def evidence_types(self) -> Set[str]:
        """Data types this collector can serve (e.g. {"schedule", "odds"})."""
        ...

    @property
    def supported_leagues(self) -> Set[str]:
        """League codes this collector supports (e.g. {"NBA", "NFL"})."""
        ...

    @property
    def trust_tier(self) -> int:
        """Default trust tier for this collector's results (1-4)."""
        ...

    def collect(
        self,
        entity: str,
        league: str,
        data_type: str,
    ) -> Optional[CollectorResult]:
        """Attempt to collect evidence for *entity* in *league*.

        Returns ``None`` if no data is available or an error occurs.
        Implementations must not raise — they should catch exceptions
        internally and return ``None``.
        """
        ...


# ---------------------------------------------------------------------------
# Collector output validation
# ---------------------------------------------------------------------------

# Build the set of all known numeric keys from archetypes (lazy, cached)
_NUMERIC_KEYS: Optional[Set[str]] = None

_ODDS_KEYS = {"moneyline_home", "moneyline_away", "spread_home", "over_under"}


def _get_numeric_keys() -> Set[str]:
    """Return all keys that should be numeric based on archetype definitions."""
    global _NUMERIC_KEYS
    if _NUMERIC_KEYS is None:
        from omega.core.simulation.archetypes import ARCHETYPE_REGISTRY
        keys: Set[str] = set()
        for arch in ARCHETYPE_REGISTRY.values():
            keys.update(arch.critical_team_keys)
            keys.update(arch.required_team_keys)
            keys.update(arch.optional_team_keys)
        _NUMERIC_KEYS = keys
    return _NUMERIC_KEYS


def validate_collector_numeric_fields(
    data: Dict[str, Any], data_type: str,
) -> Dict[str, Any]:
    """Validate numeric fields in collector output.

    For stat-type data: checks values against known archetype keys,
    coerces string-numerics, drops garbage.
    For odds data: validates standard odds keys.
    For other types: passes through unchanged.

    Returns a new dict (never mutates the input).
    """
    if data_type not in ("team_stat", "player_stat", "odds"):
        return data

    numeric_keys = _ODDS_KEYS if data_type == "odds" else _get_numeric_keys()
    cleaned: Dict[str, Any] = {}

    for key, value in data.items():
        if key not in numeric_keys:
            cleaned[key] = value
            continue

        if isinstance(value, (int, float)):
            if math.isfinite(value):
                cleaned[key] = value
            else:
                logger.warning("Collector validation: dropped %s (NaN/Inf)", key)
            continue

        if isinstance(value, str):
            try:
                num = float(value)
                if math.isfinite(num):
                    cleaned[key] = num
                    continue
            except (ValueError, TypeError):
                pass
            logger.warning("Collector validation: dropped %s = %r (non-numeric)", key, value)
            continue

        logger.warning("Collector validation: dropped %s = %r (unexpected type)", key, value)

    return cleaned
