"""
Collector registry — dispatches evidence requests to the right collectors.

Collectors are registered at startup and queried by (data_type, league).
Results are ordered by trust tier (ascending = most trusted first).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from omega.evidence.collectors.base import Collector

logger = logging.getLogger("omega.evidence.registry")


class CollectorRegistry:
    """Registry of all available evidence collectors.

    Usage::

        registry = build_default_registry()
        collectors = registry.get_collectors_for("odds", "NBA")
        # → [OddsApiCollector (tier 1), FallbackSearchCollector (tier 3)]
    """

    def __init__(self) -> None:
        self._collectors: List[Collector] = []

    def register(self, collector: Collector) -> None:
        """Add a collector to the registry."""
        self._collectors.append(collector)
        logger.debug(
            "Registered collector %s (types=%s, leagues=%s, tier=%d)",
            collector.name,
            collector.evidence_types,
            collector.supported_leagues,
            collector.trust_tier,
        )

    def get_collectors_for(
        self,
        data_type: str,
        league: str,
    ) -> List[Collector]:
        """Return collectors that can serve *data_type* + *league*, ordered by trust tier.

        Lower trust tier = higher trust = first in list.
        """
        league_upper = league.upper()
        matching = [
            c for c in self._collectors
            if data_type in c.evidence_types
            and league_upper in c.supported_leagues
        ]
        # Sort by trust tier ascending (most trusted first)
        matching.sort(key=lambda c: c.trust_tier)
        return matching

    @property
    def all_collectors(self) -> List[Collector]:
        """All registered collectors."""
        return list(self._collectors)


# ---------------------------------------------------------------------------
# Default registry builder
# ---------------------------------------------------------------------------

_default_registry: Optional[CollectorRegistry] = None


def build_default_registry() -> CollectorRegistry:
    """Wire up all built-in collectors in priority order.

    ESPN and Odds API are registered as optional tier-1 accelerators
    only when available. The LLM-powered web search collector is the
    primary data path (tier 2 for structured results, tier 3 for prose).
    """
    from omega.evidence.collectors.context import ContextCollector
    from omega.evidence.collectors.espn import EspnCollector
    from omega.evidence.collectors.injury import InjuryCollector
    from omega.evidence.collectors.news_signal import NewsSignalCollector
    from omega.evidence.collectors.odds_api import OddsApiCollector
    from omega.evidence.collectors.search import FallbackSearchCollector
    from omega.evidence.collectors.team_form import TeamFormCollector

    registry = CollectorRegistry()

    # Tier 1 (optional accelerators): Only registered when available.
    # The system works fully without these — web search covers all types.
    registry.register(EspnCollector())

    has_odds_key = bool(
        os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    )
    if has_odds_key:
        registry.register(OddsApiCollector())
    else:
        logger.info("Odds API key not set — odds will be served via web search")

    # Tier 2: LLM-powered web search (primary data path)
    # Structured JSON results with sufficient numeric fields get tier 2;
    # raw prose falls back to tier 3 within the collector itself.
    registry.register(FallbackSearchCollector())

    # Tier 2: Derived / computed from structured data
    registry.register(TeamFormCollector())
    registry.register(ContextCollector())

    # Tier 3: Dedicated web-sourced collectors
    registry.register(InjuryCollector())
    registry.register(NewsSignalCollector())

    return registry


def get_default_registry() -> CollectorRegistry:
    """Return the module-level singleton registry (lazy-init)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_registry()
    return _default_registry
