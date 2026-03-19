"""
News signal collector — breaking news, trades, coaching changes.

Purely web-search-based with aggressive freshness (15 minutes).
"""

from __future__ import annotations

import logging
from typing import Optional, Set

logger = logging.getLogger("omega.evidence.collectors.news_signal")


class NewsSignalCollector:
    """Collects breaking news signals that may affect a game.

    Implements the :class:`~omega.evidence.collectors.base.Collector` protocol.
    """

    @property
    def name(self) -> str:
        return "news_signal"

    @property
    def evidence_types(self) -> Set[str]:
        return {"news_signal"}

    @property
    def supported_leagues(self) -> Set[str]:
        return {
            "NBA", "NFL", "MLB", "NHL", "NCAAB", "NCAAF", "WNBA",
            "MLS", "EPL", "UFC", "ATP", "PGA",
        }

    @property
    def trust_tier(self) -> int:
        return 3  # Web-sourced

    def collect(
        self,
        entity: str,
        league: str,
        data_type: str,
    ):
        """Collect news signals.

        Returns None for now — delegates to FallbackSearchCollector
        via pipeline fallthrough.
        """
        if data_type != "news_signal":
            return None
        if league.upper() not in self.supported_leagues:
            return None

        # Future: focused news search with 15-min freshness window
        return None
