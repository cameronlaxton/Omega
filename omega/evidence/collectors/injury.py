"""
Injury collector — dedicated injury/availability data.

Currently delegates to web search with injury-specific prompting.
Future: direct ESPN injury endpoint when available.
"""

from __future__ import annotations

import logging
from typing import Optional, Set

logger = logging.getLogger("omega.evidence.collectors.injury")


class InjuryCollector:
    """Collects injury and availability data for a team.

    Implements the :class:`~omega.evidence.collectors.base.Collector` protocol.
    """

    @property
    def name(self) -> str:
        return "injury"

    @property
    def evidence_types(self) -> Set[str]:
        return {"injury"}

    @property
    def supported_leagues(self) -> Set[str]:
        return {"NBA", "NFL", "MLB", "NHL", "NCAAB", "NCAAF", "WNBA"}

    @property
    def trust_tier(self) -> int:
        return 3  # Web-sourced until dedicated API added

    def collect(
        self,
        entity: str,
        league: str,
        data_type: str,
    ):
        """Collect injury data.

        Currently delegates to FallbackSearchCollector with focused query.
        Returns None to let the pipeline fall through to web search.
        """
        if data_type != "injury":
            return None
        if league.upper() not in self.supported_leagues:
            return None

        # Future: ESPN injury API endpoint or rotowire scraper
        # For now, return None and let FallbackSearchCollector handle it
        return None
