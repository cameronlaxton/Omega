"""
Context collector — venue, weather, rest days, travel.

Primary: ESPN game details (venue, weather already parsed).
Computes rest days from schedule data.
"""

from __future__ import annotations

import logging
from typing import Optional, Set

logger = logging.getLogger("omega.evidence.collectors.context")


class ContextCollector:
    """Collects environmental/contextual evidence for a game.

    Implements the :class:`~omega.evidence.collectors.base.Collector` protocol.
    """

    @property
    def name(self) -> str:
        return "context"

    @property
    def evidence_types(self) -> Set[str]:
        return {"environment"}

    @property
    def supported_leagues(self) -> Set[str]:
        return {"NBA", "NFL", "MLB", "NHL", "NCAAB", "NCAAF", "WNBA", "MLS"}

    @property
    def trust_tier(self) -> int:
        return 2  # Derived from structured API data

    def collect(
        self,
        entity: str,
        league: str,
        data_type: str,
    ):
        """Collect context/environment evidence.

        Extracts venue and weather from ESPN game details.
        Returns None for now — full implementation in future iteration.
        """
        if data_type != "environment":
            return None
        if league.upper() not in self.supported_leagues:
            return None

        # Future: ESPN game details for venue/weather + schedule for rest days
        return None
