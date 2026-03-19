"""
Team form collector — recent performance, win/loss trends, rolling averages.

Primary: ESPN standings + game details for rolling stats.
Fallback: FallbackSearchCollector (via web search).
"""

from __future__ import annotations

import logging
from typing import Optional, Set

logger = logging.getLogger("omega.evidence.collectors.team_form")


class TeamFormCollector:
    """Collects team statistics and recent form data.

    Implements the :class:`~omega.evidence.collectors.base.Collector` protocol.
    """

    @property
    def name(self) -> str:
        return "team_form"

    @property
    def evidence_types(self) -> Set[str]:
        return {"team_stat"}

    @property
    def supported_leagues(self) -> Set[str]:
        return {"NBA", "NFL", "MLB", "NHL", "NCAAB", "NCAAF", "WNBA"}

    @property
    def trust_tier(self) -> int:
        return 2  # Computed from structured API data

    def collect(
        self,
        entity: str,
        league: str,
        data_type: str,
    ):
        """Collect team form evidence.

        Tries ESPN standings first, then delegates to web search.
        """
        from omega.evidence.collectors.base import CollectorResult

        if data_type != "team_stat":
            return None
        if league.upper() not in self.supported_leagues:
            return None

        # Try ESPN standings as primary source
        try:
            from omega.evidence.collectors.espn import get_standings

            standings = get_standings(league.upper())
            if standings:
                entity_lower = entity.lower()
                matched = [
                    s for s in standings
                    if entity_lower in s.get("team_name", "").lower()
                    or entity_lower == s.get("abbreviation", "").lower()
                ]
                if matched:
                    return CollectorResult(
                        data={"standings": matched, "stats": matched[0].get("stats", {})},
                        source="espn_standings",
                        method="structured_api",
                        trust_tier=2,
                        confidence=0.85,
                        entity_matched=entity,
                    )
        except Exception as exc:
            logger.debug("TeamFormCollector ESPN fallback failed: %s", exc)

        return None
