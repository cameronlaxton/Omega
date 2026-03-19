"""
Canonical data structures for the search-first data pipeline.

These are pipeline-internal types used between stages (acquisition -> extraction ->
normalization -> validation -> fusion -> orchestration). They are distinct from
omega.core.models contracts (GatherSlot, ProviderResult, GatheredFact) which define
the interface between the agent layer and the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SourceAttribution:
    """Provenance metadata for a single data observation."""

    source_name: str        # "espn", "web_search:basketball-reference.com", "odds_api"
    source_url: Optional[str] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trust_tier: int = 4     # 1=official API, 2=reference site, 3=major sports, 4=general web
    confidence: float = 0.5  # 0.0-1.0


@dataclass
class SportsFact:
    """A single extracted fact with provenance.

    Represents one data point (e.g., "Lakers offensive rating = 115.2") from
    one source. Multiple SportsFacts for the same key from different sources
    are collected into a FactBundle for fusion.
    """

    key: str                # "off_rating", "moneyline_home", "status", "pts_mean"
    value: Any              # The actual value
    data_type: str          # "team_stat", "odds", "injury", "schedule", "player_stat"
    entity: str             # Team or player name
    league: str             # "NBA", "NFL", etc.
    attribution: SourceAttribution

    normalized: bool = False
    validated: bool = False


@dataclass
class FactBundle:
    """Collection of facts gathered for a single GatherSlot from multiple sources.

    After fusion, `fused_value` contains the merged result and `fused_confidence`
    reflects agreement, trust, and freshness.
    """

    slot_key: str
    data_type: str
    entity: str
    league: str
    facts: List[SportsFact] = field(default_factory=list)

    # Populated after fusion
    fused_value: Optional[Dict[str, Any]] = None
    fused_confidence: float = 0.0


@dataclass
class SearchResult:
    """A single web search result."""

    url: str
    title: str
    snippet: str
    domain: str = ""        # extracted from url for trust tier lookup
