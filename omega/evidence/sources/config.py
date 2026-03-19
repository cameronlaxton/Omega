"""
Source trust configuration -- trust tiers, domain rules, and direct API capabilities.

Trust tiers classify data sources by reliability:
  Tier 1: Official APIs with structured responses (highest confidence)
  Tier 2: Reference sites with curated, verified data
  Tier 3: Major sports media sites with editorial standards
  Tier 4: General web (LLM-extracted, unverified)
"""

from __future__ import annotations

from typing import Any, Dict, Set
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Trust tier definitions
# ---------------------------------------------------------------------------

TRUST_TIERS: Dict[int, Dict[str, Any]] = {
    1: {
        "confidence": 0.95,
        "label": "Official API",
        "examples": ["odds_api", "espn_api"],
    },
    2: {
        "confidence": 0.90,
        "label": "Reference site",
        "examples": [
            "basketball-reference.com",
            "pro-football-reference.com",
            "baseball-reference.com",
            "hockey-reference.com",
        ],
    },
    3: {
        "confidence": 0.80,
        "label": "Major sports site",
        "examples": [
            "espn.com",
            "cbssports.com",
            "nba.com",
            "nfl.com",
            "mlb.com",
            "nhl.com",
        ],
    },
    4: {
        "confidence": 0.50,
        "label": "General web",
        "examples": ["any other source"],
    },
}


# ---------------------------------------------------------------------------
# Domain -> trust tier mapping
# ---------------------------------------------------------------------------

DOMAIN_TRUST: Dict[str, int] = {
    # Tier 2: Reference sites
    "basketball-reference.com": 2,
    "pro-football-reference.com": 2,
    "baseball-reference.com": 2,
    "hockey-reference.com": 2,
    "fbref.com": 2,
    "stathead.com": 2,
    # Tier 3: Major sports sites
    "espn.com": 3,
    "espn.go.com": 3,
    "nba.com": 3,
    "nfl.com": 3,
    "mlb.com": 3,
    "nhl.com": 3,
    "cbssports.com": 3,
    "foxsports.com": 3,
    "yahoo.com": 3,
    "sports.yahoo.com": 3,
    "theathletic.com": 3,
    "bleacherreport.com": 3,
    "covers.com": 3,
    "actionnetwork.com": 3,
    "draftkings.com": 3,
    "fanduel.com": 3,
    "vegasinsider.com": 3,
    "rotowire.com": 3,
    "numberfire.com": 3,
    "fantasylabs.com": 3,
}


# ---------------------------------------------------------------------------
# Direct API capabilities -- which modules serve which data types
# ---------------------------------------------------------------------------

DIRECT_API_CAPABILITIES: Dict[str, Dict[str, Set[str]]] = {
    "espn": {
        "data_types": {"schedule"},
        "leagues": {"NBA", "NFL", "MLB", "NHL", "NCAAB", "NCAAF", "WNBA"},
    },
    "odds_api": {
        "data_types": {"odds"},
        "leagues": {"NBA", "NFL", "MLB", "NHL", "NCAAB", "NCAAF"},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_trust_tier(source: str) -> int:
    """Get the trust tier for a source name or domain.

    Args:
        source: Either a direct API module name (e.g. "odds_api") or a domain
                (e.g. "basketball-reference.com") or a full URL.

    Returns:
        Trust tier integer (1-4). Defaults to 4 for unknown sources.
    """
    # Check if it's a known direct API module
    if source in DIRECT_API_CAPABILITIES:
        return 1

    # Check known internal source names
    _INTERNAL_SOURCES: Dict[str, int] = {
        "odds_api": 1,
        "espn": 1,
        "perplexity.structured": 2,
        "anthropic.web_search": 3,
    }
    if source in _INTERNAL_SOURCES:
        return _INTERNAL_SOURCES[source]

    # Try to extract domain from URL
    domain = source
    if "://" in source:
        try:
            parsed = urlparse(source)
            domain = parsed.netloc or source
        except Exception:
            pass

    # Strip www. prefix
    if domain.startswith("www."):
        domain = domain[4:]

    # Check domain trust map
    if domain in DOMAIN_TRUST:
        return DOMAIN_TRUST[domain]

    # Check if any known domain is a suffix (e.g., "stats.espn.com" -> espn.com)
    for known_domain, tier in DOMAIN_TRUST.items():
        if domain.endswith("." + known_domain):
            return tier

    return 4  # Default: general web


def get_confidence_for_tier(tier: int) -> float:
    """Get the default confidence value for a trust tier."""
    if tier in TRUST_TIERS:
        return TRUST_TIERS[tier]["confidence"]
    return 0.50
