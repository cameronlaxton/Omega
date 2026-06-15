"""
Sport family classification for calibration slicing.

Maps league codes to canonical sport families so context slices and
replay lanes are sport-aware.
"""

from omega.core.config.leagues import get_league_config

VALID_SPORT_FAMILIES = {
    "basketball",
    "american_football",
    "soccer",
    "tennis",
    "baseball",
    "hockey",
    "golf",
    "fighting",
    "esports",
    "unknown",
}


def sport_family_for_league(league: str) -> str:
    """Return the sport family for a given league code.

    Args:
        league: League code (e.g., 'NBA', 'NFL', 'EPL')

    Returns:
        One of the canonical sport families (e.g., 'basketball', 'american_football').
        Returns 'unknown' if the league is not recognized.
    """
    config = get_league_config(league)
    
    archetype = config.get("archetype")
    if archetype == "american_football":
        return "american_football"
        
    sport = config.get("sport", "unknown")
    if sport == "football" and archetype is None:
        return "american_football"
        
    if sport not in VALID_SPORT_FAMILIES:
        return "unknown"
        
    return sport
