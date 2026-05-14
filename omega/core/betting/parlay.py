"""
Parlay math utilities.

Pure deterministic functions for computing combined parlay odds, probabilities,
expected value, and correlation warnings. Reuses odds conversion from odds.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from omega.core.betting.odds import american_to_decimal, implied_probability


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParlayLeg:
    """A single leg in a parlay."""

    selection: str                # e.g. "SGA Over 5 Assists"
    decimal_odds: float           # e.g. 1.30
    win_probability: float        # empirical or model-based (0-1)
    player: str = ""              # player name for correlation checking
    stat_key: str = ""            # stat category for correlation checking
    team: str = ""                # team name for correlation checking

    @classmethod
    def from_american(
        cls,
        selection: str,
        american_odds: float,
        win_probability: float,
        player: str = "",
        stat_key: str = "",
        team: str = "",
    ) -> "ParlayLeg":
        """Create a leg from American odds."""
        return cls(
            selection=selection,
            decimal_odds=american_to_decimal(american_odds),
            win_probability=win_probability,
            player=player,
            stat_key=stat_key,
            team=team,
        )


@dataclass
class ParlaySlip:
    """A complete parlay with combined metrics."""

    legs: List[ParlayLeg]
    combined_decimal_odds: float
    combined_win_probability: float
    implied_probability: float        # from combined odds
    combined_edge_pct: float          # empirical - implied, in pct points
    ev_pct: float                     # expected value as % of stake
    correlation_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Correlation knowledge
# ---------------------------------------------------------------------------

# Stat pairs for the same player that are highly correlated.
# (stat_a, stat_b) — order does not matter.
_SAME_PLAYER_CORRELATED: List[Tuple[str, str]] = [
    ("pts", "pra"),
    ("reb", "pra"),
    ("ast", "pra"),
    ("pts", "pts_reb"),
    ("pts", "pts_ast"),
    ("reb", "pts_reb"),
    ("reb", "reb_ast"),
    ("ast", "pts_ast"),
    ("ast", "reb_ast"),
]


def check_correlation(legs: List[ParlayLeg]) -> List[str]:
    """Check for correlated legs and return warning strings.

    Checks:
    1. Same player, correlated stat pairs (e.g., pts + pra)
    2. Same player, same stat at different thresholds (redundant)
    3. Same team, same stat (pace-driven correlation)
    """
    warnings: List[str] = []
    correlated_pairs = {frozenset(p) for p in _SAME_PLAYER_CORRELATED}

    for i, a in enumerate(legs):
        for b in legs[i + 1:]:
            # Same player checks
            if a.player and b.player and a.player == b.player:
                pair = frozenset((a.stat_key, b.stat_key))
                if a.stat_key == b.stat_key:
                    warnings.append(
                        f"Redundant: {a.player} has two {a.stat_key} legs"
                    )
                elif pair in correlated_pairs:
                    warnings.append(
                        f"Correlated: {a.player} {a.stat_key} + {b.stat_key} "
                        f"(overlapping counting stats)"
                    )

            # Same team, same stat
            if (
                a.team and b.team
                and a.team == b.team
                and a.stat_key == b.stat_key
                and a.player != b.player
            ):
                warnings.append(
                    f"Team correlation: {a.player} and {b.player} "
                    f"({a.team}) both on {a.stat_key}"
                )

    return warnings


# ---------------------------------------------------------------------------
# Parlay math
# ---------------------------------------------------------------------------

def compute_parlay_odds(legs: List[ParlayLeg]) -> float:
    """Compute combined decimal odds for a parlay (product of leg odds)."""
    result = 1.0
    for leg in legs:
        result *= leg.decimal_odds
    return result


def compute_parlay_probability(
    legs: List[ParlayLeg],
    independence_discount: float = 1.0,
) -> float:
    """Compute combined win probability assuming near-independence.

    Args:
        legs: The parlay legs.
        independence_discount: Multiplier < 1.0 to penalize for correlation.
            1.0 = assume full independence.

    Returns:
        Combined probability (0-1).
    """
    result = 1.0
    for leg in legs:
        result *= leg.win_probability
    return result * independence_discount


def build_parlay(
    legs: List[ParlayLeg],
    independence_discount: float = 1.0,
) -> ParlaySlip:
    """Build a complete ParlaySlip with all computed metrics.

    Args:
        legs: The parlay legs.
        independence_discount: Correlation penalty multiplier.

    Returns:
        ParlaySlip with combined odds, probability, edge, EV, and warnings.
    """
    combined_odds = compute_parlay_odds(legs)
    combined_prob = compute_parlay_probability(legs, independence_discount)
    impl_prob = 1.0 / combined_odds if combined_odds > 0 else 1.0
    edge = (combined_prob - impl_prob) * 100.0
    ev = (combined_prob * combined_odds - 1.0) * 100.0
    warnings = check_correlation(legs)

    return ParlaySlip(
        legs=legs,
        combined_decimal_odds=round(combined_odds, 4),
        combined_win_probability=round(combined_prob, 4),
        implied_probability=round(impl_prob, 4),
        combined_edge_pct=round(edge, 2),
        ev_pct=round(ev, 2),
        correlation_warnings=warnings,
    )
