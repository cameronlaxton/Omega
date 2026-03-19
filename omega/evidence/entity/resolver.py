"""
Entity resolver — canonicalizes team and player names.

Uses the static alias database for exact/prefix matches, then falls back
to ``difflib.get_close_matches`` for typos and abbreviations the alias
list doesn't explicitly cover.

Usage::

    resolver = EntityResolver()
    result = resolver.resolve("Sixers", league="NBA")
    assert result.canonical == "Philadelphia 76ers"
    assert result.abbreviation == "PHI"
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from omega.evidence.entity.aliases import LEAGUE_TEAMS, TeamRecord


@dataclass(frozen=True)
class ResolvedEntity:
    """Result of entity resolution."""

    canonical: str          # "Philadelphia 76ers"
    abbreviation: str       # "PHI"
    league: Optional[str]   # "NBA"
    confidence: float       # 1.0 = exact alias hit, 0.8 = fuzzy, 0.5 = pass-through


class EntityResolver:
    """Resolves user-facing team/player names to canonical forms.

    Build the lookup index once, then call ``resolve()`` per entity.
    Thread-safe after construction (read-only lookups).
    """

    def __init__(self) -> None:
        # alias (lowercased) → (TeamRecord, league)
        self._alias_index: Dict[str, Tuple[TeamRecord, str]] = {}
        # canonical (lowercased) → (TeamRecord, league)
        self._canonical_index: Dict[str, Tuple[TeamRecord, str]] = {}
        # abbreviation (uppercased) → (TeamRecord, league) — per league to avoid cross-league collisions
        self._abbrev_index: Dict[Tuple[str, str], Tuple[TeamRecord, str]] = {}
        # all canonical names for fuzzy matching
        self._all_canonicals: List[str] = []

        self._build_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        name: str,
        league: Optional[str] = None,
    ) -> ResolvedEntity:
        """Resolve *name* to a canonical entity.

        Resolution order:
        1. Exact canonical match (case-insensitive)
        2. Exact alias match (case-insensitive)
        3. Abbreviation match (case-insensitive, league-scoped if given)
        4. Substring containment (canonical name contains query or vice versa)
        5. Fuzzy match via difflib
        6. Pass-through (return the input unchanged with low confidence)
        """
        if not name or not name.strip():
            return ResolvedEntity(canonical=name, abbreviation="", league=league, confidence=0.0)

        norm = name.strip().lower()

        # 1. Exact canonical
        hit = self._canonical_index.get(norm)
        if hit:
            rec, lg = hit
            if league is None or lg == league.upper():
                return self._to_resolved(rec, lg, 1.0)

        # 2. Exact alias
        hit = self._alias_index.get(norm)
        if hit:
            rec, lg = hit
            if league is None or lg == league.upper():
                return self._to_resolved(rec, lg, 1.0)

        # 3. Abbreviation
        upper = name.strip().upper()
        if league:
            key = (upper, league.upper())
            hit_ab = self._abbrev_index.get(key)
            if hit_ab:
                rec, lg = hit_ab
                return self._to_resolved(rec, lg, 0.95)
        else:
            # Try all leagues for abbreviation
            for (abbr, lg), (rec, _) in self._abbrev_index.items():
                if abbr == upper:
                    return self._to_resolved(rec, lg, 0.90)

        # 4. Substring containment (league-scoped if provided)
        sub = self._try_substring(norm, league)
        if sub:
            return sub

        # 5. Fuzzy match
        fuzzy = self._try_fuzzy(norm, league)
        if fuzzy:
            return fuzzy

        # 6. Pass-through
        return ResolvedEntity(
            canonical=name.strip(),
            abbreviation="",
            league=league.upper() if league else None,
            confidence=0.5,
        )

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        for league, teams in LEAGUE_TEAMS.items():
            for rec in teams:
                canon_lower = rec.canonical.lower()
                self._canonical_index[canon_lower] = (rec, league)
                self._all_canonicals.append(rec.canonical)

                for alias in rec.aliases:
                    # Don't overwrite if another league already claimed this alias
                    # (e.g. "ATL" is both Hawks and Falcons).  League-scoped
                    # resolution handles ambiguity.
                    if alias not in self._alias_index:
                        self._alias_index[alias] = (rec, league)

                self._abbrev_index[(rec.abbreviation.upper(), league)] = (rec, league)

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_resolved(rec: TeamRecord, league: str, confidence: float) -> ResolvedEntity:
        return ResolvedEntity(
            canonical=rec.canonical,
            abbreviation=rec.abbreviation,
            league=league,
            confidence=confidence,
        )

    def _try_substring(self, norm: str, league: Optional[str]) -> Optional[ResolvedEntity]:
        """Check if *norm* is a meaningful substring of any canonical name."""
        if len(norm) < 3:
            return None

        candidates: List[Tuple[TeamRecord, str]] = []
        for canon_lower, (rec, lg) in self._canonical_index.items():
            if league and lg != league.upper():
                continue
            if norm in canon_lower or canon_lower in norm:
                candidates.append((rec, lg))

        if len(candidates) == 1:
            rec, lg = candidates[0]
            return self._to_resolved(rec, lg, 0.85)
        return None

    def _try_fuzzy(self, norm: str, league: Optional[str]) -> Optional[ResolvedEntity]:
        """Use difflib for approximate matching."""
        pool = self._all_canonicals
        if league:
            league_upper = league.upper()
            pool = [
                c for c in self._all_canonicals
                if self._canonical_index.get(c.lower(), (None, ""))[1] == league_upper
            ]

        matches = difflib.get_close_matches(norm, [c.lower() for c in pool], n=1, cutoff=0.6)
        if matches:
            hit = self._canonical_index.get(matches[0])
            if hit:
                rec, lg = hit
                return self._to_resolved(rec, lg, 0.75)
        return None


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_default_resolver: Optional[EntityResolver] = None


def get_resolver() -> EntityResolver:
    """Return the module-level singleton resolver (lazy-init)."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = EntityResolver()
    return _default_resolver
