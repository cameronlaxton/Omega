"""Shared per-game team-xG prior builder for the soccer xG adapters.

Understat and FBref produce the same ``priors_xg`` rows from season team
totals; only their upstream row model, total-field names, and source tag
differ. This holds the common validate-less core — alias resolution, the
games-gate, per-game averaging, and unresolved-team exclusion — so each
adapter just normalizes its validated rows into ``(team, xg_for_total,
xg_against_total, games)`` tuples and calls :func:`build_team_xg_priors`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from omega.integrations._etl import resolve_entity
from omega.trace.priors import XgPrior

logger = logging.getLogger("omega.integrations._soccer_xg")

# (team_name, xg_for_total, xg_against_total, games)
TeamXgTotals = tuple[str, float, float, int]


def build_team_xg_priors(
    team_totals: Iterable[TeamXgTotals],
    *,
    competition: str,
    season: str,
    as_of_date: str,
    source: str,
    alias_table: dict[str, Any] | None = None,
) -> tuple[list[XgPrior], list[str]]:
    """Build per-game ``XgPrior`` rows from season team totals.

    Teams with no games are skipped. When the alias table carries a canonical
    list, an unresolved team is excluded from the write (never stored under an
    ambiguous key) and reported; without canonical names the team name is kept
    verbatim.
    """
    alias_table = alias_table or {"canonical": [], "aliases": {}}
    enforce_aliases = bool(alias_table.get("canonical"))

    priors: list[XgPrior] = []
    unresolved: list[str] = []
    for team, xg_for_total, xg_against_total, games in team_totals:
        if games <= 0:
            continue
        canonical = resolve_entity(team, alias_table)
        if canonical is None:
            if enforce_aliases:
                unresolved.append(team)
                continue
            canonical = team
        priors.append(
            XgPrior(
                team=canonical,
                competition=competition.upper(),
                season=season,
                xg_for=round(xg_for_total / games, 4),
                xg_against=round(xg_against_total / games, 4),
                matches=games,
                source=source,
                as_of_date=as_of_date,
            )
        )
    if unresolved:
        logger.warning(
            "excluded %d unresolved team(s) from %s priors write: %s",
            len(unresolved),
            source,
            unresolved,
        )
    return priors, unresolved
