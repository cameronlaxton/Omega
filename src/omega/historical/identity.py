"""Identity resolution for historical events.

Reuses the centralized alias-table machinery from ``omega.integrations._etl``
(``load_alias_table`` / ``resolve_entity``) so historical replay resolves teams
and players the same way the live integrations do. A historical alias overlay
can be dropped into ``data/aliases/<LEAGUE>.json`` to extend coverage.

Neutral-site policy (per plan): **preserve the source's nominal home/away
exactly — never swap.** A swap happens only when the source explicitly encodes
one (``explicit_swap=True``). Neutral sites are flagged so the snapshot builder
can neutralize/adjust the home-advantage term; the teams are not reordered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omega.historical.contracts import stable_hash
from omega.integrations._etl import load_alias_table, resolve_entity


@dataclass
class IdentityResolution:
    """Result of resolving one event's home/away identities."""

    home: str
    away: str
    status: str  # "complete" | "missing"
    reasons: list[str] = field(default_factory=list)
    failure_count: int = 0


def resolve_team(
    name: str,
    league: str,
    alias_table: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Resolve a single team/participant name to its canonical form.

    Returns ``(name, resolved)``. When the alias table cannot resolve the name
    we keep a trimmed fallback so downstream code still has *a* label, but flag
    it unresolved so the caller can fail closed or count an identity failure.
    """
    table = alias_table if alias_table is not None else load_alias_table(league)
    canon = resolve_entity(name or "", table)
    if canon is not None:
        return canon, True
    return (name or "").strip(), False


def resolve_event_identity(
    league: str,
    raw_home: str,
    raw_away: str,
    *,
    is_neutral_site: bool = False,
    explicit_swap: bool = False,
    alias_table: dict[str, Any] | None = None,
) -> IdentityResolution:
    """Resolve both sides of a fixture.

    ``is_neutral_site`` never reorders the teams — it is recorded on the event
    and consumed by the snapshot builder. Only ``explicit_swap`` (a source-
    encoded directive) swaps nominal home/away.
    """
    table = alias_table if alias_table is not None else load_alias_table(league)
    home_name, home_ok = resolve_team(raw_home, league, table)
    away_name, away_ok = resolve_team(raw_away, league, table)

    if explicit_swap:
        home_name, away_name = away_name, home_name
        home_ok, away_ok = away_ok, home_ok

    reasons: list[str] = []
    failures = 0
    if not home_ok:
        reasons.append(f"unresolved_home:{raw_home!r}")
        failures += 1
    if not away_ok:
        reasons.append(f"unresolved_away:{raw_away!r}")
        failures += 1

    status = "complete" if (home_ok and away_ok) else "missing"
    return IdentityResolution(
        home=home_name,
        away=away_name,
        status=status,
        reasons=reasons,
        failure_count=failures,
    )


def event_key(league: str, date_iso: str, home: str, away: str) -> str:
    """Stable key linking odds/player rows to a game across source spellings.

    Computed on *canonical* names + the calendar date so an odds CSV and a games
    CSV with different spellings still join after identity resolution.
    """
    return stable_hash(
        {
            "league": league.upper(),
            "date": (date_iso or "")[:10],
            "home": home,
            "away": away,
        }
    )
