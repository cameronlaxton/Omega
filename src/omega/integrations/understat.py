"""
omega.integrations.understat — current-season club xG adapter (soccer).

Understat (understat.com) publishes per-match team xG embedded as escaped JSON
inside the league page HTML (``var teamsData = JSON.parse('...')``). It is the
current-season source for club xG priors; StatsBomb Open Data remains the
frozen historical source and FBref the redundancy cross-check (Phase 7 M2,
design Part 5).

HTML-backed and fragile by nature, so the ETL standards apply strictly:
raw HTML is cached for 24h before any transform (``data/cache/understat/``),
the extracted rows validate against a Pydantic model (fail loud on drift), and
team names resolve through ``data/aliases/SOCCER.json`` before any priors
write. Cross-source disagreement above 15% emits a ``data_provenance`` warn
event rather than silently averaging (design Part 8).
"""

from __future__ import annotations

import codecs
import json
import logging
import re
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from omega.integrations._etl import cached_fetch, resolve_entity, validate_records
from omega.integrations._guards import assert_not_replay_mode
from omega.trace.priors import XgPrior

logger = logging.getLogger("omega.integrations.understat")

_BASE_URL = "https://understat.com/league"
_REQUEST_TIMEOUT_SECONDS = 30
_CACHE_TTL_SECONDS = 24 * 3600  # design Part 8: aggressive caching, daily refresh

# Omega league code -> Understat league slug (Understat covers the big five).
UNDERSTAT_LEAGUE_SLUGS: dict[str, str] = {
    "EPL": "EPL",
    "LA_LIGA": "La_liga",
    "BUNDESLIGA": "Bundesliga",
    "SERIE_A": "Serie_A",
    "LIGUE_1": "Ligue_1",
}

_TEAMS_DATA_RE = re.compile(r"teamsData\s*=\s*JSON\.parse\('(?P<blob>[^']+)'\)")


class UnderstatTeamSeason(BaseModel):
    """One team's season aggregate extracted from the league page."""

    team: str
    matches: int
    xg_for_total: float
    xg_against_total: float


def _download_league_html(
    league_slug: str, season: str, url_opener: Callable[..., Any]
) -> str:
    """Raw network fetch — replay-guarded; wrapped by cached_fetch upstream."""
    assert_not_replay_mode("understat league fetch")
    url = f"{_BASE_URL}/{league_slug}/{season}"
    logger.info("fetching understat league page: %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "omega-etl/1.0"})
    with url_opener(request, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_league_html(
    league: str,
    season: str,
    *,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> str:
    """Return the Understat league page HTML (cached, 24h TTL)."""
    slug = UNDERSTAT_LEAGUE_SLUGS.get(league.upper())
    if slug is None:
        raise ValueError(
            f"league {league!r} has no Understat slug; known: {sorted(UNDERSTAT_LEAGUE_SLUGS)}"
        )

    @cached_fetch("understat", ttl_seconds=_CACHE_TTL_SECONDS, fmt="html", cache_root=cache_root)
    def _fetch() -> str:
        return _download_league_html(slug, season, url_opener)

    return _fetch(cache_key=f"{slug}_{season}")


def parse_teams_data(html: str) -> list[dict[str, Any]]:
    """Extract per-team season xG aggregates from the page's teamsData blob.

    Returns raw dict rows (validated by the caller via ``validate_records``).
    Raises ValueError when the blob is absent — page-structure drift must fail
    loud, not return an empty list that silently empties the priors table.
    """
    match = _TEAMS_DATA_RE.search(html)
    if match is None:
        raise ValueError(
            "understat page structure drift: teamsData JSON blob not found "
            "(site deployment changed; update the parser)"
        )
    decoded = codecs.decode(match.group("blob"), "unicode_escape")
    teams = json.loads(decoded)

    rows: list[dict[str, Any]] = []
    for team_blob in teams.values():
        history = team_blob.get("history", [])
        rows.append(
            {
                "team": team_blob.get("title"),
                "matches": len(history),
                "xg_for_total": sum(float(h.get("xG", 0.0)) for h in history),
                "xg_against_total": sum(float(h.get("xGA", 0.0)) for h in history),
            }
        )
    return rows


def build_xg_priors(
    rows: list[dict[str, Any]],
    *,
    league: str,
    season: str,
    as_of_date: str,
    alias_table: dict[str, Any] | None = None,
    session_path: str | None = None,
) -> tuple[list[XgPrior], list[str]]:
    """Validate + alias-resolve raw team rows into per-game ``XgPrior`` rows."""
    validated = validate_records(
        rows, UnderstatTeamSeason, source="understat", session_path=session_path
    )
    alias_table = alias_table or {"canonical": [], "aliases": {}}
    enforce_aliases = bool(alias_table.get("canonical"))

    priors: list[XgPrior] = []
    unresolved: list[str] = []
    for row in validated:
        if row.matches <= 0:
            continue
        canonical = resolve_entity(row.team, alias_table)
        if canonical is None:
            if enforce_aliases:
                unresolved.append(row.team)
                continue
            canonical = row.team
        priors.append(
            XgPrior(
                team=canonical,
                competition=league.upper(),
                season=season,
                xg_for=round(row.xg_for_total / row.matches, 4),
                xg_against=round(row.xg_against_total / row.matches, 4),
                matches=row.matches,
                source="understat",
                as_of_date=as_of_date,
            )
        )
    if unresolved:
        logger.warning(
            "excluded %d unresolved team(s) from understat priors write: %s",
            len(unresolved),
            unresolved,
        )
    return priors, unresolved


def cross_check_xg(
    priors_by_source: dict[str, list[XgPrior]],
    *,
    threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """Compare per-team xG across sources; build warn events above *threshold*.

    Design Part 8: Understat, FBref and StatsBomb season xG agree within a few
    percent; relative disagreement beyond 15% means one source drifted or broke
    and must surface as a ``data_provenance`` audit event, never be silently
    averaged away. Comparison keys on the canonical (team, season) pair.
    """
    from datetime import datetime, timezone

    by_key: dict[tuple[str, str], dict[str, XgPrior]] = {}
    for source, priors in priors_by_source.items():
        for prior in priors:
            by_key.setdefault((prior.team, prior.season), {})[source] = prior

    events: list[dict[str, Any]] = []
    for (team, season), sources in sorted(by_key.items()):
        if len(sources) < 2:
            continue
        items = sorted(sources.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (src_a, a), (src_b, b) = items[i], items[j]
                baseline = max(abs(a.xg_for), abs(b.xg_for), 1e-9)
                rel = abs(a.xg_for - b.xg_for) / baseline
                if rel > threshold:
                    events.append(
                        {
                            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "event_type": "data_provenance",
                            "step": "xg_cross_check",
                            "status": "warn",
                            "notes": (
                                f"{team} {season}: xg_for disagreement "
                                f"{rel:.0%} between {src_a} ({a.xg_for:.2f}) and "
                                f"{src_b} ({b.xg_for:.2f}) exceeds {threshold:.0%}"
                            ),
                            "outputs": {
                                "team": team,
                                "season": season,
                                "sources": {src_a: a.xg_for, src_b: b.xg_for},
                                "relative_disagreement": round(rel, 4),
                            },
                        }
                    )
    return events


def load_xg_priors(
    league: str,
    season: str,
    *,
    as_of_date: str,
    cache_root: str | None = None,
    alias_table: dict[str, Any] | None = None,
    session_path: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[list[XgPrior], list[str]]:
    """Fetch (cached) + parse + validate one league season into XgPrior rows."""
    html = fetch_league_html(
        league, season, cache_root=cache_root, url_opener=url_opener
    )
    rows = parse_teams_data(html)
    return build_xg_priors(
        rows,
        league=league,
        season=season,
        as_of_date=as_of_date,
        alias_table=alias_table,
        session_path=session_path,
    )
