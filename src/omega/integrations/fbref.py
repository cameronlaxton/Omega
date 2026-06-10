"""
omega.integrations.fbref — FBref season xG adapter (soccer redundancy source).

FBref squad-stats pages carry season xG/xGA in the league standings table.
This adapter exists purely as the redundancy cross-check for Understat /
StatsBomb season xG (Phase 7 M2, design Part 5): disagreement above 15%
surfaces via ``understat.cross_check_xg`` as a ``data_provenance`` warn event.

FBref scraping is the most fragile of the three sources (Cloudflare, ToS) —
mitigations per design Part 8: aggressive 24h caching of the raw HTML under
``data/cache/fbref/``, weekly refresh cadence, and a parser that fails loud on
page-structure drift rather than returning silently empty rows. Rows validate
against a Pydantic model and team names resolve through
``data/aliases/SOCCER.json`` before any priors write.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from omega.integrations._etl import cached_fetch, resolve_entity, validate_records
from omega.integrations._guards import assert_not_replay_mode
from omega.trace.priors import XgPrior

logger = logging.getLogger("omega.integrations.fbref")

_BASE_URL = "https://fbref.com/en/comps"
_REQUEST_TIMEOUT_SECONDS = 30
_CACHE_TTL_SECONDS = 24 * 3600

# Omega league code -> (FBref competition id, slug).
FBREF_COMPETITIONS: dict[str, tuple[str, str]] = {
    "EPL": ("9", "Premier-League"),
    "LA_LIGA": ("12", "La-Liga"),
    "BUNDESLIGA": ("20", "Bundesliga"),
    "SERIE_A": ("11", "Serie-A"),
    "LIGUE_1": ("13", "Ligue-1"),
    "CHAMPIONS_LEAGUE": ("8", "Champions-League"),
}

# One standings row: team anchor text plus the games/xg_for/xg_against cells.
# FBref marks every cell with a data-stat attribute, which is far more stable
# than positional parsing.
_ROW_RE = re.compile(
    r'data-stat="team"[^>]*>\s*(?:<a[^>]*>)?(?P<team>[^<]+)</a>.*?'
    r'data-stat="games"[^>]*>(?P<games>\d+)<.*?'
    r'data-stat="xg_for"[^>]*>(?P<xg_for>[\d.]+)<.*?'
    r'data-stat="xg_against"[^>]*>(?P<xg_against>[\d.]+)<',
    re.DOTALL,
)


class FbrefTeamSeason(BaseModel):
    """One team's season standings line (the fields the xG prior consumes)."""

    team: str
    games: int
    xg_for: float
    xg_against: float


def _download_comp_html(
    comp_id: str, slug: str, url_opener: Callable[..., Any]
) -> str:
    """Raw network fetch — replay-guarded; wrapped by cached_fetch upstream."""
    assert_not_replay_mode("fbref competition fetch")
    url = f"{_BASE_URL}/{comp_id}/{slug}-Stats"
    logger.info("fetching fbref competition page: %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "omega-etl/1.0"})
    with url_opener(request, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_comp_html(
    league: str,
    *,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> str:
    """Return the FBref competition page HTML (cached, 24h TTL)."""
    comp = FBREF_COMPETITIONS.get(league.upper())
    if comp is None:
        raise ValueError(
            f"league {league!r} has no FBref competition; known: {sorted(FBREF_COMPETITIONS)}"
        )
    comp_id, slug = comp

    @cached_fetch("fbref", ttl_seconds=_CACHE_TTL_SECONDS, fmt="html", cache_root=cache_root)
    def _fetch() -> str:
        return _download_comp_html(comp_id, slug, url_opener)

    return _fetch(cache_key=f"{slug}_{comp_id}")


def parse_standings(html: str) -> list[dict[str, Any]]:
    """Extract team/games/xG rows from the standings table.

    Raises ValueError when no row matches — Cloudflare interstitials and page
    redesigns must fail loud, not write an empty priors set.
    """
    rows = [
        {
            "team": m.group("team").strip(),
            "games": int(m.group("games")),
            "xg_for": float(m.group("xg_for")),
            "xg_against": float(m.group("xg_against")),
        }
        for m in _ROW_RE.finditer(html)
    ]
    if not rows:
        raise ValueError(
            "fbref page structure drift: no standings rows with xg_for/xg_against "
            "found (Cloudflare interstitial or redesign; update the parser)"
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
    """Validate + alias-resolve standings rows into per-game ``XgPrior`` rows."""
    validated = validate_records(
        rows, FbrefTeamSeason, source="fbref", session_path=session_path
    )
    alias_table = alias_table or {"canonical": [], "aliases": {}}
    enforce_aliases = bool(alias_table.get("canonical"))

    priors: list[XgPrior] = []
    unresolved: list[str] = []
    for row in validated:
        if row.games <= 0:
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
                xg_for=round(row.xg_for / row.games, 4),
                xg_against=round(row.xg_against / row.games, 4),
                matches=row.games,
                source="fbref",
                as_of_date=as_of_date,
            )
        )
    if unresolved:
        logger.warning(
            "excluded %d unresolved team(s) from fbref priors write: %s",
            len(unresolved),
            unresolved,
        )
    return priors, unresolved


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
    """Fetch (cached) + parse + validate one competition into XgPrior rows."""
    html = fetch_comp_html(league, cache_root=cache_root, url_opener=url_opener)
    rows = parse_standings(html)
    return build_xg_priors(
        rows,
        league=league,
        season=season,
        as_of_date=as_of_date,
        alias_table=alias_table,
        session_path=session_path,
    )
