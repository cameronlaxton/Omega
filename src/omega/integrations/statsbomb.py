"""
omega.integrations.statsbomb — StatsBomb Open Data adapter (backtest/historical).

StatsBomb Open Data (github.com/statsbomb/open-data) is the primary frozen,
reproducible source for the soccer quant plane (Phase 7 M2):

* match-level (home_goals, away_goals) pairs feed the Dixon-Coles rho fit
  (``omega-fit-dixon-coles`` → ``priors_dixon_coles``), filtered per
  competition profile (``fifa_intl_v1``, ``epl_v1``, ...);
* event-level shot xG aggregates feed team attack/defense priors
  (``priors_xg``).

This module never touches the live request path — it populates priors tables
that the gatherer reads. ETL standards (docs/phase7 Part 5B) come from
``omega/integrations/_etl.py``:

  1. Raw JSON is cached under ``data/cache/statsbomb/`` before transform
     (``cached_fetch``); historical matches/events use a non-expiring cache so
     the frozen snapshot is the knowable-at-the-time fit dataset.
  2. Every consumed record validates against a Pydantic model at the boundary;
     drift raises ``SourceSchemaDriftError`` and the job fails loud.
  3. Team names resolve through ``data/aliases/SOCCER.json`` before any
     ``priors_xg`` write; unresolved teams are excluded with a warning.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from omega.integrations._etl import (
    cached_fetch,
    resolve_entity,
    validate_records,
)
from omega.integrations._guards import assert_not_replay_mode
from omega.trace.priors import XgPrior

logger = logging.getLogger("omega.integrations.statsbomb")

_RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
_REQUEST_TIMEOUT_SECONDS = 30
_COMPETITIONS_TTL_SECONDS = 7 * 24 * 3600  # the index gains competitions occasionally
_FROZEN_TTL_SECONDS: float | None = None  # played matches/events never change

# Competition-profile groups for the Dixon-Coles fit. A profile id like
# "fifa_intl_v1" strips its version suffix to the group key. International
# tournament soccer has materially different draw propensity than club play,
# so groups never mix the two (design decision 5).
PROFILE_COMPETITION_NAMES: dict[str, tuple[str, ...]] = {
    "fifa_intl": (
        "FIFA World Cup",
        "UEFA Euro",
        "Copa America",
        "Africa Cup of Nations",
        "UEFA Nations League",
    ),
    "epl": ("Premier League",),
    "laliga": ("La Liga",),
    "ucl": ("Champions League", "UEFA Champions League"),
    "bundesliga": ("1. Bundesliga", "Bundesliga"),
    "seriea": ("Serie A",),
    "ligue1": ("Ligue 1",),
}


class SBCompetition(BaseModel):
    """One row of competitions.json (extra upstream fields ignored)."""

    competition_id: int
    season_id: int
    competition_name: str
    season_name: str


class _SBHomeTeamRef(BaseModel):
    home_team_name: str


class _SBAwayTeamRef(BaseModel):
    away_team_name: str


class SBMatch(BaseModel):
    """One row of matches/<competition_id>/<season_id>.json."""

    match_id: int
    match_date: str
    home_team: _SBHomeTeamRef
    away_team: _SBAwayTeamRef
    home_score: int
    away_score: int


class _SBTeamRef(BaseModel):
    name: str


class _SBShotDetail(BaseModel):
    statsbomb_xg: float


class SBShotEvent(BaseModel):
    """The subset of a Shot event the xG aggregation consumes."""

    team: _SBTeamRef
    shot: _SBShotDetail


def profile_group(profile_id: str) -> str:
    """``fifa_intl_v1`` -> ``fifa_intl`` (strip the version suffix)."""
    return re.sub(r"_v\d+$", "", profile_id)


def _download_json(url: str, url_opener: Callable[..., Any]) -> Any:
    """Raw network fetch — replay-guarded; wrapped by cached_fetch upstream."""
    assert_not_replay_mode("statsbomb open-data fetch")
    logger.info("fetching statsbomb open-data: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_competitions(
    *,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Return the competitions.json index (cached, weekly TTL)."""

    @cached_fetch(
        "statsbomb", ttl_seconds=_COMPETITIONS_TTL_SECONDS, fmt="json", cache_root=cache_root
    )
    def _fetch() -> Any:
        return _download_json(f"{_RAW_BASE}/competitions.json", url_opener)

    return _fetch(cache_key="competitions")


def fetch_matches(
    competition_id: int,
    season_id: int,
    *,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Return one season's match list (cached, frozen — never expires)."""

    @cached_fetch("statsbomb", ttl_seconds=_FROZEN_TTL_SECONDS, fmt="json", cache_root=cache_root)
    def _fetch() -> Any:
        return _download_json(
            f"{_RAW_BASE}/matches/{competition_id}/{season_id}.json", url_opener
        )

    return _fetch(cache_key=f"matches_{competition_id}_{season_id}")


def fetch_events(
    match_id: int,
    *,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Return one match's event list (cached, frozen — never expires)."""

    @cached_fetch("statsbomb", ttl_seconds=_FROZEN_TTL_SECONDS, fmt="json", cache_root=cache_root)
    def _fetch() -> Any:
        return _download_json(f"{_RAW_BASE}/events/{match_id}.json", url_opener)

    return _fetch(cache_key=f"events_{match_id}")


def select_profile_competitions(
    competitions: list[dict[str, Any]],
    profile_id: str,
    *,
    source: str = "statsbomb",
    session_path: str | None = None,
) -> list[SBCompetition]:
    """Filter the competitions index to the profile's competition group."""
    group = profile_group(profile_id)
    names = PROFILE_COMPETITION_NAMES.get(group)
    if names is None:
        raise ValueError(
            f"unknown Dixon-Coles profile group {group!r} (from {profile_id!r}); "
            f"known groups: {sorted(PROFILE_COMPETITION_NAMES)}"
        )
    validated = validate_records(
        competitions, SBCompetition, source=source, session_path=session_path
    )
    return [c for c in validated if c.competition_name in names]


def load_profile_matches(
    profile_id: str,
    *,
    cache_root: str | None = None,
    session_path: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[tuple[int, int]]:
    """Return (home_goals, away_goals) pairs for a competition profile.

    This is the Dixon-Coles fit dataset: every completed match across all
    open-data seasons of the profile's competitions. Rows validate at the
    boundary; a missing/renamed score column fails the job loud.
    """
    competitions = fetch_competitions(cache_root=cache_root, url_opener=url_opener)
    selected = select_profile_competitions(
        competitions, profile_id, session_path=session_path
    )
    pairs: list[tuple[int, int]] = []
    for comp in selected:
        raw = fetch_matches(
            comp.competition_id, comp.season_id, cache_root=cache_root, url_opener=url_opener
        )
        matches = validate_records(
            raw, SBMatch, source="statsbomb", session_path=session_path
        )
        pairs.extend((m.home_score, m.away_score) for m in matches)
        logger.info(
            "profile %s: %s %s -> %d matches",
            profile_id,
            comp.competition_name,
            comp.season_name,
            len(matches),
        )
    return pairs


def compute_team_xg_aggregates(
    matches: list[SBMatch],
    events_by_match: dict[int, list[dict[str, Any]]],
    *,
    source: str = "statsbomb",
    session_path: str | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-team xG for/against from shot events.

    Returns ``{team: {"xg_for": float, "xg_against": float, "matches": int}}``
    keyed by the *upstream* team name (alias resolution happens at the priors
    write in :func:`build_xg_priors`).
    """
    agg: dict[str, dict[str, float]] = {}

    def _bucket(team: str) -> dict[str, float]:
        return agg.setdefault(team, {"xg_for": 0.0, "xg_against": 0.0, "matches": 0})

    for match in matches:
        home = match.home_team.home_team_name
        away = match.away_team.away_team_name
        events = events_by_match.get(match.match_id, [])
        shots = [e for e in events if (e.get("type") or {}).get("name") == "Shot"]
        validated = validate_records(
            shots, SBShotEvent, source=source, session_path=session_path
        )
        match_xg = {home: 0.0, away: 0.0}
        for shot in validated:
            if shot.team.name in match_xg:
                match_xg[shot.team.name] += shot.shot.statsbomb_xg
        _bucket(home)["xg_for"] += match_xg[home]
        _bucket(home)["xg_against"] += match_xg[away]
        _bucket(home)["matches"] += 1
        _bucket(away)["xg_for"] += match_xg[away]
        _bucket(away)["xg_against"] += match_xg[home]
        _bucket(away)["matches"] += 1
    return agg


def build_xg_priors(
    aggregates: dict[str, dict[str, float]],
    *,
    competition: str,
    season: str,
    as_of_date: str,
    alias_table: dict[str, Any] | None = None,
    source: str = "statsbomb",
    session_path: str | None = None,
) -> tuple[list[XgPrior], list[str]]:
    """Convert raw per-team aggregates into per-game ``XgPrior`` rows.

    Team names resolve through the alias table when one is provided; unresolved
    teams are excluded from the write (never stored under an ambiguous key) and
    returned for operator review.
    """
    alias_table = alias_table or {"canonical": [], "aliases": {}}
    enforce_aliases = bool(alias_table.get("canonical"))
    priors: list[XgPrior] = []
    unresolved: list[str] = []
    for team, stats in sorted(aggregates.items()):
        matches = int(stats["matches"])
        if matches <= 0:
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
                competition=competition,
                season=season,
                xg_for=round(stats["xg_for"] / matches, 4),
                xg_against=round(stats["xg_against"] / matches, 4),
                matches=matches,
                source=source,
                as_of_date=as_of_date,
            )
        )
    if unresolved:
        logger.warning(
            "excluded %d unresolved team(s) from priors_xg write: %s",
            len(unresolved),
            unresolved,
        )
    return priors, unresolved
