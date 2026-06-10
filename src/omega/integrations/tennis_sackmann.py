"""
omega.integrations.tennis_sackmann — Jeff Sackmann ATP/WTA match adapter.

The Sackmann repositories (github.com/JeffSackmann/tennis_atp, tennis_wta) are
the standard open dataset for tour-level tennis. This adapter computes the
surface-segmented rolling serve/return point-win rates (SPW%/RPW%, 12-month
half-life) that feed ``priors_tennis`` and the TennisMarkovBackend
(Phase 7 M3, design Part 5).

Local-first: ``data/tennis/`` already holds ATP singles CSVs and acts as the
pre-seeded snapshot — files found there are read directly with zero network.
Missing files (all WTA years, future ATP updates) are pulled through the ETL
cache (``data/cache/sackmann/``) from raw.githubusercontent.com.

ETL standards: every row validates against ``SackmannMatchRow`` (a renamed or
dropped column fails loud; legitimately blank serve stats — walkovers — are
tolerated and the match is excluded from the rates), and player names resolve
through ``data/aliases/TENNIS.json`` before any priors write. Rows older than
14 days relative to the repo's weekly cadence are handled by ``as_of_date``
staleness checks downstream (design Part 8).
"""

from __future__ import annotations

import csv
import io
import logging
import math
import urllib.request
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from omega.integrations._etl import cached_fetch, resolve_entity, validate_records
from omega.integrations._guards import assert_not_replay_mode
from omega.trace.priors import TennisPrior

logger = logging.getLogger("omega.integrations.tennis_sackmann")

_RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/"
    "{tour}_matches_{year}.csv"
)
_REQUEST_TIMEOUT_SECONDS = 30
_CACHE_TTL_SECONDS = 7 * 24 * 3600  # weekly refresh cadence (design Part 8)
_DEFAULT_LOCAL_ROOT = Path("data/tennis")

_HALF_LIFE_DAYS = 365.25  # 12-month half-life on point weights

TOURS = ("atp", "wta")


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


class SackmannMatchRow(BaseModel):
    """The subset of the 49-column Sackmann match row the rates consume.

    Field *presence* is mandatory (drift fails loud); serve-stat *values* may
    be blank for walkovers/retirements, in which case the match contributes
    nothing to the rates.
    """

    # Required-presence fields: `int | None` WITHOUT a default means the
    # column must exist in the CSV (drift fails loud) while a blank value
    # (walkover) coerces to None and is tolerated.
    tourney_date: int
    surface: str | None
    winner_name: str
    loser_name: str
    w_svpt: int | None
    w_1stWon: int | None
    w_2ndWon: int | None
    l_svpt: int | None
    l_1stWon: int | None
    l_2ndWon: int | None

    _coerce_blanks = field_validator(
        "surface",
        "w_svpt",
        "w_1stWon",
        "w_2ndWon",
        "l_svpt",
        "l_1stWon",
        "l_2ndWon",
        mode="before",
    )(_blank_to_none)


def _download_matches_csv(tour: str, year: int, url_opener: Callable[..., Any]) -> str:
    """Raw network fetch — replay-guarded; wrapped by cached_fetch upstream."""
    assert_not_replay_mode("sackmann match CSV fetch")
    url = _RAW_URL_TEMPLATE.format(tour=tour, year=year)
    logger.info("fetching sackmann CSV: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_matches_csv(
    tour: str,
    year: int,
    *,
    local_root: str | Path | None = None,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> str:
    """Return one season's match CSV text, local-first then cached fetch."""
    tour = tour.lower()
    if tour not in TOURS:
        raise ValueError(f"tour must be one of {TOURS}, got {tour!r}")

    local = Path(local_root or _DEFAULT_LOCAL_ROOT) / f"{tour}_matches_{year}.csv"
    if local.exists():
        return local.read_text(encoding="utf-8", errors="replace")

    @cached_fetch("sackmann", ttl_seconds=_CACHE_TTL_SECONDS, fmt="text", cache_root=cache_root)
    def _fetch() -> str:
        return _download_matches_csv(tour, year, url_opener)

    return _fetch(cache_key=f"{tour}_matches_{year}")


def parse_matches(csv_text: str, *, session_path: str | None = None) -> list[SackmannMatchRow]:
    """Parse + boundary-validate a Sackmann match CSV."""
    reader = csv.DictReader(io.StringIO(csv_text))
    raw_rows = list(reader)
    return validate_records(
        raw_rows, SackmannMatchRow, source="sackmann", session_path=session_path
    )


def _decay_weight(match_date: int, as_of: date) -> float:
    """12-month half-life weight for a YYYYMMDD tourney_date."""
    played = datetime.strptime(str(match_date), "%Y%m%d").date()
    age_days = max(0.0, (as_of - played).days)
    return 0.5 ** (age_days / _HALF_LIFE_DAYS)


def compute_rolling_rates(
    rows: list[SackmannMatchRow],
    *,
    as_of_date: str,
) -> dict[tuple[str, str], dict[str, float]]:
    """Surface-segmented weighted SPW%/RPW% per player.

    Returns ``{(player, surface): {spw_won, spw_pts, rpw_won, rpw_pts,
    matches}}`` with 12-month half-life weights on points. Matches without
    serve stats (walkovers) or without a surface contribute nothing.
    """
    as_of = date.fromisoformat(as_of_date)
    acc: dict[tuple[str, str], dict[str, float]] = {}

    def _bucket(player: str, surface: str) -> dict[str, float]:
        return acc.setdefault(
            (player, surface),
            {"spw_won": 0.0, "spw_pts": 0.0, "rpw_won": 0.0, "rpw_pts": 0.0, "matches": 0},
        )

    for row in rows:
        if row.surface is None:
            continue
        stats = (row.w_svpt, row.w_1stWon, row.w_2ndWon, row.l_svpt, row.l_1stWon, row.l_2ndWon)
        if any(s is None for s in stats) or row.w_svpt <= 0 or row.l_svpt <= 0:
            continue
        weight = _decay_weight(row.tourney_date, as_of)
        surface = row.surface.lower()

        w_serve_won = row.w_1stWon + row.w_2ndWon
        l_serve_won = row.l_1stWon + row.l_2ndWon

        winner = _bucket(row.winner_name, surface)
        winner["spw_won"] += weight * w_serve_won
        winner["spw_pts"] += weight * row.w_svpt
        winner["rpw_won"] += weight * (row.l_svpt - l_serve_won)
        winner["rpw_pts"] += weight * row.l_svpt
        winner["matches"] += 1

        loser = _bucket(row.loser_name, surface)
        loser["spw_won"] += weight * l_serve_won
        loser["spw_pts"] += weight * row.l_svpt
        loser["rpw_won"] += weight * (row.w_svpt - w_serve_won)
        loser["rpw_pts"] += weight * row.w_svpt
        loser["matches"] += 1
    return acc


def build_tennis_priors(
    rows: list[SackmannMatchRow],
    *,
    tour: str,
    as_of_date: str,
    alias_table: dict[str, Any] | None = None,
    min_matches: int = 3,
) -> tuple[list[TennisPrior], list[str]]:
    """Aggregate validated match rows into ``TennisPrior`` rows.

    Players resolve through the alias table when one is provided; unresolved
    players are excluded from the write and reported. ``min_matches`` guards
    against one-tournament noise on a surface.
    """
    alias_table = alias_table or {"canonical": [], "aliases": {}}
    enforce_aliases = bool(alias_table.get("canonical"))
    rates = compute_rolling_rates(rows, as_of_date=as_of_date)

    priors: list[TennisPrior] = []
    unresolved: set[str] = set()
    for (player, surface), bucket in sorted(rates.items()):
        if bucket["matches"] < min_matches or bucket["spw_pts"] <= 0 or bucket["rpw_pts"] <= 0:
            continue
        canonical = resolve_entity(player, alias_table)
        if canonical is None:
            if enforce_aliases:
                unresolved.add(player)
                continue
            canonical = player
        priors.append(
            TennisPrior(
                player=canonical,
                tour=tour.upper(),
                surface=surface,
                spw_pct=round(bucket["spw_won"] / bucket["spw_pts"], 4),
                rpw_pct=round(bucket["rpw_won"] / bucket["rpw_pts"], 4),
                n_matches=int(bucket["matches"]),
                as_of_date=as_of_date,
            )
        )
    if unresolved:
        logger.warning(
            "excluded %d unresolved player(s) from priors_tennis write: %s",
            len(unresolved),
            sorted(unresolved),
        )
    return priors, sorted(unresolved)


def load_tennis_priors(
    tour: str,
    years: list[int],
    *,
    as_of_date: str,
    local_root: str | Path | None = None,
    cache_root: str | None = None,
    alias_table: dict[str, Any] | None = None,
    session_path: str | None = None,
    min_matches: int = 3,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[list[TennisPrior], list[str]]:
    """Read (local-first) + parse + aggregate one tour's seasons into priors."""
    rows: list[SackmannMatchRow] = []
    for year in years:
        csv_text = fetch_matches_csv(
            tour, year, local_root=local_root, cache_root=cache_root, url_opener=url_opener
        )
        rows.extend(parse_matches(csv_text, session_path=session_path))
    return build_tennis_priors(
        rows,
        tour=tour,
        as_of_date=as_of_date,
        alias_table=alias_table,
        min_matches=min_matches,
    )
