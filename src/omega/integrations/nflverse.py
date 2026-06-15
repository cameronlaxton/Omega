"""
omega.integrations.nflverse — NFL weekly player-stats adapter (backtest/historical).

nflverse (github.com/nflverse) is the canonical open NFL data source. It is
R-native; per the Phase 7 plan we port via the nflverse-data Parquet release
files rather than an R bridge. This module extracts per-player, per-week yardage
observations and feeds ``omega-fit-nfl-dispersion`` (Negative-Binomial dispersion
``k`` with hierarchical shrinkage) — it never touches the live request path.

ETL standards (docs/phase7 Part 5B) are inherited from omega/integrations/_etl.py
(reference adapter: omega/integrations/wehoop.py):
  1. Raw Parquet is cached under data/cache/nflverse/ before transform
     (cached_fetch); a cached pull within TTL makes zero network calls and is the
     knowable-at-the-time snapshot that makes the dispersion fit reproducible.
  2. Every upstream row is validated against ``NflverseWeeklyStatRow`` at
     ingestion; a renamed/missing column raises ``SourceSchemaDriftError`` and the
     job fails loud rather than coercing it to ``None``.
  3. Player names resolve through data/aliases/NFL.json before any observation is
     emitted; unresolved players are excluded with a data_provenance warning.

Only the three NB-routed weekly yardage stats are emitted today
(``rushing_yards``/``receiving_yards``/``passing_yards`` — see
``DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT``). Longest-play markets
(``longest_rush``/``longest_reception``) require a play-by-play aggregation and
are intentionally out of scope for the weekly-stats export.
"""

from __future__ import annotations

import io
import logging
import math
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from omega.integrations._etl import (
    cached_fetch,
    load_alias_table,
    resolve_entities,
    validate_records,
)
from omega.integrations._guards import assert_not_replay_mode
from omega.ops.fit_nfl_dispersion import DispersionObservation

logger = logging.getLogger("omega.integrations.nflverse")

# nflverse-data weekly player-stats Parquet release. Documented best-known path;
# the operator should verify against the current release tag. A moved URL or
# renamed column fails loud (ETL standards 1/2) rather than coercing silently.
NFLVERSE_PLAYER_STATS_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats_{season}.parquet"
)

_CACHE_TTL_SECONDS = 7 * 24 * 3600  # weekly refresh cadence
_REQUEST_TIMEOUT_SECONDS = 30

# stat_type -> position groups it is a meaningful observation for. Restricting by
# position keeps each (position_group, stat_type) fit group clean (no all-zero
# lineman receiving rows) and matches the NB-routed NFL yardage stats.
_STAT_POSITION_GROUPS: dict[str, frozenset[str]] = {
    "rushing_yards": frozenset({"RB", "QB"}),
    "receiving_yards": frozenset({"WR", "TE", "RB"}),
    "passing_yards": frozenset({"QB"}),
}


class NflverseWeeklyStatRow(BaseModel):
    """One player's weekly stat line (nflverse player_stats shape).

    Required columns have no default, so a renamed/absent column fails loud at the
    ingestion boundary; their values are nullable (``| None``) because a player
    who didn't record a given stat that week legitimately carries a null — that
    is data, not schema drift. Extra upstream columns are ignored.
    """

    player_id: str
    player_display_name: str
    position_group: str | None
    season: int
    week: int
    rushing_yards: float | None
    receiving_yards: float | None
    passing_yards: float | None


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map pandas NaN floats to None so nullable values are not seen as drift.

    A genuinely absent column stays absent (and fails loud in validation); only
    present-but-NaN cells become None.
    """
    return {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in record.items()
    }


def _download_player_stats(season: int | str, url_opener: Callable[..., Any]) -> Any:
    """Download the nflverse weekly player-stats Parquet for *season*.

    Raw network fetch — guarded against replay mode. Wrapped by ``cached_fetch``
    in :func:`fetch_player_stats`, so within TTL this never runs.
    """
    import pandas as pd

    assert_not_replay_mode("nflverse player-stats fetch")
    url = NFLVERSE_PLAYER_STATS_URL_TEMPLATE.format(season=season)
    logger.info("fetching nflverse player stats: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        raw = resp.read()
    return pd.read_parquet(io.BytesIO(raw))


def fetch_player_stats(
    season: int | str,
    *,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> Any:
    """Return the nflverse weekly player-stats DataFrame for *season*, cached."""

    @cached_fetch(
        "nflverse", ttl_seconds=_CACHE_TTL_SECONDS, fmt="parquet", cache_root=cache_root
    )
    def _fetch() -> Any:
        return _download_player_stats(season, url_opener)

    return _fetch(cache_key=f"nfl_player_stats_{season}")


def build_dispersion_observations(
    rows: list[dict[str, Any]],
    *,
    alias_table: dict[str, Any] | None = None,
    source: str = "nflverse",
    session_path: str | None = None,
) -> tuple[list[DispersionObservation], list[str]]:
    """Build NB-dispersion observations from nflverse weekly stat rows.

    Validates every row (fail-loud on drift), resolves player names through the
    alias table, and emits one :class:`DispersionObservation` per
    (player, eligible stat_type) with a finite value. Returns
    ``(observations, unresolved_players)``; unresolved players are excluded (never
    written under an ambiguous key) and reported via a ``data_provenance`` warning
    when ``session_path`` is given.
    """
    alias_table = alias_table or {"canonical": [], "aliases": {}}
    cleaned = [_clean_record(r) for r in rows]
    validated = validate_records(
        cleaned, NflverseWeeklyStatRow, source=source, session_path=session_path
    )

    skill_rows = [row for row in validated if row.position_group is not None]
    unique_names = sorted({row.player_display_name for row in skill_rows})

    if alias_table.get("canonical"):
        resolved_map, unresolved = resolve_entities(
            unique_names, alias_table, source=source, session_path=session_path
        )
    else:  # no alias table -> normalize-only pass-through (matches wehoop)
        resolved_map = {name: name for name in unique_names}
        unresolved = []

    observations: list[DispersionObservation] = []
    for row in skill_rows:
        canon = resolved_map.get(row.player_display_name)
        if canon is None:
            continue  # unresolved -> excluded
        for stat_type, groups in _STAT_POSITION_GROUPS.items():
            if row.position_group not in groups:
                continue
            value = getattr(row, stat_type)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            observations.append(
                DispersionObservation(
                    entity=canon,
                    stat_type=stat_type,
                    position_group=row.position_group,
                    value=float(value),
                )
            )
    return observations, unresolved


def load_dispersion_observations(
    season: int | str,
    *,
    cache_root: str | None = None,
    alias_table: dict[str, Any] | None = None,
    session_path: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[DispersionObservation]:
    """Fetch (cached) nflverse weekly stats for *season* and build observations.

    This is the entry point ``omega-fit-nfl-dispersion`` calls for its live
    ``--season`` load. The NFL alias table is loaded by default.
    """
    if alias_table is None:
        alias_table = load_alias_table("NFL")
    df = fetch_player_stats(season, cache_root=cache_root, url_opener=url_opener)
    rows = df.to_dict(orient="records")
    observations, unresolved = build_dispersion_observations(
        rows, alias_table=alias_table, session_path=session_path
    )
    if unresolved:
        logger.warning(
            "nflverse: excluded %d unresolved players from dispersion observations",
            len(unresolved),
        )
    return observations
