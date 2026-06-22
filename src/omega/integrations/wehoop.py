"""
omega.integrations.wehoop — WNBA historical team-box adapter (backtest/historical).

`wehoop` (github.com/sportsdataverse/wehoop) is the richer historical source for
WNBA play-by-play, box scores, and shot locations. It is R-native; per the Phase 7
plan we port via the sportsdataverse data exports (Parquet release files) rather
than an R bridge. This module feeds **backtest artifacts and calibration**, not the
live request path — the live in-season fetch path is omega/integrations/espn_wnba.py.

ETL standards (docs/phase7 Part 5B) are inherited from omega/integrations/_etl.py:
  1. Raw Parquet is cached under data/cache/wehoop/ before transform (cached_fetch);
     a cached pull within TTL makes zero network calls and is the knowable-at-the-
     time snapshot that makes WNBA replay reproducible.
  2. Every upstream row is validated against WehoopTeamBoxRow at ingestion; a
     renamed/missing column raises SourceSchemaDriftError and the job fails loud.
  3. Team names resolve through data/aliases/WNBA.json before any artifact is built;
     unresolved teams are excluded with a data_provenance warning.

Possessions are estimated with the canonical box-score formula
``POSS ≈ FGA - OREB + TOV + 0.44*FTA``; offensive/defensive ratings are points per
100 possessions. These are standard definitions, not invented heuristics.
"""

from __future__ import annotations

import io
import logging
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
from omega.strategy.artifacts import FrozenArtifact, compute_artifact_id

logger = logging.getLogger("omega.integrations.wehoop")

# sportsdataverse wehoop WNBA team-box Parquet release. Documented best-known
# path; the operator should verify against the current release tag. Schema drift
# (renamed columns / moved URL) fails loud rather than silently coercing.
WEHOOP_TEAM_BOX_URL_TEMPLATE = (
    "https://github.com/sportsdataverse/wehoop-wnba-data/releases/download/"
    "wnba_team_box/team_box_{season}.parquet"
)

_CACHE_TTL_SECONDS = 7 * 24 * 3600  # weekly refresh cadence
_REQUEST_TIMEOUT_SECONDS = 30


class WehoopTeamBoxRow(BaseModel):
    """One team's box-score line for one WNBA game (wehoop team_box shape).

    Field names track the sportsdataverse wehoop team-box export. Extra upstream
    columns are ignored; the required ones below must be present or ingestion
    fails loud (ETL standard 2).
    """

    game_id: int
    season: int
    game_date: str
    team_display_name: str
    opponent_team_display_name: str
    home_away: str  # "home" | "away"
    team_score: int
    opponent_team_score: int
    field_goals_attempted: int
    free_throws_attempted: int
    offensive_rebounds: int
    turnovers: int


def estimate_possessions(
    field_goals_attempted: float,
    offensive_rebounds: float,
    turnovers: float,
    free_throws_attempted: float,
) -> float:
    """Canonical possessions estimate: FGA - OREB + TOV + 0.44*FTA."""
    poss = field_goals_attempted - offensive_rebounds + turnovers + 0.44 * free_throws_attempted
    return max(1.0, poss)


def _download_team_box(season: int, url_opener: Callable[..., Any]) -> Any:
    """Download the wehoop WNBA team-box Parquet for *season* into a DataFrame.

    Raw network fetch — guarded against replay mode. Wrapped by cached_fetch in
    :func:`fetch_team_box`, so within TTL this never runs.
    """
    import pandas as pd

    assert_not_replay_mode("wehoop WNBA team-box fetch")
    url = WEHOOP_TEAM_BOX_URL_TEMPLATE.format(season=season)
    logger.info("fetching wehoop WNBA team box: %s", url)
    with url_opener(url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        raw = resp.read()
    return pd.read_parquet(io.BytesIO(raw))


def fetch_team_box(
    season: int,
    *,
    cache_root: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> Any:
    """Return the wehoop WNBA team-box DataFrame for *season*, cached to Parquet."""

    @cached_fetch("wehoop", ttl_seconds=_CACHE_TTL_SECONDS, fmt="parquet", cache_root=cache_root)
    def _fetch() -> Any:
        return _download_team_box(season, url_opener)

    return _fetch(cache_key=f"wnba_team_box_{season}")


def build_wnba_artifacts(
    rows: list[dict[str, Any]],
    *,
    alias_table: dict[str, Any] | None = None,
    source: str = "wehoop",
    session_path: str | None = None,
) -> tuple[list[FrozenArtifact], list[str]]:
    """Build deterministic WNBA backtest artifacts from team-box rows.

    Validates every row (fail-loud on drift), resolves team names through the
    alias table, pairs the home/away rows of each game, derives off/def rating +
    pace from box-score possessions, and attaches the final score as the grading
    outcome.

    Returns ``(artifacts, unresolved_games)`` where ``unresolved_games`` lists
    game_ids skipped because a team could not be resolved or the home/away pair
    was incomplete.
    """
    alias_table = alias_table or {"canonical": [], "aliases": {}}
    validated = validate_records(rows, WehoopTeamBoxRow, source=source, session_path=session_path)

    # Group the two team rows per game.
    by_game: dict[int, dict[str, WehoopTeamBoxRow]] = {}
    for row in validated:
        by_game.setdefault(row.game_id, {})[row.home_away.lower()] = row

    artifacts: list[FrozenArtifact] = []
    skipped: list[str] = []

    for game_id, sides in sorted(by_game.items()):
        home = sides.get("home")
        away = sides.get("away")
        if home is None or away is None:
            skipped.append(f"{game_id} (incomplete home/away pair)")
            continue

        home_team = resolve_entity(home.team_display_name, alias_table) or home.team_display_name
        away_team = resolve_entity(away.team_display_name, alias_table) or away.team_display_name
        # Exclude only when an alias table is provided and a name is unresolved
        # (matches ETL standard 3: do not write under an ambiguous key).
        if alias_table.get("canonical") and (
            resolve_entity(home.team_display_name, alias_table) is None
            or resolve_entity(away.team_display_name, alias_table) is None
        ):
            skipped.append(
                f"{game_id} (unresolved team: {home.team_display_name} / {away.team_display_name})"
            )
            continue

        home_poss = estimate_possessions(
            home.field_goals_attempted,
            home.offensive_rebounds,
            home.turnovers,
            home.free_throws_attempted,
        )
        away_poss = estimate_possessions(
            away.field_goals_attempted,
            away.offensive_rebounds,
            away.turnovers,
            away.free_throws_attempted,
        )
        pace = round((home_poss + away_poss) / 2.0, 2)

        home_context = {
            "off_rating": round(home.team_score / home_poss * 100.0, 2),
            "def_rating": round(away.team_score / away_poss * 100.0, 2),
            "pace": pace,
        }
        away_context = {
            "off_rating": round(away.team_score / away_poss * 100.0, 2),
            "def_rating": round(home.team_score / home_poss * 100.0, 2),
            "pace": pace,
        }

        date = home.game_date[:10]
        artifacts.append(
            FrozenArtifact(
                artifact_id=compute_artifact_id(home_team, away_team, "WNBA", date),
                source_trace_id=None,
                home_team=home_team,
                away_team=away_team,
                league="WNBA",
                date=date,
                home_context=home_context,
                away_context=away_context,
                game_context={"source": "wehoop", "game_id": game_id, "season": home.season},
                odds={},  # wehoop carries no market lines; replay is sim-deterministic
                simulation_seed=42,
                outcome={"home_score": home.team_score, "away_score": away.team_score},
            )
        )

    return artifacts, skipped


def load_wnba_artifacts(
    season: int,
    *,
    cache_root: str | None = None,
    alias_table: dict[str, Any] | None = None,
    session_path: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[list[FrozenArtifact], list[str]]:
    """Fetch (cached) the wehoop WNBA team box for *season* and build artifacts."""
    df = fetch_team_box(season, cache_root=cache_root, url_opener=url_opener)
    rows = df.to_dict(orient="records")
    return build_wnba_artifacts(rows, alias_table=alias_table, session_path=session_path)
