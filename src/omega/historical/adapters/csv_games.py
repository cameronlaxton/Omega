"""Generic games CSV adapter.

Reads a games file with date/home/away (+ optional scores, season, playoff and
neutral-site flags) into canonical ``HistoricalEvent`` / ``HistoricalOutcome``.
Sport-specific adapters subclass this and override only ``COLUMN_MAP`` /
``source_name``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from omega.historical.adapters.base import CsvAdapterBase
from omega.historical.contracts import HistoricalEvent, HistoricalOutcome
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import (
    parse_datetime_utc,
    sport_family_for,
    to_bool,
    to_int_or_none,
)
from omega.integrations._etl import load_alias_table


class GamesRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str
    home_team: str
    away_team: str
    home_score: Any | None = None
    away_score: Any | None = None
    season: str | None = None
    is_playoff: Any | None = None
    is_neutral_site: Any | None = None
    explicit_swap: Any | None = None
    event_id: str | None = None


class CsvGamesAdapter(CsvAdapterBase):
    source_name = "csv_games"
    ROW_MODEL = GamesRow
    COLUMN_MAP: dict[str, str] = {}

    def __init__(self, league: str, source_name: str | None = None) -> None:
        self.league = league.upper()
        if source_name:
            self.source_name = source_name

    def _resolve(self, row: GamesRow, table: dict[str, Any]):
        start_iso = parse_datetime_utc(row.date)
        swapped = to_bool(row.explicit_swap)
        ident = resolve_event_identity(
            self.league,
            row.home_team,
            row.away_team,
            is_neutral_site=to_bool(row.is_neutral_site),
            explicit_swap=swapped,
            alias_table=table,
        )
        # Always key on the canonical event_key so events, outcomes, and odds
        # join across files/spellings. A source-supplied event_id is retained as
        # provenance only (source_row_ref), never as the join key.
        eid = event_key(self.league, start_iso, ident.home, ident.away)
        return eid, ident, start_iso, swapped

    def read_events(self, path: str | Path, **kwargs: Any) -> list[HistoricalEvent]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        family = sport_family_for(self.league)
        events: list[HistoricalEvent] = []
        for row in self.read_rows(path):
            assert isinstance(row, GamesRow)
            eid, ident, start_iso, _ = self._resolve(row, table)
            events.append(
                HistoricalEvent(
                    event_id=eid,
                    league=self.league,
                    sport_family=family,
                    season=row.season,
                    start_time=start_iso,
                    home_team=ident.home,
                    away_team=ident.away,
                    is_neutral_site=to_bool(row.is_neutral_site),
                    is_playoff=to_bool(row.is_playoff),
                    identity_status=ident.status,
                    raw_home=row.home_team,
                    raw_away=row.away_team,
                    source_name=self.source_name,
                    source_row_ref=row.event_id,
                )
            )
        return events

    def read_outcomes(self, path: str | Path, **kwargs: Any) -> list[HistoricalOutcome]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        outcomes: list[HistoricalOutcome] = []
        for row in self.read_rows(path):
            assert isinstance(row, GamesRow)
            eid, _ident, _start, swapped = self._resolve(row, table)
            home_score = to_int_or_none(row.home_score)
            away_score = to_int_or_none(row.away_score)
            if swapped:
                home_score, away_score = away_score, home_score
            if home_score is None and away_score is None:
                continue
            outcomes.append(
                HistoricalOutcome(
                    event_id=eid,
                    home_score=home_score,
                    away_score=away_score,
                    result=HistoricalOutcome.derive_result(home_score, away_score),
                    source=self.source_name,
                )
            )
        return outcomes
