"""nflverse schedules/games CSV adapter (NFL).

Column map targets the nflverse ``games``/``schedules`` schema: ``gameday``,
``home_team``, ``away_team``, ``home_score``, ``away_score``, ``season``,
``game_type`` (REG → regular, else playoff), and ``location`` (Neutral → neutral
site). nflverse schedules carry no odds, so ``read_odds`` returns nothing —
NFL replays are probability-only unless an odds CSV is supplied separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from omega.historical.adapters.base import CsvAdapterBase
from omega.historical.contracts import HistoricalEvent, HistoricalOutcome
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import parse_datetime_utc, sport_family_for, to_int_or_none
from omega.integrations._etl import load_alias_table


class NflGamesRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gameday: str
    home_team: str
    away_team: str
    home_score: Any | None = None
    away_score: Any | None = None
    season: str | None = None
    game_type: str | None = None
    location: str | None = None
    game_id: str | None = None


class NflfastCsvAdapter(CsvAdapterBase):
    source_name = "nflfast_csv"
    ROW_MODEL = NflGamesRow

    def __init__(self, league: str = "NFL") -> None:
        self.league = league.upper()

    @staticmethod
    def _is_playoff(game_type: str | None) -> bool:
        return bool(game_type) and game_type.strip().upper() not in ("REG", "REG_SEASON")

    @staticmethod
    def _is_neutral(location: str | None) -> bool:
        return bool(location) and location.strip().lower() == "neutral"

    def _resolve(self, row: NflGamesRow, table: dict[str, Any]):
        start = parse_datetime_utc(row.gameday)
        ident = resolve_event_identity(
            self.league,
            row.home_team,
            row.away_team,
            is_neutral_site=self._is_neutral(row.location),
            alias_table=table,
        )
        eid = event_key(self.league, start, ident.home, ident.away)
        return eid, ident, start

    def read_events(self, path: str | Path, **kwargs: Any) -> list[HistoricalEvent]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        family = sport_family_for(self.league)
        events: list[HistoricalEvent] = []
        for row in self.read_rows(path):
            assert isinstance(row, NflGamesRow)
            eid, ident, start = self._resolve(row, table)
            events.append(
                HistoricalEvent(
                    event_id=eid,
                    league=self.league,
                    sport_family=family,
                    season=row.season,
                    start_time=start,
                    home_team=ident.home,
                    away_team=ident.away,
                    is_neutral_site=self._is_neutral(row.location),
                    is_playoff=self._is_playoff(row.game_type),
                    identity_status=ident.status,
                    raw_home=row.home_team,
                    raw_away=row.away_team,
                    source_name=self.source_name,
                    source_row_ref=row.game_id,
                )
            )
        return events

    def read_outcomes(self, path: str | Path, **kwargs: Any) -> list[HistoricalOutcome]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        outcomes: list[HistoricalOutcome] = []
        for row in self.read_rows(path):
            assert isinstance(row, NflGamesRow)
            eid, _ident, _start = self._resolve(row, table)
            hs = to_int_or_none(row.home_score)
            as_ = to_int_or_none(row.away_score)
            if hs is None and as_ is None:
                continue
            outcomes.append(
                HistoricalOutcome(
                    event_id=eid,
                    home_score=hs,
                    away_score=as_,
                    result=HistoricalOutcome.derive_result(hs, as_),
                    source=self.source_name,
                )
            )
        return outcomes
