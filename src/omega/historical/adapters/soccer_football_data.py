"""football-data.co.uk CSV adapter (soccer).

One row carries the result *and* odds, so this adapter implements events,
outcomes, and odds. Targets the canonical football-data columns: ``Date``
(DD/MM/YY[YY]), ``HomeTeam``, ``AwayTeam``, ``FTHG``/``FTAG`` (full-time goals),
``B365H``/``B365D``/``B365A`` (Bet365 3-way **decimal** odds) and
``B365>2.5``/``B365<2.5`` (over/under 2.5 goals). Decimal odds are converted to
American so they flow through the normal analyze() path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from omega.historical.adapters.base import CsvAdapterBase
from omega.historical.contracts import HistoricalEvent, HistoricalOutcome, OddsObservation
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import (
    decimal_to_american,
    parse_datetime_utc,
    sport_family_for,
    to_float_or_none,
    to_int_or_none,
)
from omega.integrations._etl import load_alias_table


class FootballDataRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Date: str
    HomeTeam: str
    AwayTeam: str
    FTHG: Any | None = None
    FTAG: Any | None = None
    FTR: str | None = None
    B365H: Any | None = None
    B365D: Any | None = None
    B365A: Any | None = None
    b365_over25: Any | None = None
    b365_under25: Any | None = None


class SoccerFootballDataAdapter(CsvAdapterBase):
    source_name = "soccer_football_data"
    ROW_MODEL = FootballDataRow
    # The over/under columns are not valid Python identifiers; map them.
    COLUMN_MAP = {"B365>2.5": "b365_over25", "B365<2.5": "b365_under25"}

    def __init__(self, league: str) -> None:
        self.league = league.upper()

    def _resolve(self, row: FootballDataRow, table: dict[str, Any]):
        start = parse_datetime_utc(row.Date)
        ident = resolve_event_identity(self.league, row.HomeTeam, row.AwayTeam, alias_table=table)
        eid = event_key(self.league, start, ident.home, ident.away)
        return eid, ident, start

    def read_events(self, path: str | Path, **kwargs: Any) -> list[HistoricalEvent]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        family = sport_family_for(self.league)
        events: list[HistoricalEvent] = []
        for row in self.read_rows(path):
            assert isinstance(row, FootballDataRow)
            eid, ident, start = self._resolve(row, table)
            events.append(
                HistoricalEvent(
                    event_id=eid,
                    league=self.league,
                    sport_family=family,
                    start_time=start,
                    home_team=ident.home,
                    away_team=ident.away,
                    identity_status=ident.status,
                    raw_home=row.HomeTeam,
                    raw_away=row.AwayTeam,
                    source_name=self.source_name,
                )
            )
        return events

    def read_outcomes(self, path: str | Path, **kwargs: Any) -> list[HistoricalOutcome]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        outcomes: list[HistoricalOutcome] = []
        for row in self.read_rows(path):
            assert isinstance(row, FootballDataRow)
            eid, _ident, _start = self._resolve(row, table)
            hs = to_int_or_none(row.FTHG)
            as_ = to_int_or_none(row.FTAG)
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

    def read_odds(self, path: str | Path, **kwargs: Any) -> list[OddsObservation]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        obs: list[OddsObservation] = []
        for row in self.read_rows(path):
            assert isinstance(row, FootballDataRow)
            eid, _ident, _start = self._resolve(row, table)

            for sel, raw in (
                ("home", row.B365H),
                ("draw", row.B365D),
                ("away", row.B365A),
            ):
                american = decimal_to_american(to_float_or_none(raw))
                if american is not None:
                    obs.append(
                        OddsObservation(
                            event_key=eid,
                            market="home_draw_away",
                            selection_descriptor=sel,
                            odds=american,
                        )
                    )

            for sel, raw in (("over_2.5", row.b365_over25), ("under_2.5", row.b365_under25)):
                american = decimal_to_american(to_float_or_none(raw))
                if american is not None:
                    obs.append(
                        OddsObservation(
                            event_key=eid,
                            market="total",
                            selection_descriptor=sel,
                            odds=american,
                            line=2.5,
                        )
                    )
        return obs
