"""Generic player-stats CSV adapter.

Reads per-player rows used for (a) prop outcomes and (b) pre-game player-context
backfill. Rows are keyed by ``event_key`` so they join to the same events the
games adapter produces. Stat rows used as *outcomes* require a ``stat_value``;
rows used only for pre-game context may omit it.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from omega.historical.adapters.base import CsvAdapterBase
from omega.historical.contracts import HistoricalPropOutcome
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import parse_datetime_utc, to_float_or_none
from omega.integrations._etl import load_alias_table


class PlayerStatRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str
    home_team: str
    away_team: str
    player_name: str
    stat_type: str
    stat_value: Any | None = None


class CsvPlayerStatsAdapter(CsvAdapterBase):
    source_name = "csv_player_stats"
    ROW_MODEL = PlayerStatRow
    COLUMN_MAP: dict[str, str] = {}

    def __init__(self, league: str, source_name: str | None = None) -> None:
        self.league = league.upper()
        if source_name:
            self.source_name = source_name

    def _event_key_for(self, row: PlayerStatRow, table: dict[str, Any]) -> str:
        date_iso = parse_datetime_utc(row.date)
        ident = resolve_event_identity(
            self.league, row.home_team, row.away_team, alias_table=table
        )
        return event_key(self.league, date_iso, ident.home, ident.away)

    def read_prop_outcomes(
        self, path: str | Path, **kwargs: Any
    ) -> dict[str, list[HistoricalPropOutcome]]:
        """Group prop outcomes by ``event_key``. Rows lacking a value are skipped."""
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        grouped: dict[str, list[HistoricalPropOutcome]] = defaultdict(list)
        for row in self.read_rows(path):
            assert isinstance(row, PlayerStatRow)
            value = to_float_or_none(row.stat_value)
            if value is None:
                continue
            ek = self._event_key_for(row, table)
            grouped[ek].append(
                HistoricalPropOutcome(
                    player_name=row.player_name,
                    stat_type=row.stat_type,
                    stat_value=value,
                )
            )
        return dict(grouped)
