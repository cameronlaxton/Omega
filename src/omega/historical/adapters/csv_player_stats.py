"""Generic player-stats CSV adapter.

Reads per-player rows used for (a) prop outcomes and (b) pre-game player-context
backfill. Rows are keyed by ``event_key`` so they join to the same events the
games adapter produces. Stat rows used as *outcomes* require a ``stat_value``;
rows used only for pre-game context may omit it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from omega.historical.adapters.base import CsvAdapterBase
from omega.historical.contracts import HistoricalPropMarket, HistoricalPropOutcome
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import parse_datetime_utc, to_bool, to_float_or_none
from omega.integrations._etl import load_alias_table


class PlayerStatRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str
    home_team: str
    away_team: str
    player_name: str
    stat_type: str
    stat_value: Any | None = None
    void: Any | None = None
    player_id: str | None = None
    team: str | None = None
    season: str | None = None


class PropMarketRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str
    home_team: str
    away_team: str
    player_name: str
    stat_type: str
    line: Any
    over_price: Any | None = None
    under_price: Any | None = None
    book: str | None = None
    timestamp: Any | None = None
    tier_hint: Any | None = None


@dataclass(frozen=True)
class PlayerStatObservation:
    """A stat row that is safe to use as pre-decision player history."""

    event_key: str
    date: str
    player_name: str
    stat_type: str
    stat_value: float
    player_id: str | None = None
    team: str | None = None
    season: str | None = None


class CsvPlayerStatsAdapter(CsvAdapterBase):
    source_name = "csv_player_stats"
    ROW_MODEL = PlayerStatRow
    COLUMN_MAP: dict[str, str] = {}

    def __init__(self, league: str, source_name: str | None = None) -> None:
        self.league = league.upper()
        if source_name:
            self.source_name = source_name

    def _event_key_for(self, row: PlayerStatRow | PropMarketRow, table: dict[str, Any]) -> str:
        date_iso = parse_datetime_utc(row.date)
        ident = resolve_event_identity(
            self.league, row.home_team, row.away_team, alias_table=table
        )
        return event_key(self.league, date_iso, ident.home, ident.away)

    @staticmethod
    def _optional_str(raw: Any | None) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return text

    @staticmethod
    def _optional_timestamp(raw: Any | None) -> str | None:
        text = CsvPlayerStatsAdapter._optional_str(raw)
        return parse_datetime_utc(text) if text is not None else None

    @staticmethod
    def _optional_tier_hint(raw: Any | None) -> Literal["opening", "closing"] | None:
        text = CsvPlayerStatsAdapter._optional_str(raw)
        if text is None:
            return None
        value = text.lower()
        if value not in {"opening", "closing"}:
            raise ValueError(f"invalid prop market tier_hint={text!r}")
        return value  # type: ignore[return-value]

    def read_stat_observations(
        self, path: str | Path, **kwargs: Any
    ) -> list[PlayerStatObservation]:
        """Return non-void stat rows with dates/event keys for as-of context builds."""
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        observations: list[PlayerStatObservation] = []
        for row in self.read_rows(path):
            assert isinstance(row, PlayerStatRow)
            if to_bool(row.void):
                continue
            value = to_float_or_none(row.stat_value)
            if value is None:
                continue
            observations.append(
                PlayerStatObservation(
                    event_key=self._event_key_for(row, table),
                    date=parse_datetime_utc(row.date),
                    player_name=row.player_name.strip(),
                    stat_type=row.stat_type.strip(),
                    stat_value=value,
                    player_id=self._optional_str(row.player_id),
                    team=self._optional_str(row.team),
                    season=self._optional_str(row.season),
                )
            )
        return observations

    def read_prop_outcomes(
        self, path: str | Path, **kwargs: Any
    ) -> dict[str, list[HistoricalPropOutcome]]:
        """Group prop outcomes by ``event_key``.

        Rows lacking a value are skipped unless explicitly marked void/DNP; void
        rows attach no-action outcomes that the calibration fitter excludes.
        """
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        grouped: dict[str, list[HistoricalPropOutcome]] = defaultdict(list)
        for row in self.read_rows(path):
            assert isinstance(row, PlayerStatRow)
            is_void = to_bool(row.void)
            value = to_float_or_none(row.stat_value)
            if value is None and not is_void:
                continue
            ek = self._event_key_for(row, table)
            grouped[ek].append(
                HistoricalPropOutcome(
                    player_name=row.player_name.strip(),
                    stat_type=row.stat_type.strip(),
                    stat_value=value,
                    void=is_void,
                )
            )
        return dict(grouped)

    def read_prop_markets(
        self, path: str | Path, **kwargs: Any
    ) -> dict[str, list[HistoricalPropMarket]]:
        """Group decision-time prop markets by ``event_key``. Rows lacking a line are skipped."""
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        grouped: dict[str, list[HistoricalPropMarket]] = defaultdict(list)
        for raw in self._read_csv(path):
            row = PropMarketRow.model_validate(raw)
            line = to_float_or_none(row.line)
            if line is None:
                continue
            ek = self._event_key_for(row, table)
            grouped[ek].append(
                HistoricalPropMarket(
                    event_key=ek,
                    player_name=row.player_name.strip(),
                    stat_type=row.stat_type.strip(),
                    line=line,
                    over_price=to_float_or_none(row.over_price),
                    under_price=to_float_or_none(row.under_price),
                    book=self._optional_str(row.book),
                    timestamp=self._optional_timestamp(row.timestamp),
                    tier_hint=self._optional_tier_hint(row.tier_hint),
                )
            )
        return dict(grouped)
