"""Generic odds CSV adapter.

Reads per-event odds observations (date/home/away/market/selection/odds) into
``OddsObservation`` rows keyed by the same ``event_key`` the games adapter
produces, so odds join to events after identity resolution. The optional
``tier`` column lets a source label an observation as opening/closing; otherwise
tiering is decided purely by timestamp in ``odds_snapshots``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from omega.historical.adapters.base import CsvAdapterBase
from omega.historical.contracts import OddsObservation
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import (
    canonical_market,
    parse_datetime_utc,
    to_float_or_none,
)
from omega.integrations._etl import load_alias_table


class OddsRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str
    home_team: str
    away_team: str
    market: str
    selection_descriptor: str
    odds: float
    line: Any | None = None
    book: str | None = None
    timestamp: str | None = None
    tier: str | None = None


class CsvOddsAdapter(CsvAdapterBase):
    source_name = "csv_odds"
    ROW_MODEL = OddsRow
    COLUMN_MAP: dict[str, str] = {}

    def __init__(self, league: str, source_name: str | None = None) -> None:
        self.league = league.upper()
        if source_name:
            self.source_name = source_name

    def read_odds(self, path: str | Path, **kwargs: Any) -> list[OddsObservation]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        obs: list[OddsObservation] = []
        for row in self.read_rows(path):
            assert isinstance(row, OddsRow)
            date_iso = parse_datetime_utc(row.date)
            ident = resolve_event_identity(
                self.league, row.home_team, row.away_team, alias_table=table
            )
            ek = event_key(self.league, date_iso, ident.home, ident.away)
            ts = parse_datetime_utc(row.timestamp) if row.timestamp else None
            tier = row.tier.strip().lower() if row.tier else None
            tier_hint = tier if tier in ("opening", "closing") else None
            obs.append(
                OddsObservation(
                    event_key=ek,
                    market=canonical_market(row.market),
                    selection_descriptor=row.selection_descriptor,
                    odds=row.odds,
                    line=to_float_or_none(row.line),
                    book=row.book,
                    timestamp=ts,
                    tier_hint=tier_hint,
                )
            )
        return obs
