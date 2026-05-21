"""Typed market snapshot rows for line-movement audit.

These rows capture provider market observations. They are not recommendations
and do not replace the `closing_lines` table used for CLV joins.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field


class MarketSnapshot(BaseModel):
    snapshot_id: str | None = Field(default=None)
    league: str
    provider: str = "the-odds-api"
    provider_event_id: str
    home_team: str
    away_team: str
    commence_time: str | None = None
    bookmaker: str
    market: str
    selection: str
    player: str | None = None
    point: float | None = None
    price: float
    snapshot_timestamp: str
    provider_last_update: str | None = None
    source: str
    schema_version: int = 1

    def stable_id(self) -> str:
        if self.snapshot_id:
            return self.snapshot_id
        raw = "|".join(
            [
                self.league.upper(),
                self.provider,
                self.provider_event_id,
                self.bookmaker,
                self.market,
                self.selection,
                self.player or "",
                "" if self.point is None else str(self.point),
                str(self.price),
                self.snapshot_timestamp,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class MarketMovement(BaseModel):
    market: str
    selection: str
    bookmaker: str
    first_timestamp: str
    last_timestamp: str
    first_point: float | None
    last_point: float | None
    first_price: float
    last_price: float
    point_delta: float | None
    price_delta: float
