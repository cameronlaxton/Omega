"""Sackmann ATP match CSV adapter (tennis).

Tennis is winner/loser, not home/away — so home/away are assigned
**deterministically and outcome-independently** by sorting the two player names
(home = alphabetically first). The outcome then records which side actually won,
never the seating. Per-player serve/return point tallies are emitted as
``TeamGameRow`` history (``read_serve_history``) so the snapshot builder can
derive ``serve_win_pct`` / ``return_win_pct`` — no generic score-variance logic.

Targets the Sackmann schema: ``tourney_date`` (YYYYMMDD), ``surface``,
``best_of``, ``winner_name``, ``loser_name``, and the serve columns
``w_svpt``/``w_1stWon``/``w_2ndWon`` (and ``l_*``). Sackmann carries no odds, so
tennis replays are probability-only.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from omega.historical.adapters.base import CsvAdapterBase
from omega.historical.contracts import HistoricalEvent, HistoricalOutcome
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import parse_datetime_utc, sport_family_for, to_float_or_none
from omega.historical.snapshots import TeamGameRow
from omega.integrations._etl import load_alias_table


class TennisRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tourney_date: str
    winner_name: str
    loser_name: str
    surface: str | None = None
    best_of: Any | None = None
    w_svpt: Any | None = None
    w_1stWon: Any | None = None
    w_2ndWon: Any | None = None
    l_svpt: Any | None = None
    l_1stWon: Any | None = None
    l_2ndWon: Any | None = None


class TennisAtpCsvAdapter(CsvAdapterBase):
    source_name = "tennis_atp_csv"
    ROW_MODEL = TennisRow

    def __init__(self, league: str = "ATP") -> None:
        self.league = league.upper()

    def _resolve(self, row: TennisRow, table: dict[str, Any]):
        start = parse_datetime_utc(row.tourney_date)
        # Outcome-independent seating: alphabetical order, never winner-first.
        home_raw, away_raw = sorted([row.winner_name, row.loser_name])
        ident = resolve_event_identity(self.league, home_raw, away_raw, alias_table=table)
        eid = event_key(self.league, start, ident.home, ident.away)
        home_won = home_raw == row.winner_name
        return eid, ident, start, home_won

    def read_events(self, path: str | Path, **kwargs: Any) -> list[HistoricalEvent]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        family = sport_family_for(self.league)
        events: list[HistoricalEvent] = []
        for row in self.read_rows(path):
            assert isinstance(row, TennisRow)
            eid, ident, start, _ = self._resolve(row, table)
            events.append(
                HistoricalEvent(
                    event_id=eid,
                    league=self.league,
                    sport_family=family,
                    start_time=start,
                    home_team=ident.home,
                    away_team=ident.away,
                    identity_status=ident.status,
                    raw_home=ident.home,
                    raw_away=ident.away,
                    source_name=self.source_name,
                )
            )
        return events

    def read_outcomes(self, path: str | Path, **kwargs: Any) -> list[HistoricalOutcome]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        outcomes: list[HistoricalOutcome] = []
        for row in self.read_rows(path):
            assert isinstance(row, TennisRow)
            eid, _ident, _start, home_won = self._resolve(row, table)
            hs, as_ = (1, 0) if home_won else (0, 1)
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

    def read_extra_context(self, path: str | Path, **kwargs: Any) -> dict[str, dict]:
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        extra: dict[str, dict] = {}
        for row in self.read_rows(path):
            assert isinstance(row, TennisRow)
            eid, _ident, _start, _ = self._resolve(row, table)
            ctx: dict[str, Any] = {}
            if row.surface:
                ctx["surface"] = str(row.surface).strip().lower()
            bo = to_float_or_none(row.best_of)
            if bo is not None:
                ctx["best_of"] = int(bo)
            if ctx:
                extra[eid] = ctx
        return extra

    def read_serve_history(self, path: str | Path, **kwargs: Any) -> dict[str, list[TeamGameRow]]:
        """Per-player serve/return point history (keyed by canonical player name)."""
        table = kwargs.get("alias_table") or load_alias_table(self.league)
        hist: dict[str, list[TeamGameRow]] = defaultdict(list)
        for row in self.read_rows(path):
            assert isinstance(row, TennisRow)
            start = parse_datetime_utc(row.tourney_date)
            home_raw, away_raw = sorted([row.winner_name, row.loser_name])
            ident = resolve_event_identity(self.league, home_raw, away_raw, alias_table=table)
            winner = ident.home if home_raw == row.winner_name else ident.away
            loser = ident.away if home_raw == row.winner_name else ident.home

            w_svpt = to_float_or_none(row.w_svpt)
            w_won = (to_float_or_none(row.w_1stWon) or 0) + (to_float_or_none(row.w_2ndWon) or 0)
            l_svpt = to_float_or_none(row.l_svpt)
            l_won = (to_float_or_none(row.l_1stWon) or 0) + (to_float_or_none(row.l_2ndWon) or 0)
            if not w_svpt or not l_svpt:
                continue

            hist[winner].append(
                TeamGameRow(
                    date=start,
                    serve_points_won=w_won,
                    serve_points_total=w_svpt,
                    return_points_won=l_svpt - l_won,
                    return_points_total=l_svpt,
                )
            )
            hist[loser].append(
                TeamGameRow(
                    date=start,
                    serve_points_won=l_won,
                    serve_points_total=l_svpt,
                    return_points_won=w_svpt - w_won,
                    return_points_total=w_svpt,
                )
            )
        return dict(hist)
