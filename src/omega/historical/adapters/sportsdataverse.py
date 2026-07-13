"""SportsDataverse schedule-master adapters for NBA, WNBA, and NHL.

The release assets are Parquet files, so this adapter keeps Parquet reading at
the source boundary and emits Omega's existing historical contracts. Only
completed games are admitted; future/scheduled rows never enter grading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from omega.historical.contracts import HistoricalEvent, HistoricalOutcome, OddsObservation
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import parse_datetime_utc, sport_family_for, to_bool, to_int_or_none
from omega.integrations._etl import load_alias_table, validate_records

logger = logging.getLogger(__name__)


class BasketballScheduleRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    game_id: Any
    start_date: Any
    home_display_name: str
    away_display_name: str
    home_score: Any | None = None
    away_score: Any | None = None
    status_type_completed: Any
    neutral_site: Any | None = None
    season: Any | None = None
    season_type: Any | None = None


class NhlScheduleRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    game_id: Any
    game_time: Any
    home_team_abbr: str
    away_team_abbr: str
    home_score: Any | None = None
    away_score: Any | None = None
    game_state: str
    season_full: Any | None = None
    game_type: str | None = None


_NHL_TEAMS = {
    "ANA": "Anaheim Ducks", "ARI": "Arizona Coyotes", "ATL": "Atlanta Thrashers",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres", "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets", "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils", "NSH": "Nashville Predators",
    "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PHX": "Phoenix Coyotes",
    "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks", "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs", "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights", "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}


class SportsDataverseScheduleAdapter:
    """Read a local SportsDataverse ``schedule_master.parquet`` file."""

    source_name = "sportsdataverse"

    def __init__(self, league: str) -> None:
        self.league = league.upper()
        if self.league not in {"NBA", "WNBA", "NHL"}:
            raise ValueError("SportsDataverse schedule adapter supports NBA, WNBA, and NHL")

    def _read_raw(self, path: str | Path) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{self.source_name}: no Parquet file at {p}")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "SportsDataverse ingestion requires omega[integrations] (pandas + pyarrow)"
            ) from exc
        return pd.read_parquet(p).to_dict(orient="records")

    def read_rows(self, path: str | Path) -> list[BasketballScheduleRow | NhlScheduleRow]:
        model = NhlScheduleRow if self.league == "NHL" else BasketballScheduleRow
        rows = validate_records(self._read_raw(path), model, source=self.source_name)
        if self.league == "NHL":
            completed = [r for r in rows if str(r.game_state).upper() in {"OFF", "FINAL"}]
            # The release master contains repeated copies of some NHL game IDs.
            # Collapse exact copies at the source boundary; conflicting copies
            # are excluded because choosing one would silently change an outcome.
            grouped: dict[str, list[NhlScheduleRow]] = {}
            for row in completed:
                assert isinstance(row, NhlScheduleRow)
                grouped.setdefault(str(row.game_id), []).append(row)
            by_game: dict[str, NhlScheduleRow] = {}
            conflicting: set[str] = set()
            template_duplicates = 0
            for key, copies in grouped.items():
                signatures = {
                    (
                        row.game_time, row.home_team_abbr, row.away_team_abbr,
                        row.home_score, row.away_score,
                    )
                    for row in copies
                }
                if len(signatures) == 1:
                    by_game[key] = copies[0]
                    continue
                # The current master has a known duplicated block where one
                # copy of every game carries the synthetic 5-4 score. Preserve
                # the sole non-template copy; never select by row order.
                non_template = [
                    row for row in copies
                    if not (to_int_or_none(row.home_score) == 5 and to_int_or_none(row.away_score) == 4)
                ]
                if len(non_template) == 1:
                    by_game[key] = non_template[0]
                    template_duplicates += 1
                else:
                    conflicting.add(key)
            if conflicting:
                logger.warning(
                    "sportsdataverse: excluded %d conflicting duplicate NHL game IDs",
                    len(conflicting),
                )
            if template_duplicates:
                logger.warning(
                    "sportsdataverse: discarded synthetic 5-4 copies for %d NHL game IDs",
                    template_duplicates,
                )
            return list(by_game.values())
        return [r for r in rows if to_bool(r.status_type_completed)]

    def row_count(self, path: str | Path) -> int:
        return len(self.read_rows(path))

    def _basketball_aliases(self, rows: list[BasketballScheduleRow]) -> dict[str, Any]:
        table = load_alias_table(self.league)
        canonical = set(table.get("canonical", []))
        canonical.update(r.home_display_name for r in rows)
        canonical.update(r.away_display_name for r in rows)
        return {"canonical": sorted(canonical), "aliases": table.get("aliases", {})}

    def _resolved_rows(self, path: str | Path):
        rows = self.read_rows(path)
        if self.league == "NHL":
            table = {"canonical": sorted(set(_NHL_TEAMS.values())), "aliases": dict(_NHL_TEAMS)}
        else:
            table = self._basketball_aliases(rows)  # type: ignore[arg-type]
        for row in rows:
            if isinstance(row, NhlScheduleRow):
                raw_home, raw_away = row.home_team_abbr, row.away_team_abbr
                start = parse_datetime_utc(row.game_time)
                season = str(row.season_full) if row.season_full is not None else None
                playoff = str(row.game_type or "").upper() == "P"
                neutral = False
            else:
                raw_home, raw_away = row.home_display_name, row.away_display_name
                start = parse_datetime_utc(row.start_date)
                season = str(row.season) if row.season is not None else None
                playoff = to_int_or_none(row.season_type) == 3
                neutral = to_bool(row.neutral_site)
            ident = resolve_event_identity(
                self.league, raw_home, raw_away,
                is_neutral_site=neutral, alias_table=table,
            )
            yield row, ident, start, season, playoff, neutral

    def read_events(self, path: str | Path, **kwargs: Any) -> list[HistoricalEvent]:
        family = sport_family_for(self.league)
        events: list[HistoricalEvent] = []
        for row, ident, start, season, playoff, neutral in self._resolved_rows(path):
            eid = event_key(self.league, start, ident.home, ident.away)
            events.append(HistoricalEvent(
                event_id=eid, league=self.league, sport_family=family, season=season,
                start_time=start, home_team=ident.home, away_team=ident.away,
                is_neutral_site=neutral, is_playoff=playoff, identity_status=ident.status,
                raw_home=row.home_team_abbr if isinstance(row, NhlScheduleRow) else row.home_display_name,
                raw_away=row.away_team_abbr if isinstance(row, NhlScheduleRow) else row.away_display_name,
                source_name=self.source_name, source_row_ref=str(row.game_id),
            ))
        return events

    def read_outcomes(self, path: str | Path, **kwargs: Any) -> list[HistoricalOutcome]:
        outcomes: list[HistoricalOutcome] = []
        for row, ident, start, _season, _playoff, _neutral in self._resolved_rows(path):
            home_score = to_int_or_none(row.home_score)
            away_score = to_int_or_none(row.away_score)
            if home_score is None or away_score is None:
                continue
            outcomes.append(HistoricalOutcome(
                event_id=event_key(self.league, start, ident.home, ident.away),
                home_score=home_score, away_score=away_score,
                result=HistoricalOutcome.derive_result(home_score, away_score),
                source=self.source_name,
            ))
        return outcomes

    def read_odds(self, path: str | Path, **kwargs: Any) -> list[OddsObservation]:
        return []
