"""Retrosheet 161-column game-log ZIP adapter for MLB historical replay."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    OddsObservation,
    stable_hash,
)
from omega.historical.identity import event_key, resolve_event_identity
from omega.historical.normalize import parse_datetime_utc, sport_family_for, to_int_or_none
from omega.integrations._etl import validate_records


class RetrosheetGameRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str
    game_number: str
    away_team_code: str
    home_team_code: str
    away_score: str
    home_score: str
    source_row_ref: str


_MLB_TEAM_CODES = {
    "ANA": "Los Angeles Angels", "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHA": "Chicago White Sox", "CHN": "Chicago Cubs",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCA": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAN": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYA": "New York Yankees",
    "NYN": "New York Mets", "OAK": "Oakland Athletics", "ATH": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SDN": "San Diego Padres",
    "SEA": "Seattle Mariners", "SFN": "San Francisco Giants", "SLN": "St. Louis Cardinals",
    "TBA": "Tampa Bay Rays", "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays",
    "WAS": "Washington Nationals",
}


class RetrosheetGameLogAdapter:
    source_name = "retrosheet"

    def __init__(self, league: str = "MLB") -> None:
        self.league = league.upper()
        if self.league != "MLB":
            raise ValueError("Retrosheet game-log adapter supports MLB")

    def source_files(self, path: str | Path) -> list[Path]:
        p = Path(path)
        if p.is_dir():
            files = sorted(p.glob("gl*.zip")) + sorted(p.glob("GL*.zip"))
        elif p.is_file():
            files = [p]
        else:
            raise FileNotFoundError(f"{self.source_name}: no ZIP or directory at {p}")
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"{self.source_name}: no gl*.zip files under {p}")
        return files

    def _read_raw(self, path: str | Path) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for archive_path in self.source_files(path):
            with zipfile.ZipFile(archive_path) as archive:
                members = sorted(n for n in archive.namelist() if n.upper().endswith(".TXT"))
                for member in members:
                    text = archive.read(member).decode("latin-1")
                    for line_no, row in enumerate(csv.reader(io.StringIO(text)), start=1):
                        if len(row) < 11:
                            raise ValueError(f"{archive_path}:{member}:{line_no}: expected 11+ columns")
                        records.append({
                            "date": row[0], "game_number": row[1],
                            "away_team_code": row[3], "home_team_code": row[6],
                            "away_score": row[9], "home_score": row[10],
                            "source_row_ref": f"{archive_path.name}:{member}:{line_no}",
                        })
        return records

    def read_rows(self, path: str | Path) -> list[RetrosheetGameRow]:
        rows = validate_records(self._read_raw(path), RetrosheetGameRow, source=self.source_name)
        return [r for r in rows if isinstance(r, RetrosheetGameRow)]

    def row_count(self, path: str | Path) -> int:
        return len(self.read_rows(path))

    def _resolve(self, row: RetrosheetGameRow):
        try:
            raw_home = _MLB_TEAM_CODES[row.home_team_code]
            raw_away = _MLB_TEAM_CODES[row.away_team_code]
        except KeyError as exc:
            raise ValueError(f"Unknown Retrosheet team code: {exc.args[0]}") from exc
        table = {"canonical": sorted(set(_MLB_TEAM_CODES.values())), "aliases": _MLB_TEAM_CODES}
        start = parse_datetime_utc(row.date)
        return resolve_event_identity(self.league, raw_home, raw_away, alias_table=table), start

    def _event_id(self, row: RetrosheetGameRow, start: str, home: str, away: str) -> str:
        base = event_key(self.league, start, home, away)
        return stable_hash({"base_event_key": base, "game_number": row.game_number})

    def read_events(self, path: str | Path, **kwargs: Any) -> list[HistoricalEvent]:
        family = sport_family_for(self.league)
        events: list[HistoricalEvent] = []
        for row in self.read_rows(path):
            ident, start = self._resolve(row)
            events.append(HistoricalEvent(
                event_id=self._event_id(row, start, ident.home, ident.away),
                league=self.league, sport_family=family, season=row.date[:4], start_time=start,
                home_team=ident.home, away_team=ident.away, identity_status=ident.status,
                raw_home=row.home_team_code, raw_away=row.away_team_code,
                source_name=self.source_name, source_row_ref=row.source_row_ref,
            ))
        return events

    def read_outcomes(self, path: str | Path, **kwargs: Any) -> list[HistoricalOutcome]:
        outcomes: list[HistoricalOutcome] = []
        for row in self.read_rows(path):
            ident, start = self._resolve(row)
            home_score, away_score = to_int_or_none(row.home_score), to_int_or_none(row.away_score)
            outcomes.append(HistoricalOutcome(
                event_id=self._event_id(row, start, ident.home, ident.away),
                home_score=home_score, away_score=away_score,
                result=HistoricalOutcome.derive_result(home_score, away_score),
                source=self.source_name,
            ))
        return outcomes

    def read_odds(self, path: str | Path, **kwargs: Any) -> list[OddsObservation]:
        return []
