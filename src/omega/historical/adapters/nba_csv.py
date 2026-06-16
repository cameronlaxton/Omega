"""NBA games CSV adapter (Kaggle-style schema).

A thin :class:`CsvGamesAdapter` subclass: the column map targets the common
Kaggle NBA ``games.csv`` headers (``GAME_DATE_EST``, ``HOME_TEAM``,
``VISITOR_TEAM``, ``PTS_home``, ``PTS_away``, ``SEASON``, optional ``PLAYOFF``),
renaming them onto the generic games fields so all the as-of/identity/snapshot
logic is reused unchanged.
"""

from __future__ import annotations

from omega.historical.adapters.csv_games import CsvGamesAdapter


class NbaCsvAdapter(CsvGamesAdapter):
    source_name = "nba_csv"
    COLUMN_MAP = {
        "GAME_DATE_EST": "date",
        "HOME_TEAM": "home_team",
        "VISITOR_TEAM": "away_team",
        "PTS_home": "home_score",
        "PTS_away": "away_score",
        "SEASON": "season",
        "PLAYOFF": "is_playoff",
    }

    def __init__(self, league: str = "NBA") -> None:
        super().__init__(league, source_name="nba_csv")
