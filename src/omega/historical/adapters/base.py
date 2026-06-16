"""Adapter base classes following the three-stage ETL standard.

Stage 1 (cache) is a no-op this pass — historical adapters read *local* files
only; ``assert_not_replay_mode`` still guards any future network fetch. Stage 2
(validate-or-fail) is enforced via ``validate_records`` against a per-adapter
Pydantic row model. Stage 3 (identity resolution) happens in ``normalize`` /
``identity`` when rows are assembled into canonical contracts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    OddsObservation,
)
from omega.integrations._etl import validate_records


@runtime_checkable
class HistoricalAdapter(Protocol):
    """Interface every historical dataset adapter implements."""

    source_name: str

    def read_events(self, path: str | Path, **kwargs: Any) -> list[HistoricalEvent]: ...

    def read_outcomes(self, path: str | Path, **kwargs: Any) -> list[HistoricalOutcome]: ...

    def read_odds(self, path: str | Path, **kwargs: Any) -> list[OddsObservation]: ...


class CsvAdapterBase:
    """Shared CSV plumbing: read → column-map → validate.

    Subclasses declare ``ROW_MODEL`` (a Pydantic model with ``extra="ignore"``)
    and ``COLUMN_MAP`` (source-column → canonical-field). They then implement the
    ``read_*`` methods by calling :meth:`read_rows` and assembling canonical
    contracts via ``normalize``/``identity``.
    """

    source_name: ClassVar[str] = "csv"
    ROW_MODEL: ClassVar[type[BaseModel]]
    COLUMN_MAP: ClassVar[dict[str, str]] = {}

    # -- stage 1: read (local file, no network) -----------------------------

    def _read_csv(self, path: str | Path) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{self.source_name}: no CSV at {p}")
        with p.open("r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    def _map_columns(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rename source columns to canonical fields via ``COLUMN_MAP``.

        Unmapped columns are preserved under their original name; row models use
        ``extra="ignore"`` so they fall away at validation without breaking.
        """
        mapped: list[dict[str, Any]] = []
        for row in rows:
            out: dict[str, Any] = dict(row)
            for src, dst in self.COLUMN_MAP.items():
                if src in row:
                    out[dst] = row[src]
            mapped.append(out)
        return mapped

    # -- stage 2: validate-or-fail at the boundary --------------------------

    def read_rows(self, path: str | Path) -> list[BaseModel]:
        """Read + column-map + validate every row against ``ROW_MODEL``.

        Raises ``SourceSchemaDriftError`` on the first row that fails validation
        (a missing/renamed required column), never coercing it to ``None``.
        """
        raw = self._read_csv(path)
        mapped = self._map_columns(raw)
        return validate_records(mapped, self.ROW_MODEL, source=self.source_name)

    def row_count(self, path: str | Path) -> int:
        return len(self._read_csv(path))

    # -- stage 3: subclasses assemble canonical contracts -------------------

    def read_events(self, path: str | Path, **kwargs: Any) -> list[HistoricalEvent]:
        raise NotImplementedError

    def read_outcomes(self, path: str | Path, **kwargs: Any) -> list[HistoricalOutcome]:
        raise NotImplementedError

    def read_odds(self, path: str | Path, **kwargs: Any) -> list[OddsObservation]:
        raise NotImplementedError
