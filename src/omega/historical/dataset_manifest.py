"""Dataset manifest: identity + integrity contract for a historical dataset.

Every ingested dataset is pinned by a manifest containing per-file sha256
hashes, row counts, league/sport_family, date range, and documented
limitations. Re-importing a dataset whose file hashes changed fails closed
unless an explicit refresh is requested — this keeps replay deterministic and
makes silent upstream edits impossible to ignore.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omega.historical.contracts import stable_hash

UTC = timezone.utc

_HASH_CHUNK = 1 << 20  # 1 MiB


class DatasetFileRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path as supplied at ingest time")
    sha256: str
    row_count: int = Field(default=0, ge=0)
    bytes: int = Field(default=0, ge=0)


class DatasetManifest(BaseModel):
    """Immutable description of an ingested historical dataset."""

    model_config = ConfigDict(extra="forbid")

    manifest_id: str = Field(description="Deterministic id over source + file hashes + config")
    source_name: str
    league: str
    sport_family: str
    date_range_start: str | None = Field(default=None, description="Earliest event date (ISO)")
    date_range_end: str | None = Field(default=None, description="Latest event date (ISO)")
    total_rows: int = Field(default=0, ge=0)
    files: list[DatasetFileRef] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=list,
        description="Known gaps/biases: e.g. 'no closing odds', 'home/away nominal only'",
    )
    odds_timing_class: str | None = Field(
        default=None, description="decision_time_safe | closing_only | timing_unknown"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def file_hash_index(self) -> dict[str, str]:
        return {f.path: f.sha256 for f in self.files}


class DatasetHashDriftError(RuntimeError):
    """Raised when on-disk file hashes no longer match the recorded manifest."""

    def __init__(self, drifted: dict[str, tuple[str, str]]):
        self.drifted = drifted
        detail = ", ".join(
            f"{path}: manifest={old[:12]}… disk={new[:12]}…"
            for path, (old, new) in drifted.items()
        )
        super().__init__(f"dataset file hash drift detected: {detail}")


def hash_file(path: str | Path) -> tuple[str, int]:
    """Return (sha256_hexdigest, byte_size) for a file, streamed in chunks."""
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


def compute_manifest(
    files: list[str | Path],
    *,
    source_name: str,
    league: str,
    sport_family: str,
    row_counts: dict[str, int] | None = None,
    date_range: tuple[str | None, str | None] = (None, None),
    limitations: list[str] | None = None,
) -> DatasetManifest:
    """Build a :class:`DatasetManifest` from a set of source files.

    ``row_counts`` maps the supplied path string to a parsed row count (adapters
    know how to count rows; this module only owns hashing and identity).
    """
    row_counts = row_counts or {}
    refs: list[DatasetFileRef] = []
    for f in sorted(str(x) for x in files):
        digest, size = hash_file(f)
        refs.append(
            DatasetFileRef(
                path=f,
                sha256=digest,
                row_count=int(row_counts.get(f, 0)),
                bytes=size,
            )
        )

    total_rows = sum(r.row_count for r in refs)
    manifest_id = stable_hash(
        {
            "source_name": source_name,
            "league": league.upper(),
            "sport_family": sport_family,
            "files": [(r.path, r.sha256) for r in refs],
        },
        length=24,
    )
    return DatasetManifest(
        manifest_id=manifest_id,
        source_name=source_name,
        league=league.upper(),
        sport_family=sport_family,
        date_range_start=date_range[0],
        date_range_end=date_range[1],
        total_rows=total_rows,
        files=refs,
        limitations=limitations or [],
    )


def verify_manifest(
    manifest: DatasetManifest,
    files: list[str | Path] | None = None,
    *,
    refresh: bool = False,
) -> DatasetManifest:
    """Re-hash the manifest's files and confirm they match.

    Raises :class:`DatasetHashDriftError` on any mismatch unless ``refresh`` is
    set, in which case a *new* manifest reflecting the current files is returned.
    The unchanged manifest is returned when everything matches.
    """
    targets = [str(x) for x in files] if files is not None else [f.path for f in manifest.files]
    drifted: dict[str, tuple[str, str]] = {}
    recorded = manifest.file_hash_index()
    for path in targets:
        disk_hash, _ = hash_file(path)
        old = recorded.get(path)
        if old is not None and old != disk_hash:
            drifted[path] = (old, disk_hash)

    if drifted and not refresh:
        raise DatasetHashDriftError(drifted)

    if drifted and refresh:
        return compute_manifest(
            targets,
            source_name=manifest.source_name,
            league=manifest.league,
            sport_family=manifest.sport_family,
            row_counts={f.path: f.row_count for f in manifest.files},
            date_range=(manifest.date_range_start, manifest.date_range_end),
            limitations=manifest.limitations,
        )
    return manifest
