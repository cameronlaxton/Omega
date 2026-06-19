"""Dataset manifest determinism, hash-drift enforcement, and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from omega.historical.dataset_manifest import (
    DatasetHashDriftError,
    compute_manifest,
    verify_manifest,
)
from omega.historical.manifests import (
    list_dataset_manifests,
    load_dataset_manifest,
    save_dataset_manifest,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_manifest_id_is_deterministic(tmp_path: Path):
    f = _write(tmp_path / "games.csv", "date,home_team,away_team\n2023-09-10,A,B\n")
    m1 = compute_manifest(
        [f], source_name="csv_games", league="NFL", sport_family="american_football",
        row_counts={str(f): 1},
    )
    m2 = compute_manifest(
        [f], source_name="csv_games", league="NFL", sport_family="american_football",
        row_counts={str(f): 1},
    )
    assert m1.manifest_id == m2.manifest_id
    assert m1.schema_version == 1
    assert m1.total_rows == 1
    assert m1.files[0].sha256 == m2.files[0].sha256


def test_manifest_id_changes_with_odds_timing_class(tmp_path: Path):
    f = _write(tmp_path / "games.csv", "a,b\n1,2\n")
    m1 = compute_manifest(
        [f],
        source_name="s",
        league="NFL",
        sport_family="american_football",
        odds_timing_class="decision_time_safe",
    )
    m2 = compute_manifest(
        [f],
        source_name="s",
        league="NFL",
        sport_family="american_football",
        odds_timing_class="closing_only",
    )
    assert m1.manifest_id != m2.manifest_id


def test_manifest_id_changes_with_content(tmp_path: Path):
    f = _write(tmp_path / "games.csv", "a,b\n1,2\n")
    m1 = compute_manifest([f], source_name="s", league="NFL", sport_family="american_football")
    _write(f, "a,b\n1,3\n")
    m2 = compute_manifest([f], source_name="s", league="NFL", sport_family="american_football")
    assert m1.manifest_id != m2.manifest_id


def test_verify_passes_when_unchanged(tmp_path: Path):
    f = _write(tmp_path / "games.csv", "a,b\n1,2\n")
    m = compute_manifest([f], source_name="s", league="NFL", sport_family="american_football")
    # No exception, returns the same manifest object.
    assert verify_manifest(m, [f]) is m


def test_verify_raises_on_hash_drift(tmp_path: Path):
    f = _write(tmp_path / "games.csv", "a,b\n1,2\n")
    m = compute_manifest([f], source_name="s", league="NFL", sport_family="american_football")
    _write(f, "a,b\n9,9\n")  # edit the file after pinning
    with pytest.raises(DatasetHashDriftError):
        verify_manifest(m, [f])


def test_verify_refresh_returns_new_manifest(tmp_path: Path):
    f = _write(tmp_path / "games.csv", "a,b\n1,2\n")
    m = compute_manifest([f], source_name="s", league="NFL", sport_family="american_football")
    _write(f, "a,b\n9,9\n")
    refreshed = verify_manifest(m, [f], refresh=True)
    assert refreshed.manifest_id != m.manifest_id
    assert refreshed.files[0].sha256 != m.files[0].sha256


def test_verify_refresh_preserves_odds_timing_class(tmp_path: Path):
    f = _write(tmp_path / "games.csv", "a,b\n1,2\n")
    m = compute_manifest(
        [f],
        source_name="s",
        league="NFL",
        sport_family="american_football",
        odds_timing_class="closing_only",
    )
    _write(f, "a,b\n9,9\n")
    refreshed = verify_manifest(m, [f], refresh=True)
    assert refreshed.odds_timing_class == "closing_only"


def test_manifest_persistence_roundtrip(tmp_path: Path):
    f = _write(tmp_path / "games.csv", "a,b\n1,2\n")
    m = compute_manifest(
        [f], source_name="s", league="NFL", sport_family="american_football",
        date_range=("2023-09-01", "2023-12-31"),
        limitations=["no closing odds"],
    )
    root = tmp_path / "hist"
    save_dataset_manifest(m, root=root)
    loaded = load_dataset_manifest(m.manifest_id, root=root)
    assert loaded == m
    assert loaded.limitations == ["no closing odds"]
    assert list_dataset_manifests(root=root) == [m.manifest_id]
