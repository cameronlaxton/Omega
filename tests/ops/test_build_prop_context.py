"""omega-build-prop-context CLI artifact write path."""

from __future__ import annotations

import json

from omega.historical.contracts import HistoricalEvent, HistoricalOutcome, HistoricalPropMarket
from omega.historical.dataset_manifest import DatasetManifest
from omega.historical.identity import event_key
from omega.historical.manifests import save_dataset_manifest, save_normalized_dataset
from omega.historical.normalize import parse_datetime_utc
from omega.ops import build_prop_context

LG, FAM = "NBA", "basketball"


def _event(date: str, home: str, away: str) -> HistoricalEvent:
    start = parse_datetime_utc(date)
    return HistoricalEvent(
        event_id=event_key(LG, start, home, away),
        league=LG,
        sport_family=FAM,
        season="2024",
        start_time=start,
        home_team=home,
        away_team=away,
        source_name="test",
    )


def test_build_prop_context_cli_writes_dataset_artifacts(tmp_path):
    root = tmp_path / "historical"
    manifest = DatasetManifest(
        manifest_id="m-props",
        source_name="test",
        league=LG,
        sport_family=FAM,
    )
    e1 = _event("2024-01-01", "A", "B")
    e2 = _event("2024-01-08", "C", "A")
    target = _event("2024-01-22", "A", "E")
    market = HistoricalPropMarket(
        event_key=target.event_id,
        player_name="Player A",
        stat_type="pts",
        line=24.5,
    )
    save_dataset_manifest(manifest, root=root)
    save_normalized_dataset(
        manifest.manifest_id,
        events=[e1, e2, target],
        outcomes=[HistoricalOutcome(event_id=target.event_id, home_score=100, away_score=90)],
        prop_markets={target.event_id: [market]},
        root=root,
    )
    stats = tmp_path / "stats.csv"
    stats.write_text(
        "date,home_team,away_team,player_name,stat_type,stat_value,player_id,season\n"
        "2024-01-01,A,B,Player A,pts,20,p1,2024\n"
        "2024-01-08,C,A,Player A,pts,24,p1,2024\n"
        "2024-01-22,A,E,Player A,pts,100,p1,2024\n",
        encoding="utf-8",
    )

    rc = build_prop_context.main(
        [
            "--manifest-id",
            manifest.manifest_id,
            "--player-stats",
            str(stats),
            "--root",
            str(root),
            "--min-history-games",
            "2",
        ]
    )

    assert rc == 0
    base = root / "datasets" / manifest.manifest_id
    context = json.loads((base / "prop_context.json").read_text(encoding="utf-8"))
    audit = json.loads((base / "prop_context_audit.json").read_text(encoding="utf-8"))
    only_entry = next(iter(context.values()))
    assert only_entry["pts_mean"] == 22.0
    assert only_entry["sample_size"] == 2
    assert audit["schema_version"] == 1
    assert audit["missing_context_rate"] == 0.0
