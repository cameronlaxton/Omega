"""End-to-end test for omega-replay-history.

Ingest a tiny in-repo NFL fixture dataset, replay it into a dedicated DB, and
assert every persisted trace is tagged execution_mode=historical_replay, the
RUN_AUDIT.md + replay_summary.json sidecars exist, and at least one trace is
calibration-eligible. No network — local files only.
"""

from __future__ import annotations

import json
import sqlite3

from omega.historical.contracts import HistoricalEvent, HistoricalOutcome
from omega.historical.dataset_manifest import DatasetManifest
from omega.historical.identity import event_key
from omega.historical.manifests import save_dataset_manifest, save_normalized_dataset
from omega.historical.normalize import parse_datetime_utc
from omega.ops import replay_history

LEAGUE = "NFL"
FAMILY = "american_football"
MANIFEST_ID = "m-test"


def _event(date: str, home: str, away: str) -> HistoricalEvent:
    start = parse_datetime_utc(date)
    return HistoricalEvent(
        event_id=event_key(LEAGUE, start, home, away),
        league=LEAGUE,
        sport_family=FAMILY,
        start_time=start,
        home_team=home,
        away_team=away,
        source_name="test",
    )


def _outcome(ev: HistoricalEvent, hs: int, as_: int) -> HistoricalOutcome:
    return HistoricalOutcome(
        event_id=ev.event_id,
        home_score=hs,
        away_score=as_,
        result=HistoricalOutcome.derive_result(hs, as_),
    )


def _seed_dataset(root) -> None:
    e1 = _event("2023-09-10", "Team A", "Team B")
    e2 = _event("2023-09-17", "Team C", "Team A")
    e3 = _event("2023-09-24", "Team B", "Team C")
    target = _event("2023-10-01", "Team A", "Team C")
    events = [e1, e2, e3, target]
    outcomes = [
        _outcome(e1, 24, 17),
        _outcome(e2, 20, 27),
        _outcome(e3, 30, 21),
        _outcome(target, 28, 24),
    ]
    manifest = DatasetManifest(
        manifest_id=MANIFEST_ID,
        source_name="test",
        league=LEAGUE,
        sport_family=FAMILY,
    )
    save_dataset_manifest(manifest, root=root)
    save_normalized_dataset(MANIFEST_ID, events=events, outcomes=outcomes, root=root)


def test_replay_history_tags_and_audits(tmp_path):
    root = tmp_path / "historical"
    db = tmp_path / "replay_nfl.db"
    _seed_dataset(root)

    rc = replay_history.main(
        [
            "--league",
            LEAGUE,
            "--manifest-id",
            MANIFEST_ID,
            "--db",
            str(db),
            "--root",
            str(root),
            "--mode",
            "calibration",
            "--n-iterations",
            "200",
        ]
    )
    assert rc == 0

    # Every persisted trace carries the historical_replay selection tag.
    con = sqlite3.connect(str(db))
    try:
        modes = sorted({r[0] for r in con.execute("SELECT DISTINCT execution_mode FROM traces")})
        n_traces = con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    finally:
        con.close()
    assert modes == ["historical_replay"]
    assert n_traces == 4

    # Sidecars exist beside the replay manifest.
    replay_dir = root / "replays" / f"replay_{MANIFEST_ID}"
    audit = replay_dir / "RUN_AUDIT.md"
    summary_path = replay_dir / "replay_summary.json"
    assert audit.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_persisted"] == 4
    assert summary["eligible_count"] >= 1
    assert summary["context_source_distribution"].get("provided", 0) >= 1
    assert "Replay Run Audit" in audit.read_text(encoding="utf-8")
