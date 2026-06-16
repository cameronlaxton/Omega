"""Ingest quarantine: identity-missing + duplicate rows are rejected, not replayed."""

from __future__ import annotations

import json

from omega.historical.contracts import HistoricalEvent, HistoricalOutcome, OddsObservation
from omega.historical.quarantine import partition_events, quarantine_path, write_rejected
from omega.ops.ingest_historical_dataset import IngestBundle, _apply_clean_events


def _ev(event_id: str, *, identity_status: str = "complete", raw_home="H", raw_away="A") -> HistoricalEvent:
    return HistoricalEvent(
        event_id=event_id,
        league="NFL",
        sport_family="american_football",
        start_time="2023-09-10T00:00:00+00:00",
        home_team="H",
        away_team="A",
        identity_status=identity_status,
        raw_home=raw_home,
        raw_away=raw_away,
        source_name="test",
        source_row_ref="file.csv:2",
    )


def test_partition_rejects_missing_identity_and_duplicates():
    events = [
        _ev("good-1"),
        _ev("bad", identity_status="missing"),
        _ev("good-1"),  # duplicate of the first
        _ev("good-2"),
    ]
    clean, rejected = partition_events(events)
    assert [e.event_id for e in clean] == ["good-1", "good-2"]
    codes = sorted(r["reason_code"] for r in rejected)
    assert codes == ["duplicate_event_key", "missing_identity"]


def test_write_rejected_appends_jsonl(tmp_path):
    _clean, rejected = partition_events([_ev("bad", identity_status="missing")])
    path = write_rejected(rejected, "NFL", root=tmp_path)
    assert path == quarantine_path("NFL", root=tmp_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["reason_code"] == "missing_identity"
    assert row["league"] == "NFL"
    assert row["schema_version"] == 1
    assert "quarantined_at" in row


def test_write_rejected_noop_when_empty(tmp_path):
    path = write_rejected([], "NFL", root=tmp_path)
    assert not path.exists()


def test_apply_clean_events_keeps_canonical_duplicate_records():
    clean = [_ev("dup")]
    bundle = IngestBundle(
        events=[_ev("dup"), _ev("dup")],
        outcomes=[HistoricalOutcome(event_id="dup", home_score=1, away_score=0)],
        odds=[
            OddsObservation(
                event_key="dup",
                market="moneyline",
                selection_descriptor="home",
                odds=-110,
            )
        ],
    )
    _apply_clean_events(bundle, clean)
    assert [event.event_id for event in bundle.events] == ["dup"]
    assert [outcome.event_id for outcome in bundle.outcomes] == ["dup"]
    assert [odds.event_key for odds in bundle.odds] == ["dup"]
