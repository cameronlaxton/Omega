"""omega-ingest-historical-dataset — ingest a local historical CSV dataset.

Reads source CSVs through the matching adapter, resolves identities, computes a
:class:`DatasetManifest` (pinning per-file hashes + row counts + date range),
and persists the normalized, identity-resolved dataset for replay. No network
this pass — local files only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omega.historical.adapters.csv_games import CsvGamesAdapter
from omega.historical.adapters.csv_odds import CsvOddsAdapter
from omega.historical.adapters.csv_player_stats import CsvPlayerStatsAdapter
from omega.historical.adapters.nba_csv import NbaCsvAdapter
from omega.historical.adapters.nflfast_csv import NflfastCsvAdapter
from omega.historical.adapters.soccer_football_data import SoccerFootballDataAdapter
from omega.historical.adapters.tennis_atp_csv import TennisAtpCsvAdapter
from omega.historical.contracts import HistoricalEvent, HistoricalOutcome, OddsObservation
from omega.historical.dataset_manifest import compute_manifest
from omega.historical.manifests import save_dataset_manifest, save_normalized_dataset
from omega.historical.normalize import sport_family_for
from omega.historical.odds_timing import timing_class_for_source
from omega.historical.quarantine import partition_events, write_rejected
from omega.historical.snapshots import TeamGameRow

logger = logging.getLogger("omega.ops.ingest_historical_dataset")

SOURCES = ("nflfast", "nba_csv", "football_data", "tennis_atp", "csv_games")


@dataclass
class IngestBundle:
    events: list[HistoricalEvent]
    outcomes: list[HistoricalOutcome]
    odds: list[OddsObservation] = field(default_factory=list)
    extra_context: dict[str, dict] = field(default_factory=dict)
    history_override: dict[str, list[TeamGameRow]] | None = None
    prop_markets: dict[str, list[HistoricalPropMarket]] = field(default_factory=dict)
    prop_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)


def _merge_prop_outcomes(
    outcomes: list[HistoricalOutcome], po_by_event: dict[str, list]
) -> list[HistoricalOutcome]:
    """Attach prop outcomes onto matching event outcomes (or create new ones)."""
    by_id = {o.event_id: o for o in outcomes}
    orphan_count = 0
    for ek, pos in po_by_event.items():
        if ek in by_id:
            by_id[ek] = by_id[ek].model_copy(update={"prop_outcomes": pos})
        else:
            orphan_count += 1
            by_id[ek] = HistoricalOutcome(event_id=ek, prop_outcomes=pos)
    if orphan_count:
        logger.warning(
            "%d prop-outcome event(s) have no matching game outcome — "
            "created HistoricalOutcome records with prop_outcomes only.",
            orphan_count,
        )
    return list(by_id.values())


def _build_bundle(args: argparse.Namespace) -> IngestBundle:
    games = args.games
    if args.source in ("nflfast", "nba_csv"):
        a = NflfastCsvAdapter(args.league) if args.source == "nflfast" else NbaCsvAdapter(args.league)
        return IngestBundle(
            events=a.read_events(games),
            outcomes=a.read_outcomes(games),
            files=[games],
            row_counts={games: a.row_count(games)},
        )
    if args.source == "football_data":
        a = SoccerFootballDataAdapter(args.league)
        return IngestBundle(
            events=a.read_events(games),
            outcomes=a.read_outcomes(games),
            odds=a.read_odds(games),
            files=[games],
            row_counts={games: a.row_count(games)},
        )
    if args.source == "tennis_atp":
        a = TennisAtpCsvAdapter(args.league)
        return IngestBundle(
            events=a.read_events(games),
            outcomes=a.read_outcomes(games),
            extra_context=a.read_extra_context(games),
            history_override=a.read_serve_history(games),
            files=[games],
            row_counts={games: a.row_count(games)},
        )

    # csv_games (+ optional generic odds)
    g = CsvGamesAdapter(args.league)
    bundle = IngestBundle(
        events=g.read_events(games),
        outcomes=g.read_outcomes(games),
        files=[games],
        row_counts={games: g.row_count(games)},
    )
    if args.odds:
        o = CsvOddsAdapter(args.league)
        bundle.odds = o.read_odds(args.odds)
        bundle.files.append(args.odds)
        bundle.row_counts[args.odds] = o.row_count(args.odds)
    return bundle


def _date_range(events: list[HistoricalEvent]) -> tuple[str | None, str | None]:
    if not events:
        return None, None
    dates = sorted(e.start_time for e in events)
    return dates[0], dates[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a local historical CSV dataset.")
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--league", required=True, help="League code, e.g. NFL, EPL, ATP")
    parser.add_argument("--games", required=True, help="Path to the games/matches CSV")
    parser.add_argument("--odds", default=None, help="Optional separate odds CSV (csv_games source)")
    parser.add_argument("--player-stats", default=None, help="Optional player-stats CSV (prop outcomes)")
    parser.add_argument(
        "--prop-markets", default=None, help="Optional prop-markets CSV (decision-time lines/prices)"
    )
    parser.add_argument(
        "--prop-context", default=None, help="Optional prop player-context JSON (as-of means)"
    )
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument(
        "--limitations", action="append", default=[], help="Documented dataset limitation (repeatable)"
    )
    parser.add_argument(
        "--odds-timing-class",
        choices=["decision_time_safe", "closing_only", "timing_unknown"],
        default=None,
        help="Override source odds timing class (default: per-source registry).",
    )
    parser.add_argument(
        "--quarantine-root", default=None, help="Quarantine root (default data/historical/quarantine)"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not Path(args.games).exists():
        logger.error("games file not found: %s", args.games)
        return 1

    bundle = _build_bundle(args)
    if not bundle.events:
        logger.error("no events parsed from %s", args.games)
        return 1

    # Fail-soft: quarantine identity-missing + duplicate-key rows rather than
    # silently replaying them. Outcomes/odds for rejected events are dropped too.
    clean_events, rejected = partition_events(bundle.events)
    if rejected:
        path = write_rejected(rejected, args.league, root=args.quarantine_root)
        rejected_ids = {r["event_id"] for r in rejected}
        bundle.events = clean_events
        bundle.outcomes = [o for o in bundle.outcomes if o.event_id not in rejected_ids]
        bundle.odds = [o for o in bundle.odds if o.event_key not in rejected_ids]
        logger.warning("Quarantined %d row(s) -> %s", len(rejected), path)
    if not bundle.events:
        logger.error("no clean events after quarantine for %s", args.games)
        return 1

    # Optional player-prop inputs (league-scoped player-stat markets).
    if args.player_stats or args.prop_markets:
        pa = CsvPlayerStatsAdapter(args.league)
        if args.player_stats:
            bundle.outcomes = _merge_prop_outcomes(
                bundle.outcomes, pa.read_prop_outcomes(args.player_stats)
            )
            bundle.files.append(args.player_stats)
            bundle.row_counts[args.player_stats] = pa.row_count(args.player_stats)
        if args.prop_markets:
            bundle.prop_markets = pa.read_prop_markets(args.prop_markets)
            bundle.files.append(args.prop_markets)
            bundle.row_counts[args.prop_markets] = pa.row_count(args.prop_markets)
    if args.prop_context:
        try:
            bundle.prop_context = json.loads(Path(args.prop_context).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in prop-context file %s: %s", args.prop_context, exc
            )
            return 1
        bundle.files.append(args.prop_context)

    family = sport_family_for(args.league)
    manifest = compute_manifest(
        bundle.files,
        source_name=args.source,
        league=args.league,
        sport_family=family,
        row_counts=bundle.row_counts,
        date_range=_date_range(bundle.events),
        limitations=args.limitations,
    )
    timing = args.odds_timing_class or timing_class_for_source(args.source).value
    manifest = manifest.model_copy(update={"odds_timing_class": timing})
    save_dataset_manifest(manifest, root=args.root)
    save_normalized_dataset(
        manifest.manifest_id,
        events=bundle.events,
        outcomes=bundle.outcomes,
        odds=bundle.odds,
        extra_context=bundle.extra_context,
        history_override=bundle.history_override,
        prop_markets=bundle.prop_markets,
        prop_context=bundle.prop_context,
        root=args.root,
    )

    logger.info(
        "Ingested %d events, %d outcomes, %d odds observations for %s (%s).",
        len(bundle.events),
        len(bundle.outcomes),
        len(bundle.odds),
        args.league,
        args.source,
    )
    logger.info("manifest_id=%s", manifest.manifest_id)
    print(manifest.manifest_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
