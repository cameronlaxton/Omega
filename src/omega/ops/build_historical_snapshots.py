"""omega-build-historical-snapshots — pre-build as-of snapshots + leakage report.

Builds the feature and odds snapshots for every event in a normalized dataset
(without replaying), runs the leakage guard, and writes the snapshots plus a
leakage/health report. Useful to confirm a dataset is as-of safe before replay.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from omega.historical.leakage import evaluate_leakage
from omega.historical.manifests import datasets_dir, load_normalized_dataset
from omega.historical.odds_snapshots import build_odds_snapshot
from omega.historical.replay import build_team_histories
from omega.historical.snapshots import MatchupHistory, build_feature_snapshot

logger = logging.getLogger("omega.ops.build_historical_snapshots")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build as-of snapshots + leakage report.")
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ds = load_normalized_dataset(args.manifest_id, root=args.root)
    events = sorted(ds["events"], key=lambda e: e.start_time)
    outcomes = ds["outcomes"]
    odds = ds["odds"]
    extra_context = ds["extra_context"]
    team_hist = ds["history_override"] or build_team_histories(events, outcomes)

    leakage_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    context_counts: Counter = Counter()
    stale = 0
    missing_odds = 0
    snapshots_out = []

    for ev in events:
        decision_time = ev.start_time
        history = MatchupHistory(
            home_rows=team_hist.get(ev.home_team, []),
            away_rows=team_hist.get(ev.away_team, []),
        )
        snap = build_feature_snapshot(
            ev, history, decision_time, extra_game_context=extra_context.get(ev.event_id, {})
        )
        odds_snap = build_odds_snapshot(
            ev.event_id, odds.get(ev.event_id, []), decision_time, event_start=ev.start_time
        )
        leak = evaluate_leakage(ev, snap, odds_snap)

        leakage_counts[leak.status] += 1
        for r in leak.reasons:
            reason_counts[r] += 1
        context_counts[snap.context_source] += 1
        if snap.is_stale:
            stale += 1
        if odds_snap.missing_odds:
            missing_odds += 1

        snapshots_out.append(
            {
                "event_id": ev.event_id,
                "feature_snapshot": snap.model_dump(mode="json"),
                "odds_snapshot": odds_snap.model_dump(mode="json"),
                "leakage_status": leak.status,
                "leakage_reasons": leak.reasons,
            }
        )

    out_dir = datasets_dir(args.root) / args.manifest_id / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_snapshots.json").write_text(
        json.dumps(snapshots_out, indent=2), encoding="utf-8"
    )
    report = {
        "manifest_id": args.manifest_id,
        "n_events": len(events),
        "leakage_counts": dict(leakage_counts),
        "leakage_reasons": dict(reason_counts),
        "context_source_counts": dict(context_counts),
        "stale_context": stale,
        "missing_odds": missing_odds,
    }
    (out_dir / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "Built %d snapshots: clean=%d skipped=%d failed=%d | missing_odds=%d stale=%d",
        len(events),
        leakage_counts.get("clean", 0),
        leakage_counts.get("skipped", 0),
        leakage_counts.get("failed", 0),
        missing_odds,
        stale,
    )
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
