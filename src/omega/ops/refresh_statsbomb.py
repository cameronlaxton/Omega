"""
omega.ops.refresh_statsbomb — pull StatsBomb Open Data into the soccer quant plane.

Two jobs (Phase 7 M2):

1. ``--profile <id>`` — warm the local cache with every open-data match of the
   profile's competition group (e.g. ``fifa_intl_v1`` -> FIFA World Cup + UEFA
   Euro + Copa America + AFCON + Nations League) and report the fit-dataset
   size. ``omega-fit-dixon-coles`` reads the same cache, so after this runs the
   rho fit works fully offline.
2. ``--xg --competition-id N --season-id M`` — fetch that season's matches and
   per-match events, aggregate team xG for/against, resolve team names through
   ``data/aliases/SOCCER.json``, and upsert per-game ``priors_xg`` rows into the
   trace store.

Replay-safe: cached pulls are served under OMEGA_REPLAY_MODE=1; cold pulls are
blocked by the ETL guard.

Usage:
    omega-refresh-statsbomb --profile fifa_intl_v1
    omega-refresh-statsbomb --xg --competition-id 43 --season-id 106
    omega-refresh-statsbomb --xg --competition-id 43 --season-id 106 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.integrations._etl import load_alias_table, validate_records  # noqa: E402
from omega.integrations.statsbomb import (  # noqa: E402
    SBMatch,
    build_xg_priors,
    compute_team_xg_aggregates,
    fetch_competitions,
    fetch_events,
    fetch_matches,
    load_profile_matches,
)

logger = logging.getLogger("refresh_statsbomb")


def _warm_profile(args: argparse.Namespace) -> int:
    pairs = load_profile_matches(args.profile, cache_root=args.cache_root)
    if not pairs:
        logger.error("profile %s matched no open-data matches", args.profile)
        return 1
    draws = sum(1 for h, a in pairs if h == a)
    logger.info(
        "profile %s: cached %d matches (draw rate %.1f%%); fit dataset ready for "
        "omega-fit-dixon-coles",
        args.profile,
        len(pairs),
        draws / len(pairs) * 100.0,
    )
    return 0


def _refresh_xg(args: argparse.Namespace) -> int:
    raw_matches = fetch_matches(
        args.competition_id, args.season_id, cache_root=args.cache_root
    )
    matches = validate_records(raw_matches, SBMatch, source="statsbomb")
    if args.max_matches:
        matches = matches[: args.max_matches]

    competitions = fetch_competitions(cache_root=args.cache_root)
    comp_name, season_name = f"competition_{args.competition_id}", str(args.season_id)
    for comp in competitions:
        if (
            comp.get("competition_id") == args.competition_id
            and comp.get("season_id") == args.season_id
        ):
            comp_name = comp.get("competition_name", comp_name)
            season_name = comp.get("season_name", season_name)
            break

    events_by_match = {
        m.match_id: fetch_events(m.match_id, cache_root=args.cache_root) for m in matches
    }
    aggregates = compute_team_xg_aggregates(matches, events_by_match)
    priors, unresolved = build_xg_priors(
        aggregates,
        competition=comp_name,
        season=season_name,
        as_of_date=args.as_of or date.today().isoformat(),
        alias_table=load_alias_table("SOCCER"),
    )
    logger.info(
        "%s %s: %d team xG priors from %d matches (%d unresolved team(s) excluded)",
        comp_name,
        season_name,
        len(priors),
        len(matches),
        len(unresolved),
    )
    for prior in priors:
        logger.info(
            "  %-25s xg_for=%.2f xg_against=%.2f (n=%d)",
            prior.team,
            prior.xg_for,
            prior.xg_against,
            prior.matches,
        )

    if args.dry_run:
        logger.info("Dry run — not writing priors_xg.")
        return 0

    from omega.trace.priors import upsert_xg_prior
    from omega.trace.store import TraceStore

    store = TraceStore(db_path=args.db) if args.db else TraceStore()
    try:
        for prior in priors:
            upsert_xg_prior(store, prior)
    finally:
        store.close()
    logger.info("Wrote %d priors_xg rows.", len(priors))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh StatsBomb Open Data caches and soccer xG priors"
    )
    parser.add_argument(
        "--profile", help="Warm the match cache for a Dixon-Coles profile (e.g. fifa_intl_v1)"
    )
    parser.add_argument("--xg", action="store_true", help="Build priors_xg for one season")
    parser.add_argument("--competition-id", type=int, help="StatsBomb competition_id (for --xg)")
    parser.add_argument("--season-id", type=int, help="StatsBomb season_id (for --xg)")
    parser.add_argument("--as-of", default=None, help="as_of_date stamp (default: today)")
    parser.add_argument("--max-matches", type=int, default=None, help="Cap matches (testing)")
    parser.add_argument("--cache-root", default=None, help="Override ETL cache root")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.profile:
        return _warm_profile(args)
    if args.xg:
        if args.competition_id is None or args.season_id is None:
            parser.error("--xg requires --competition-id and --season-id")
        return _refresh_xg(args)
    parser.error("nothing to do: pass --profile or --xg")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    sys.exit(main())
