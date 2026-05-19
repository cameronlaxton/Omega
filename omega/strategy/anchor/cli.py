"""
CLI entry point for the anchor parlay scanner.

Usage:
    python -m omega.strategy.anchor.cli --league NBA
    python -m omega.strategy.anchor.cli --league NBA --json
    python -m omega.strategy.anchor.cli --league NBA --min-hit-rate 0.75 --max-legs 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from omega.strategy.anchor.formatter import format_scan_result
from omega.strategy.anchor.scanner import AnchorParlayConfig, ScanResult, run_full_scan


def _scan_result_to_dict(result: ScanResult) -> dict[str, Any]:
    """Convert ScanResult to a JSON-serializable dict."""
    d: dict[str, Any] = {
        "league": result.league,
        "date": result.date,
        "games_scanned": result.games_scanned,
        "players_scanned": result.players_scanned,
        "anchors_found": result.anchors_found,
        "parlays_built": result.parlays_built,
        "scan_metadata": result.scan_metadata,
        "parlays": [],
    }
    for p in result.parlays:
        parlay_dict = {
            "game": p.game,
            "legs": [
                {
                    "player": leg.player_name,
                    "team": leg.team,
                    "stat": leg.stat_key,
                    "threshold": leg.threshold,
                    "hit_rate": leg.hit_rate,
                    "games_checked": leg.games_checked,
                    "odds_over": leg.odds_over,
                    "implied_prob": leg.implied_prob,
                    "edge_pct": leg.edge_pct,
                }
                for leg in p.legs
            ],
            "combined_decimal_odds": p.combined_decimal_odds,
            "combined_hit_rate": p.combined_hit_rate,
            "implied_probability": p.implied_probability,
            "combined_edge_pct": p.combined_edge_pct,
            "ev_pct": p.ev_pct,
            "correlation_warnings": p.correlation_warnings,
            "recommended_units": p.recommended_units,
            "kelly_fraction": p.kelly_fraction,
            "confidence_tier": p.confidence_tier,
        }
        d["parlays"].append(parlay_dict)
    return d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omega-anchor-scan",
        description="Scan today's slate for anchor parlay opportunities.",
    )
    parser.add_argument(
        "--league", default="NBA",
        help="League to scan (default: NBA)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--min-hit-rate", type=float, default=0.70,
        help="Minimum empirical hit rate for anchors (default: 0.70)",
    )
    parser.add_argument(
        "--min-legs", type=int, default=2,
        help="Minimum legs per parlay (default: 2)",
    )
    parser.add_argument(
        "--max-legs", type=int, default=4,
        help="Maximum legs per parlay (default: 4)",
    )
    parser.add_argument(
        "--target-min-odds", type=float, default=1.80,
        help="Minimum combined decimal odds (default: 1.80)",
    )
    parser.add_argument(
        "--target-max-odds", type=float, default=3.00,
        help="Maximum combined decimal odds (default: 3.00)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Bankroll for stake sizing (default: 1000)",
    )
    parser.add_argument(
        "--max-results", type=int, default=20,
        help="Maximum parlays to return (default: 20)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    config = AnchorParlayConfig(
        league=args.league,
        min_hit_rate=args.min_hit_rate,
        min_legs=args.min_legs,
        max_legs=args.max_legs,
        target_min_odds=args.target_min_odds,
        target_max_odds=args.target_max_odds,
        bankroll=args.bankroll,
        max_results=args.max_results,
    )

    result = run_full_scan(league=args.league, config=config)

    if args.json_output:
        print(json.dumps(_scan_result_to_dict(result), indent=2))
    else:
        print(format_scan_result(result))

    # Exit 0 if parlays found, 1 if no data/parlays
    return 0 if result.parlays_built > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
