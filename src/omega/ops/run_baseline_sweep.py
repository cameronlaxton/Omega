"""Baseline sweep: modifier-free CRPS comparison between fast_score and markov_state.

Runs both simulation backends on historical settled games with no transition_modifiers.
Outputs a markdown table showing mean CRPS per backend and target.

Usage:
    omega-run-baseline-sweep [--league NBA] [--json data/sweep_out.json]

Data sourcing (no network calls):
  1. TraceStore.query_traces(has_outcome=True) -- uses real graded data if present
  2. Falls back to src/omega/ops/fixtures/baseline_games.json if zero graded traces found

The script does NOT persist rows to TraceStore. Sweep runs are read-only audit
operations so they do not pollute the calibration ledger with synthetic re-runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.simulation.engine import (  # noqa: E402
    MarkovGameSimulationBackend,
    OmegaSimulationEngine,
)
from omega.strategy.distribution_metrics import crps_from_distribution_row  # noqa: E402

_FIXTURES_PATH = Path(__file__).parent / "fixtures" / "baseline_games.json"
_FAST_ENGINE = OmegaSimulationEngine()
_MARKOV_ENGINE = OmegaSimulationEngine(game_backend=MarkovGameSimulationBackend())


def _load_fixtures(league_filter: str | None) -> list[dict]:
    """Load games from the bundled fixtures file."""
    with _FIXTURES_PATH.open() as f:
        games = json.load(f)
    if league_filter:
        games = [g for g in games if g.get("league", "").upper() == league_filter.upper()]
    return games


def _load_graded_traces(league_filter: str | None) -> list[dict]:
    """Attempt to load graded traces from TraceStore."""
    try:
        from omega.trace.store import TraceStore

        store = TraceStore()
        traces = store.query_traces(has_outcome=True, limit=200)
        if league_filter:
            traces = [t for t in traces if t.get("league", "").upper() == league_filter.upper()]
        return traces
    except Exception as exc:
        print(f"[warn] TraceStore unavailable ({exc}); using bundled fixtures")
        return []


def _trace_to_game(trace: dict) -> dict | None:
    """Convert a graded trace to the fixture dict shape. Returns None if unusable."""
    matchup = trace.get("matchup", "")
    if " @ " not in matchup:
        return None
    away, home = matchup.split(" @ ", 1)
    exec_result = trace.get("execution_result") or trace.get("input_snapshot") or {}
    home_ctx = exec_result.get("home_context") or {}
    away_ctx = exec_result.get("away_context") or {}
    outcome = trace.get("outcome")
    if not outcome or not home_ctx or not away_ctx:
        return None
    return {
        "home_team": home.strip(),
        "away_team": away.strip(),
        "league": trace.get("league", "NBA"),
        "home_context": home_ctx,
        "away_context": away_ctx,
        "outcome": outcome,
        "evidence_signals": [],
    }


def _run_backend(engine: OmegaSimulationEngine, game: dict, n_iterations: int = 500) -> dict | None:
    """Run one backend on one game. Returns raw sim result or None on skip/error."""
    result = engine.run_fast_game_simulation(
        home_team=game["home_team"],
        away_team=game["away_team"],
        league=game["league"],
        n_iterations=n_iterations,
        home_context=game.get("home_context"),
        away_context=game.get("away_context"),
        seed=42,
    )
    if not result.get("success"):
        return None
    return result


def _crps_for_result(sim_result: dict, outcome: dict) -> list[dict]:
    """Compute CRPS for each distribution row given the observed outcome."""
    home_score = float(outcome.get("home_score", 0))
    away_score = float(outcome.get("away_score", 0))
    total = home_score + away_score
    spread = home_score - away_score

    obs_map = {
        "home_score": home_score,
        "away_score": away_score,
        "total": total,
        "spread": spread,
    }

    rows = sim_result.get("simulation_distributions") or []
    results = []
    for row in rows:
        target = row.get("target", "")
        observed = obs_map.get(target)
        if observed is None:
            continue
        try:
            metric = crps_from_distribution_row(row, observed)
            results.append(
                {
                    "target": target,
                    "crps": metric["value"],
                    "distribution_type": row.get("distribution_type"),
                }
            )
        except Exception:
            pass
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _print_diagnostics(games: list[dict]) -> None:
    """Print per-game Markov diagnostic stats for calibration inspection."""
    from omega.core.simulation.markov_engine import MarkovSimulator

    print("\n[diagnose] Per-game Markov internals (500-game sample each):\n")
    header = f"{'matchup':<35} {'poss':>4} {'E[home]':>7} {'E[away]':>7} {'E[tot]':>7} {'std[tot]':>8} {'ppp_h':>6} {'ppp_a':>6}"
    print(header)
    print("-" * len(header))
    for game in games:
        try:
            sim = MarkovSimulator(
                league=game["league"],
                home_context=game.get("home_context"),
                away_context=game.get("away_context"),
            )
            d = sim.run_diagnostics(n_games=500)
            label = f"{game['away_team'][:12]} @ {game['home_team'][:12]}"
            real_tot = ""
            outcome = game.get("outcome")
            if outcome:
                real_tot = f"  (real {int(outcome['home_score']) + int(outcome['away_score'])})"
            print(
                f"{label:<35} {d['base_possessions']:>4} {d['mean_home_score']:>7.1f} "
                f"{d['mean_away_score']:>7.1f} {d['mean_total']:>7.1f}{real_tot:12} "
                f"{d['std_total']:>8.1f} {d['expected_ppp_home']:>6.4f} {d['expected_ppp_away']:>6.4f}"
            )
        except Exception as exc:
            print(f"  {game.get('home_team', '?')} â€” error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Modifier-free CRPS baseline sweep")
    parser.add_argument("--league", default=None, help="Filter to one league (e.g. NBA)")
    parser.add_argument(
        "--json", default=None, metavar="PATH", help="Write JSON summary to this path"
    )
    parser.add_argument(
        "--n-iterations", type=int, default=500, help="Simulation iterations per game"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print per-game Markov diagnostics (possession count, PPP, expected scores)",
    )
    args = parser.parse_args()

    # Source games: prefer graded TraceStore records; fall back to bundled fixtures
    games: list[dict] = []
    trace_games = _load_graded_traces(args.league)
    for t in trace_games:
        g = _trace_to_game(t)
        if g:
            games.append(g)

    if games:
        print(f"[info] Using {len(games)} games from TraceStore")
    else:
        games = _load_fixtures(args.league)
        print(f"[info] Using {len(games)} games from bundled fixtures")

    if not games:
        print("[error] No games available for sweep. Check fixtures or TraceStore.")
        sys.exit(1)

    if args.diagnose:
        _print_diagnostics(games)
        print()

    # Accumulate CRPS per (backend, target)
    fast_crps: dict[str, list[float]] = {}
    markov_crps: dict[str, list[float]] = {}
    skipped = 0

    for i, game in enumerate(games):
        outcome = game.get("outcome")
        if not outcome:
            skipped += 1
            continue

        fast_result = _run_backend(_FAST_ENGINE, game, args.n_iterations)
        markov_result = _run_backend(_MARKOV_ENGINE, game, args.n_iterations)

        if fast_result is None or markov_result is None:
            skipped += 1
            continue

        for entry in _crps_for_result(fast_result, outcome):
            fast_crps.setdefault(entry["target"], []).append(entry["crps"])

        for entry in _crps_for_result(markov_result, outcome):
            markov_crps.setdefault(entry["target"], []).append(entry["crps"])

        if (i + 1) % 5 == 0:
            print(f"  processed {i + 1}/{len(games)} games...")

    print(
        f"\n[info] Sweep complete. Processed {len(games) - skipped}/{len(games)} games ({skipped} skipped)\n"
    )

    # Build summary table
    all_targets = sorted(set(fast_crps) | set(markov_crps))
    rows = []
    for target in all_targets:
        fast_vals = fast_crps.get(target, [])
        markov_vals = markov_crps.get(target, [])
        fast_mean = _mean(fast_vals)
        markov_mean = _mean(markov_vals)
        delta = (
            markov_mean - fast_mean
            if not (math.isnan(fast_mean) or math.isnan(markov_mean))
            else float("nan")
        )
        rows.append(
            {
                "target": target,
                "fast_n": len(fast_vals),
                "fast_mean_crps": round(fast_mean, 4) if not math.isnan(fast_mean) else None,
                "markov_n": len(markov_vals),
                "markov_mean_crps": round(markov_mean, 4) if not math.isnan(markov_mean) else None,
                "delta_markov_vs_fast": round(delta, 4) if not math.isnan(delta) else None,
            }
        )

    # Print markdown table
    header = f"{'target':<12} {'fast_n':>6} {'fast_crps':>10} {'markov_n':>8} {'markov_crps':>12} {'delta':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        delta_str = (
            f"{r['delta_markov_vs_fast']:+.4f}"
            if r["delta_markov_vs_fast"] is not None
            else "   n/a"
        )
        note = (
            " <-- better"
            if (r["delta_markov_vs_fast"] is not None and r["delta_markov_vs_fast"] < 0)
            else ""
        )
        fast_str = f"{r['fast_mean_crps']:.4f}" if r["fast_mean_crps"] is not None else "   n/a"
        markov_str = (
            f"{r['markov_mean_crps']:.4f}" if r["markov_mean_crps"] is not None else "   n/a"
        )
        print(
            f"{r['target']:<12} {r['fast_n']:>6} {fast_str:>10} "
            f"{r['markov_n']:>8} {markov_str:>12} {delta_str:>8}{note}"
        )

    print("\nDelta interpretation: negative = markov better, positive = fast_score better")

    if args.json:
        out = {"games_processed": len(games) - skipped, "games_skipped": skipped, "results": rows}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n[info] JSON summary written to {args.json}")


if __name__ == "__main__":
    main()
