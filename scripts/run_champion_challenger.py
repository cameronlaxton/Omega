"""Champion/Challenger: compare raw Markov vs. evidence-adjusted Markov CRPS.

Champion:   markov_state backend, no transition_modifiers
Challenger: markov_state backend + EvidenceSignals mapped via evidence_to_modifier

Games with empty evidence_signals arrays contribute only to Champion metrics.
Games with populated evidence_signals arrays contribute to both.

Usage:
    python scripts/run_champion_challenger.py [--league NBA] [--json data/cc_out.json]

Data source: scripts/fixtures/baseline_games.json (or graded TraceStore data if available).

Phase 3b limitation: fast_score is excluded from this comparison because the
fast backend ignores transition_modifiers. A full cross-backend comparison
(fast_score vs. Markov on identical evidence-adjusted parameters) requires a
separate fast_score evidence handler mapping, which is Phase 3c scope.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.core.contracts.evidence import EvidenceSignal  # noqa: E402
from omega.core.simulation.engine import (  # noqa: E402
    MarkovGameSimulationBackend,
    OmegaSimulationEngine,
)
from omega.core.simulation.evidence_to_modifier import signals_to_transition_modifiers  # noqa: E402
from omega.strategy.distribution_metrics import crps_from_distribution_row  # noqa: E402

_FIXTURES_PATH = Path(__file__).parent / "fixtures" / "baseline_games.json"
_MARKOV_ENGINE = OmegaSimulationEngine(game_backend=MarkovGameSimulationBackend())


def _load_fixtures(league_filter: str | None) -> list[dict]:
    with _FIXTURES_PATH.open() as f:
        games = json.load(f)
    if league_filter:
        games = [g for g in games if g.get("league", "").upper() == league_filter.upper()]
    return games


def _trace_league(trace: dict) -> str:
    return str(trace.get("league") or "").upper()


def _load_graded_traces(league_filter: str | None) -> list[dict]:
    try:
        from omega.trace.store import TraceStore

        store = TraceStore()
        traces = store.query_traces(has_outcome=True, limit=200)
        if league_filter:
            league_upper = league_filter.upper()
            traces = [t for t in traces if _trace_league(t) == league_upper]
        return traces
    except Exception as exc:
        print(f"[warn] TraceStore unavailable ({exc}); using bundled fixtures")
        return []


def _trace_to_game(trace: dict) -> dict | None:
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


def _parse_signals(raw_signals: list[dict]) -> list[EvidenceSignal]:
    """Parse raw signal dicts from fixtures into typed EvidenceSignal objects."""
    signals: list[EvidenceSignal] = []
    for s in raw_signals:
        try:
            signals.append(EvidenceSignal(**s))
        except Exception as exc:
            print(f"[warn] Could not parse evidence signal {s.get('signal_type')!r}: {exc}")
    return signals


def _run_markov(
    game: dict,
    transition_modifiers: dict | None,
    n_iterations: int,
) -> dict | None:
    result = _MARKOV_ENGINE.run_fast_game_simulation(
        home_team=game["home_team"],
        away_team=game["away_team"],
        league=game["league"],
        n_iterations=n_iterations,
        home_context=game.get("home_context"),
        away_context=game.get("away_context"),
        seed=42,
        transition_modifiers=transition_modifiers,
    )
    return result if result.get("success") else None


def _crps_for_result(sim_result: dict, outcome: dict) -> dict[str, float]:
    """Returns {target: crps_value} for each distribution row with a known outcome."""
    home_score = float(outcome.get("home_score", 0))
    away_score = float(outcome.get("away_score", 0))
    obs_map = {
        "home_score": home_score,
        "away_score": away_score,
        "total": home_score + away_score,
        "spread": home_score - away_score,
    }
    out: dict[str, float] = {}
    for row in sim_result.get("simulation_distributions") or []:
        target = row.get("target", "")
        observed = obs_map.get(target)
        if observed is None:
            continue
        try:
            out[target] = crps_from_distribution_row(row, observed)["value"]
        except Exception:
            pass
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Markov Champion/Challenger CRPS comparison")
    parser.add_argument("--league", default=None, help="Filter to one league (e.g. NBA)")
    parser.add_argument("--json", default=None, metavar="PATH", help="Write JSON summary to this path")
    parser.add_argument("--n-iterations", type=int, default=500, help="Simulation iterations per game")
    args = parser.parse_args()

    # Source games: prefer graded TraceStore records; fall back to bundled fixtures
    games: list[dict] = []
    trace_games = _load_graded_traces(args.league)
    trace_skipped = 0
    for t in trace_games:
        g = _trace_to_game(t)
        if g:
            games.append(g)
        else:
            trace_skipped += 1

    if games:
        print(
            f"[info] Using {len(games)} games from TraceStore "
            f"({len(trace_games)} loaded, {trace_skipped} skipped during conversion)"
        )
    else:
        if trace_games:
            print(
                "[warn] HARD_FALLBACK: TraceStore rows were found but none converted "
                "to champion/challenger games; using bundled fixtures."
            )
        games = _load_fixtures(args.league)
        print(f"[info] Using {len(games)} games from bundled fixtures")

    if not games:
        print("[error] No games available. Check fixtures or TraceStore.")
        sys.exit(1)

    # Accumulate CRPS per (role, target)
    champ_crps: dict[str, list[float]] = {}
    challenger_crps: dict[str, list[float]] = {}
    challenger_games = 0
    skipped = 0

    for i, game in enumerate(games):
        outcome = game.get("outcome")
        if not outcome:
            skipped += 1
            continue

        # Champion: no modifiers
        champ_result = _run_markov(game, transition_modifiers=None, n_iterations=args.n_iterations)
        if champ_result is None:
            skipped += 1
            continue

        for target, crps_val in _crps_for_result(champ_result, outcome).items():
            champ_crps.setdefault(target, []).append(crps_val)

        # Challenger: with evidence signals (only for games that have them)
        raw_signals = game.get("evidence_signals") or []
        if raw_signals:
            signals = _parse_signals(raw_signals)
            modifiers = signals_to_transition_modifiers(signals, home_team=game["home_team"])
            if modifiers:
                challenger_result = _run_markov(game, transition_modifiers=modifiers, n_iterations=args.n_iterations)
                if challenger_result is not None:
                    for target, crps_val in _crps_for_result(challenger_result, outcome).items():
                        challenger_crps.setdefault(target, []).append(crps_val)
                    challenger_games += 1

        if (i + 1) % 5 == 0:
            print(f"  processed {i + 1}/{len(games)} games...")

    total_processed = len(games) - skipped
    print(
        f"\n[info] Sweep complete. Champion games: {total_processed}, "
        f"Challenger games: {challenger_games} ({skipped} skipped)\n"
    )

    if challenger_games == 0:
        print("[warn] No challenger games with evidence signals. Add evidence_signals to fixtures.")

    # Build summary table
    all_targets = sorted(set(champ_crps) | set(challenger_crps))
    rows = []
    for target in all_targets:
        champ_vals = champ_crps.get(target, [])
        chall_vals = challenger_crps.get(target, [])
        champ_mean = _mean(champ_vals)
        chall_mean = _mean(chall_vals)
        delta = chall_mean - champ_mean if (chall_vals and not math.isnan(champ_mean)) else float("nan")
        rows.append(
            {
                "target": target,
                "champion_n": len(champ_vals),
                "champion_mean_crps": round(champ_mean, 4) if not math.isnan(champ_mean) else None,
                "challenger_n": len(chall_vals),
                "challenger_mean_crps": round(chall_mean, 4) if not math.isnan(chall_mean) else None,
                "delta_vs_champion": round(delta, 4) if not math.isnan(delta) else None,
                "promotes": (not math.isnan(delta) and delta < 0),
            }
        )

    # Print markdown table
    header = (
        f"{'target':<12} {'champ_n':>7} {'champ_crps':>10} "
        f"{'chall_n':>7} {'chall_crps':>10} {'delta':>8} {'verdict':>10}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        delta_str = f"{r['delta_vs_champion']:+.4f}" if r["delta_vs_champion"] is not None else "    n/a"
        verdict = "PROMOTES" if r["promotes"] else ("REJECTS" if r["challenger_n"] > 0 else "NO_DATA")
        chall_crps_str = f"{r['challenger_mean_crps']}" if r["challenger_mean_crps"] is not None else "   n/a"
        print(
            f"{r['target']:<12} {r['champion_n']:>7} {r['champion_mean_crps']:>10} "
            f"{r['challenger_n']:>7} {chall_crps_str:>10} {delta_str:>8} {verdict:>10}"
        )

    print("\nDelta < 0 --> Challenger is better --> eligible for evidence policy promotion")
    print("Phase 3b limitation: fast_score excluded (fast backend ignores transition_modifiers)")

    if args.json:
        out = {
            "champion_games": total_processed,
            "challenger_games": challenger_games,
            "skipped": skipped,
            "results": rows,
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n[info] JSON summary written to {args.json}")


if __name__ == "__main__":
    main()
