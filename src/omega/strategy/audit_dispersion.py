"""
omega.strategy.audit_dispersion -- Read-only fit/report command to compare distributions.

Compares Poisson, Normal, Negative Binomial, and current fast_score
by league/market using frozen graded traces.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.simulation.backends import PropSimulationInput, resolve_prop_backend  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("audit_dispersion")


def brier_score(y_true: float, y_pred: float) -> float:
    return (y_true - y_pred) ** 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit prop dispersion across distributions.")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument("--league", default="NFL", help="League to audit (default: NFL)")
    parser.add_argument("--stat", default="rushing_yards", help="Stat type to audit (default: rushing_yards)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    store = TraceStore(db_path=args.db)

    traces = store.query_traces(
        league=args.league,
        has_outcome=True,
        limit=5000,
    )

    # Filter for prop traces of the requested stat type
    props = [
        t for t in traces
        if t.get("kind") == "prop" and t.get("request", {}).get("prop_type") == args.stat
    ]

    if not props:
        logger.warning("No graded prop traces found for %s %s", args.league, args.stat)
        return 0

    logger.info("Found %d graded prop traces for %s %s", len(props), args.league, args.stat)

    # We will simulate each trace using Poisson, Normal, and Negative Binomial
    # to compare their Brier scores for the Over prediction.

    dist_router = resolve_prop_backend("prop_distribution_router")
    nb_router = resolve_prop_backend("prop_neg_binom")

    results = {
        "poisson": [],
        "normal": [],
        "negative_binomial": [],
    }

    for trace in props:
        req = trace.get("request", {})
        mean = req.get("projection_mean", req.get("line", 0))
        if mean <= 0:
            continue

        std_value = req.get("projection_std")
        std = std_value if std_value is not None else max(1.0, mean * 0.3)
        variance = std ** 2
        line = req.get("line", mean)

        # Outcome is typically in trace["outcome"]["away_score"] or "home_score" but
        # for props it might be trace["outcome"]["actual_value"] if it exists.
        # Fallback to checking the trace model.
        outcome = trace.get("outcome", {})
        if "actual_value" in outcome:
            actual = outcome["actual_value"]
        elif "actual" in outcome:
            actual = outcome["actual"]
        else:
            # Can't grade without actual
            continue

        is_over = 1.0 if actual > line else 0.0

        sim_input_base = PropSimulationInput(
            player_name=req.get("player_name", "Unknown"),
            league=args.league,
            stat_type=args.stat,
            line=line,
            projection_mean=mean,
            projection_std=std,
            n_iter=2000,
            seed=42,
            prior_payload={},
        )

        # 1. Poisson
        sim_input_poisson = PropSimulationInput(
            **{**sim_input_base.__dict__, "prior_payload": {"distribution": "poisson"}}
        )
        if dist_router:
            res_poisson = dist_router.run(sim_input_poisson)
            results["poisson"].append(brier_score(is_over, res_poisson["over_prob"]))

        # 2. Normal
        sim_input_normal = PropSimulationInput(
            **{**sim_input_base.__dict__, "prior_payload": {"distribution": "normal"}}
        )
        if dist_router:
            res_normal = dist_router.run(sim_input_normal)
            results["normal"].append(brier_score(is_over, res_normal["over_prob"]))

        # 3. Negative Binomial
        # estimate k using method of moments: variance = mean + mean^2 / k => k = mean^2 / (variance - mean)
        if variance > mean:
            k = (mean**2) / (variance - mean)
        else:
            k = 100.0  # large k approaches Poisson

        sim_input_nb = PropSimulationInput(
            **{**sim_input_base.__dict__, "prior_payload": {"nb_dispersion_k": k}}
        )
        if nb_router:
            res_nb = nb_router.run(sim_input_nb)
            results["negative_binomial"].append(brier_score(is_over, res_nb["over_prob"]))

    store.close()

    print(f"=== Dispersion Audit for {args.league} {args.stat} ===")
    print(f"Sample size: {len(results['normal'])}")
    for dist, scores in results.items():
        if scores:
            avg_brier = sum(scores) / len(scores)
            print(f"{dist.rjust(18)} : Brier = {avg_brier:.4f}")
        else:
            print(f"{dist.rjust(18)} : N/A")

    return 0


if __name__ == "__main__":
    sys.exit(main())
