"""omega-historical-lab-run — orchestrate a full historical validation lab run.

Replays a dataset into an isolated DB, fits a variant grid, seals the holdout,
runs walk-forward + parity, and writes the four net-new artifacts under
``var/historical/lab_runs/<lab_run_id>/``. Auto-promotion is OFF unless
``--auto-promote`` is passed, and even then only fires through the single
fail-closed ``CalibrationRegistry.promote()`` gate on an all-green, clean tree.

Exit codes:
    0 — run completed (inspect promotion_status in LAB_RUN.json)
    1 — fatal error (bad windows, replay failure, etc.)
    2 — refused: --replay-db points at the production trace DB
"""

from __future__ import annotations

import argparse
import logging
import sys

from omega.historical.lab.orchestrator import run_lab
from omega.historical.lab.schemas import Window
from omega.paths import default_trace_db_path, is_production_trace_db

logger = logging.getLogger("omega.ops.historical_lab_run")


def _window(arg: str) -> Window:
    if ".." not in arg:
        raise argparse.ArgumentTypeError(f"window must be START..END, got {arg!r}")
    start, end = arg.split("..", 1)
    try:
        return Window(start=start.strip(), end=end.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate a historical validation lab run over the existing engine."
    )
    parser.add_argument("--league", required=True, help="League code, e.g. FIFA_INTL")
    parser.add_argument("--manifest-id", required=True, help="Ingested dataset manifest id")
    parser.add_argument(
        "--plane",
        choices=["game", "prop", "draw", "cover", "over", "under"],
        default="game",
    )
    parser.add_argument(
        "--replay-db", required=True, help="Isolated sqlite path (NEVER the production DB)"
    )
    parser.add_argument(
        "--production-db", default=None, help="Live DB for historical-live parity (read-only)"
    )
    parser.add_argument(
        "--train-window", required=True, type=_window, help="START..END (ISO dates)"
    )
    parser.add_argument("--validation-window", required=True, type=_window, help="START..END")
    parser.add_argument("--holdout-window", required=True, type=_window, help="START..END")
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Arm auto-promotion through the single fail-closed gate (default off).",
    )
    parser.add_argument("--methods", nargs="+", default=["isotonic", "shrinkage"])
    parser.add_argument("--slices", nargs="*", default=[])
    parser.add_argument("--sport-family", default=None)
    parser.add_argument(
        "--rho-profile", default=None, help="Frozen Dixon-Coles rho for soccer replay"
    )
    parser.add_argument("--lab-run-id", default=None)
    parser.add_argument(
        "--marginal-value",
        action="store_true",
        help=(
            "Compute exact per-signal marginal value via counterfactual re-simulation "
            "over the live evidence-bearing traces (requires --production-db). Default "
            "off — the re-sim cost scales with live traces × applied signals."
        ),
    )
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if is_production_trace_db(args.replay_db):
        logger.error(
            "Refusing to use the production trace DB (%s) as --replay-db. Pass an "
            "isolated path such as var/historical/lab_replay_%s.db.",
            default_trace_db_path(),
            args.league.lower(),
        )
        return 2

    try:
        result = run_lab(
            league=args.league,
            manifest_id=args.manifest_id,
            plane=args.plane,
            replay_db_path=args.replay_db,
            production_db_path=args.production_db,
            train_window=args.train_window,
            validation_window=args.validation_window,
            holdout_window=args.holdout_window,
            auto_promote=args.auto_promote,
            compute_marginal_value=args.marginal_value,
            methods=tuple(args.methods),
            slices=tuple(args.slices),
            sport_family=args.sport_family,
            rho_profile=args.rho_profile,
            lab_run_id=args.lab_run_id,
            root=args.root,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:  # pragma: no cover - surface a clean nonzero on any failure
        logger.exception("historical lab run failed")
        return 1

    run = result.lab_run
    winner = result.evidence.candidate_id or (
        result.ledger.selected.variant_id if result.ledger.selected else None
    )
    logger.info(
        "lab_run=%s promotion_status=%s winner=%s attempts=%d holdout_sealed=%s",
        run.lab_run_id,
        run.promotion_status,
        winner,
        run.attempted_variant_count,
        run.holdout_sealed,
    )
    print(run.lab_run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
