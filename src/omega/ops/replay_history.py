"""omega-replay-history â€” replay a historical dataset into a DEDICATED calibration DB.

Calibration-oriented sibling of ``replay_historical_slate``. It runs the same
deterministic :class:`~omega.historical.replay.ReplayEngine` but is purpose-built
for the calibration backfill loop:

* **fails closed** if ``--db`` points at the production trace DB (exit 2);
* every replayed trace is tagged ``execution_mode=historical_replay`` (done in the
  engine) so the fitter can include/exclude it explicitly;
* writes a human ``RUN_AUDIT.md`` plus a machine ``replay_summary.json`` beside the
  replay manifest under ``var/historical/replays/<replay_id>/``.

Replayed traces are synthetic: they live in their own DB and only enter the
calibration fitter when an operator explicitly passes ``--historical-db`` /
``--include-historical`` to ``omega-fit-calibration``. Production
``var/omega_traces.db`` is never touched. No network â€” local files only.
"""

from __future__ import annotations

import argparse
import logging
import sys

from omega.historical.contracts import ReplayConfig
from omega.historical.manifests import (
    load_dataset_manifest,
    load_normalized_dataset,
    save_replay_manifest,
    save_replay_summary,
    save_run_audit,
    save_selections,
)
from omega.historical.replay import ReplayDataset, ReplayEngine
from omega.historical.reports import build_replay_summary, render_run_audit
from omega.paths import default_trace_db_path, is_production_trace_db
from omega.trace.store import TraceStore, log_effective_db

logger = logging.getLogger("omega.ops.replay_history")


def _resolve_frozen_prior_payload(args: argparse.Namespace) -> dict | None:
    """Resolve a frozen game-level prior_payload from CLI args (once, up front).

    Priority: explicit --rho > --rho-profile lookup > None. The --rho-profile
    lookup reads the PRODUCTION Dixon-Coles rho (read-only) from --priors-db
    (default: production trace DB) and freezes the value + provenance, so the
    replay run is deterministic and independent of later priors-table edits.
    Raises SystemExit(2) if a named profile has no production row (fail closed).
    """
    if args.rho is not None:
        return {"rho": float(args.rho), "rho_profile_id": "explicit", "rho_as_of_date": None}
    if not args.rho_profile:
        return None

    from omega.trace.priors import get_production_dc_profile

    priors_db = args.priors_db or str(default_trace_db_path())
    store = TraceStore(db_path=priors_db)
    try:
        prof = get_production_dc_profile(store, args.rho_profile)
    finally:
        store.close()
    if prof is None:
        logger.error(
            "No production Dixon-Coles profile %r in %s; cannot inject rho. "
            "Fit/promote via omega-fit-dixon-coles first.",
            args.rho_profile,
            priors_db,
        )
        raise SystemExit(2)
    logger.info(
        "Frozen rho prior: profile=%s rho=%s as_of=%s (from %s)",
        prof.profile_id,
        prof.rho,
        prof.as_of_date,
        priors_db,
    )
    return {
        "rho": prof.rho,
        "rho_profile_id": prof.profile_id,
        "rho_as_of_date": prof.as_of_date,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a historical dataset into a dedicated calibration DB."
    )
    parser.add_argument("--league", required=True, help="League code, e.g. EPL")
    parser.add_argument("--manifest-id", required=True, help="Ingested dataset manifest id")
    parser.add_argument(
        "--db",
        required=True,
        help="Dedicated sqlite path for replayed traces (NEVER the production DB).",
    )
    parser.add_argument(
        "--mode",
        choices=["calibration"],
        default="calibration",
        help="Replay mode (calibration: staking off by default).",
    )
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument("--replay-id", default=None, help="Replay run id (default derived)")
    parser.add_argument("--session-id", default="historical-replay")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--n-iterations", type=int, default=1000)
    parser.add_argument("--simulation-backend", default="fast_score")
    parser.add_argument(
        "--rho-profile",
        default=None,
        help=(
            "Dixon-Coles profile_id (e.g. fifa_intl_v1) to inject as a FROZEN rho "
            "prior for the soccer bivariate-DC backend. Resolved ONCE from "
            "--priors-db at replay start and applied to every request. Required "
            "for soccer_bivariate_poisson_dc replay (else the backend fails closed)."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Explicit frozen rho value (overrides --rho-profile lookup).",
    )
    parser.add_argument(
        "--priors-db",
        default=None,
        help=(
            "DB to read the production Dixon-Coles rho from for --rho-profile "
            "(default: the production trace DB). Read-only; never written."
        ),
    )
    parser.add_argument(
        "--enable-staking",
        action="store_true",
        help="Size historical bets (off by default in calibration mode).",
    )
    parser.add_argument("--leakage-policy", choices=["skip", "fail"], default="skip")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Fail closed before doing any work: replayed (synthetic) traces must never
    # pollute the live calibration DB.
    if is_production_trace_db(args.db):
        logger.error(
            "Refusing to replay into the production trace DB (%s). Pass a dedicated "
            "--db such as var/historical/replay_%s.db.",
            default_trace_db_path(),
            args.league.lower(),
        )
        return 2

    try:
        manifest = load_dataset_manifest(args.manifest_id, root=args.root)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if manifest.league.upper() != args.league.upper():
        logger.warning(
            "league mismatch: --league=%s but manifest league=%s; using manifest league.",
            args.league.upper(),
            manifest.league,
        )

    ds_parts = load_normalized_dataset(args.manifest_id, root=args.root)
    dataset = ReplayDataset(
        events=ds_parts["events"],
        outcomes=ds_parts["outcomes"],
        odds=ds_parts["odds"],
        extra_context=ds_parts["extra_context"],
        history_override=ds_parts["history_override"],
        prop_markets=ds_parts.get("prop_markets", {}),
        prop_context=ds_parts.get("prop_context", {}),
    )

    # Resolve a FROZEN game-level prior_payload ONCE (replay must not depend on
    # live priors-table mutation mid-run). Used by the soccer bivariate-DC backend.
    prior_payload = _resolve_frozen_prior_payload(args)

    config = ReplayConfig(
        dataset_manifest_id=manifest.manifest_id,
        backtest_db_path=args.db,
        session_id=args.session_id,
        bankroll=args.bankroll,
        n_iterations=args.n_iterations,
        simulation_backend=args.simulation_backend,
        enable_staking=args.enable_staking,
        leakage_policy=args.leakage_policy,
        prior_payload=prior_payload,
        odds_timing_class=manifest.odds_timing_class or "decision_time_safe",
    )
    replay_id = args.replay_id or f"replay_{manifest.manifest_id}"

    store = TraceStore(db_path=args.db)
    try:
        log_effective_db(store, logger)
        engine = ReplayEngine(store, config)
        result = engine.run(dataset, replay_id=replay_id, league=manifest.league)
        # Count of calibration-eligible graded replay traces now in this dedicated
        # DB (the calibration pool size). DB-wide so appended seasons accumulate.
        eligible_count = len(
            store.query_traces(
                execution_mode="historical_replay",
                has_outcome=True,
                calibration_eligible_only=True,
                limit=1_000_000,
            )
        )
        eligible_denominator = len(
            store.query_traces(
                execution_mode="historical_replay",
                has_outcome=True,
                limit=1_000_000,
            )
        )
    finally:
        store.close()

    save_replay_manifest(result.manifest, root=args.root)
    save_selections(replay_id, result.selections, root=args.root)
    summary = build_replay_summary(
        result.manifest,
        eligible_count=eligible_count,
        eligible_denominator=eligible_denominator,
        league=manifest.league,
    )
    save_replay_summary(replay_id, summary, root=args.root)
    save_run_audit(replay_id, render_run_audit(summary), root=args.root)

    logger.info(
        "Replay %s: persisted=%d skipped=%d eligible=%d selections=%d",
        replay_id,
        result.n_persisted,
        result.n_skipped,
        eligible_count,
        len(result.selections),
    )
    print(replay_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
