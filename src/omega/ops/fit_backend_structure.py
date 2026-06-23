"""omega-fit-backend-structure — tune a backend's structural sharpness knob to
minimize RAW out-of-sample ECE, and register the winner as a CANDIDATE
``BackendParameterProfile`` (Structural Calibration Loop, Part 2).

This is the producer the upstream loop was missing. The evaluation/selection harness
(``sweep_backend_variants``), the fail-closed promotion gate
(``omega.core.governance.promotion_gates`` via ``promote_parameter_profile``), the
raw OOS-ECE estimator (``cv_calibration_diagnostic.raw_oos``), and the candidate
store (``register_parameter_profile``) all already exist and are individually
tested — this CLI is the thin orchestration that connects them, adding **no** new
simulation, metric, or promotion logic:

1. load graded historical-replay traces for a league as :class:`FrozenArtifact`s;
2. build a grid of :class:`BackendParameterProfile` candidates over one mean-
   preserving sharpness knob (``lambda_gap_scale`` / ``margin_sd_scale`` /
   ``nb_k_scale``), always including the identity baseline;
3. run :func:`sweep_backend_variants` (exact, sealed holdout, raw-ECE selection,
   noise guard) to pick the winner on a no-leak validation split;
4. compute a no-leak **raw cross-validated ECE** for the winner via the shared
   ``raw_oos`` estimator (the metric the promotion ``ECE_FLOOR`` reads); and
5. ``--register`` the winner as a CANDIDATE so ``omega-promote-parameter-profile``
   can gate it to PRODUCTION once raw CV-ECE ≤ 0.05 (+ parity / CLV evidence).

Default is ``--dry-run`` (print the grid + sealed metrics, write nothing). The knob
defaults to identity, so a backend with no promoted profile is bit-identical to the
pre-knob engine; this tool only ever produces CANDIDATEs.

Usage:
    omega-fit-backend-structure --backend soccer_bivariate_poisson_dc --league EPL \\
        --historical-db var/historical/replay_epl.db \\
        --validation-start 2024-08-01 --holdout-start 2025-05-01 \\
        --base-params '{"rho": -0.12}'
    omega-fit-backend-structure --backend fast_score --league NBA \\
        --knob margin_sd_scale --grid 0.9,1.0,1.1,1.2,1.3 \\
        --db var/omega_traces.db --validation-start 2025-01-01 --holdout-start 2025-04-01 \\
        --register

Exit codes:
    0 — completed (dry-run or registered); 1 — fatal error / no winner.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.core.calibration.fitter import (  # noqa: E402
    _adaptive_calibration_error,
    stratified_folds,
)
from omega.core.calibration.league_buckets import resolve_calibration_bucket  # noqa: E402
from omega.core.simulation.backends import resolve_game_backend  # noqa: E402
from omega.core.simulation.engine import OmegaSimulationEngine  # noqa: E402
from omega.core.simulation.parameter_profile import (  # noqa: E402
    BackendParameterProfile,
    make_parameter_profile_id,
)
from omega.historical.contracts import stable_hash  # noqa: E402
from omega.strategy.artifacts import FrozenArtifact, trace_to_artifact  # noqa: E402
from omega.strategy.backtest.variant_sweep import (  # noqa: E402
    _artifact_records,
    _simulate_pairs,
    sweep_backend_variants,
)
from omega.trace.parameter_profiles import (  # noqa: E402
    get_production_parameter_profile,
    register_parameter_profile,
)
from omega.trace.store import TraceStore, log_effective_db  # noqa: E402

logger = logging.getLogger("fit_backend_structure")

# The mean-preserving knob each backend tunes by default. fast_score serves both
# Normal (NBA -> margin_sd_scale) and Poisson (MLB/NHL -> lambda_gap_scale)
# archetypes, so its knob is intentionally absent here and must be passed via
# --knob (the league's archetype decides which one is the calibration lever).
_BACKEND_DEFAULT_KNOB: dict[str, str] = {
    "soccer_bivariate_poisson_dc": "lambda_gap_scale",
    "nfl_neg_binom": "nb_k_scale",
}

# Natural sweep range per knob. The three sharpness knobs are mean-preserving and
# centered on the 1.0 identity (compress<1 vs widen>1). home_advantage is a *bias*
# corrector (goals added to the home rate, removed from the away rate) reachable on
# the soccer DC backend; it is the right lever when the diagnostic shows a
# systematic mean bias (mean_pred != base_rate) rather than a dispersion problem.
_KNOB_DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "lambda_gap_scale": (0.6, 0.7, 0.8, 0.9, 1.0, 1.1),
    "margin_sd_scale": (0.85, 0.9, 1.0, 1.1, 1.2, 1.3),
    "nb_k_scale": (0.5, 0.75, 1.0, 1.5, 2.0),
    "home_advantage": (0.0, 0.15, 0.3, 0.45, 0.6),
}

# The identity (no-op) value per knob — the multiplicative sharpness knobs are 1.0,
# the additive home_advantage bias knob is 0.0. The CLI always keeps the identity in
# the grid so the sweep scores the current production behavior as a baseline.
_KNOB_IDENTITY: dict[str, float] = {
    "lambda_gap_scale": 1.0,
    "margin_sd_scale": 1.0,
    "nb_k_scale": 1.0,
    "home_advantage": 0.0,
}

_VALID_KNOBS = frozenset(_KNOB_DEFAULT_GRID)
_MIN_SAMPLES_FOR_CV = 60  # raw_oos needs enough per fold; mirrors cv-diagnostic floor


def build_candidates(
    *,
    backend_name: str,
    backend_component_version: str,
    competition_bucket: str,
    knob: str,
    grid: tuple[float, ...],
    base_params: dict,
    dataset_hash: str,
    priors_as_of_date: str | None = None,
    dataset_manifest_id: str | None = None,
) -> list[BackendParameterProfile]:
    """One CANDIDATE per knob value (base_params + {knob: value}); identity included.

    Every candidate shares the same backend + bucket (a sweep is within one of
    each). ``base_params`` carries the backend's REQUIRED priors (e.g. soccer's
    ``rho``) so the backend can actually run on each candidate's payload.
    """
    candidates: list[BackendParameterProfile] = []
    for i, value in enumerate(grid, start=1):
        params = {**base_params, knob: round(float(value), 6)}
        candidates.append(
            BackendParameterProfile(
                profile_id=make_parameter_profile_id(backend_name, competition_bucket, i, params),
                version=i,
                backend_name=backend_name,
                backend_component_version=backend_component_version,
                competition_bucket=competition_bucket,
                params=params,
                priors_as_of_date=priors_as_of_date,
                dataset_manifest_id=dataset_manifest_id,
                dataset_hash=dataset_hash,
                sample_size=0,  # filled with the sealed-holdout n on the winner
            )
        )
    return candidates


def _raw_cv_ece(
    predictions: list[float],
    outcomes: list[int],
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> dict:
    """No-leak raw cross-validated ECE over fixed (already-selected) raw probs.

    Reuses the SAME primitives the promotion floor checks against
    (``stratified_folds`` + ``_adaptive_calibration_error``). There is no parameter
    fitting here — the winner's params are frozen — so this is a parameter-free
    generalization-ECE estimate (with a CI), exactly what the gate's
    ``cv_calibration_error`` should carry for a structural profile. Returns the gate
    keys; ``cv_n_folds == 0`` signals "not enough data, fall back to single split".
    """
    n = len(predictions)
    if n < _MIN_SAMPLES_FOR_CV or len(set(outcomes)) < 2:
        return {"cv_calibration_error": 0.0, "cv_n_folds": 0}
    eces: list[float] = []
    for r in range(repeats):
        assignment = stratified_folds(outcomes, folds, seed + r)
        for f in range(folds):
            te = assignment[f]
            te_p = [predictions[i] for i in te]
            te_o = [outcomes[i] for i in te]
            if not te_p:
                continue
            eces.append(_adaptive_calibration_error(te_p, te_o))
    k = len(eces)
    if k == 0:
        return {"cv_calibration_error": 0.0, "cv_n_folds": 0}
    mean = sum(eces) / k
    var = sum((e - mean) ** 2 for e in eces) / (k - 1) if k > 1 else 0.0
    se = math.sqrt(var) / math.sqrt(k)
    return {
        "cv_calibration_error": round(mean, 6),
        "cv_ece_ci_low": round(mean - 1.96 * se, 6),
        "cv_ece_ci_high": round(mean + 1.96 * se, 6),
        "cv_n_folds": k,
    }


def tune_backend_structure(
    artifacts: list[FrozenArtifact],
    *,
    backend_name: str,
    competition_bucket: str,
    knob: str,
    grid: tuple[float, ...],
    base_params: dict,
    validation_start: str,
    holdout_start: str,
    n_iterations: int = 2000,
    seed: int = 42,
    cv_folds: int = 5,
    cv_repeats: int = 5,
    dataset_manifest_id: str | None = None,
    priors_as_of_date: str | None = None,
):
    """Pure core: sweep the knob, then attach the winner's raw CV-ECE + holdout
    metrics. Returns ``(report, winner_candidate_or_None)``.

    The returned winner is a CANDIDATE ``BackendParameterProfile`` with its
    ``metrics`` filled exactly as the promotion gate reads them — ready for
    ``register_parameter_profile``. No registration happens here (the CLI decides).
    """
    if knob not in _VALID_KNOBS:
        raise ValueError(f"knob must be one of {sorted(_VALID_KNOBS)}; got {knob!r}")
    backend = resolve_game_backend(backend_name)
    if backend is None:
        raise ValueError(f"backend {backend_name!r} is not registered; cannot tune")

    dataset_hash = stable_hash(
        {
            "artifacts": sorted(a.artifact_id for a in artifacts),
            "backend": backend_name,
            "bucket": competition_bucket,
            "knob": knob,
            "grid": list(grid),
            "base_params": base_params,
            "windows": [validation_start, holdout_start],
        }
    )

    candidates = build_candidates(
        backend_name=backend_name,
        backend_component_version=getattr(backend, "component_version", "unknown"),
        competition_bucket=competition_bucket,
        knob=knob,
        grid=grid,
        base_params=base_params,
        dataset_hash=dataset_hash,
        priors_as_of_date=priors_as_of_date,
        dataset_manifest_id=dataset_manifest_id,
    )

    report = sweep_backend_variants(
        artifacts,
        candidates,
        validation_start=validation_start,
        holdout_start=holdout_start,
        n_iterations=n_iterations,
        seed=seed,
        exact=True,
        selection_metric="raw_ece",
        dataset_manifest_id=dataset_manifest_id,
        dataset_hash=dataset_hash,
    )
    if report.winner_profile_id is None:
        return report, None

    winner = next(c for c in candidates if c.profile_id == report.winner_profile_id)
    winner_score = next(s for s in report.scores if s.profile_id == report.winner_profile_id)

    # Raw CV-ECE over the full eval universe with the winner's frozen params — the
    # robust metric the ECE_FLOOR prefers. Re-simulating via the sweep's own
    # primitive keeps the probabilities identical to what was selected.
    engine = OmegaSimulationEngine()
    all_records = _artifact_records(artifacts, validation_start)
    pairs = _simulate_pairs(
        engine,
        backend,
        winner.params,
        all_records,
        n_iterations=n_iterations,
        seed=seed,
        exact=True,
    )
    preds = [pairs[i][0] for i in sorted(pairs)]
    outs = [pairs[i][1] for i in sorted(pairs)]
    cv = _raw_cv_ece(
        preds, outs, folds=cv_folds, repeats=cv_repeats, seed=int(dataset_hash[:8], 16)
    )

    hold = winner_score.holdout
    metrics: dict = {
        "brier_score": hold.raw_brier if hold else None,
        "calibration_error": hold.raw_ece if hold else None,
        "log_loss": hold.raw_log_loss if hold else None,
        "n_eval": winner_score.n_holdout,
        "selection_metric": report.selection_metric,
        "validation_raw_ece": winner_score.validation.raw_ece,
        **cv,
    }
    winner = winner.model_copy(update={"metrics": metrics, "sample_size": winner_score.n_holdout})
    return report, winner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _artifacts_from_traces(graded: list[dict]) -> list[FrozenArtifact]:
    """Convert graded trace dicts to outcome-attached FrozenArtifacts.

    Skips traces without a parseable matchup or an attached score. Overrides the
    artifact date with the match ``decision_time`` (trace_to_artifact otherwise uses
    the replay RUN timestamp, which is a single day for a batch — defeating the
    no-leak validation/holdout split).
    """
    artifacts: list[FrozenArtifact] = []
    for t in graded:
        outcome = t.get("_outcome") or t.get("outcome")
        art = trace_to_artifact(t, outcome=outcome)
        decision_date = str(t.get("decision_time") or "")[:10]
        if decision_date:
            art = art.model_copy(update={"date": decision_date})
        if not art.home_team or not art.away_team or art.date == "":
            continue
        if not (art.outcome and art.outcome.get("home_score") is not None):
            continue
        artifacts.append(art)
    return artifacts


def load_graded_artifacts(
    *,
    league: str,
    db: str | None = None,
    historical_db: str | None = None,
    historical_only: bool = False,
    include_historical: bool = False,
) -> list[FrozenArtifact]:
    """Graded traces for the league as FrozenArtifacts (reusable by domain fitters).

    Mirrors omega-fit-calibration / cv-calibration-diagnostic loading: the live
    ``db`` unless ``historical_only``, plus the dedicated ``historical_db``
    (execution_mode='historical_replay', calibration-eligible, graded).
    """
    graded: list[dict] = []
    if not historical_only:
        store = TraceStore(db_path=db, read_only=True)
        log_effective_db(store, logger)
        graded.extend(store.get_graded_traces(league=league, limit=100_000))
        store.close()
    if historical_only or include_historical:
        hstore = TraceStore(db_path=historical_db, read_only=True)
        logger.info("historical DB: %s", historical_db)
        graded.extend(
            hstore.query_traces(
                league=league,
                execution_mode="historical_replay",
                has_outcome=True,
                calibration_eligible_only=True,
                limit=1_000_000,
            )
        )
        hstore.close()
    return _artifacts_from_traces(graded)


def _load_artifacts(args: argparse.Namespace) -> list[FrozenArtifact]:
    return load_graded_artifacts(
        league=args.league,
        db=args.db,
        historical_db=args.historical_db,
        historical_only=args.historical_only,
        include_historical=args.include_historical,
    )


def emit_structure_candidate_after_fit(
    *,
    backend_name: str,
    league: str,
    knob: str,
    base_params: dict,
    validation_start: str,
    holdout_start: str,
    bucket: str | None = None,
    db: str | None = None,
    historical_db: str | None = None,
    historical_only: bool = True,
    include_historical: bool = False,
    priors_as_of: str | None = None,
    register: bool = False,
    n_iterations: int = 2000,
) -> int:
    """Shared ``--emit-structure-candidate`` body for the domain fitters.

    After a domain fit produces base params (e.g. Dixon-Coles ``rho``), this loads
    graded artifacts, sweeps ``knob`` to minimize raw OOS ECE via
    :func:`tune_backend_structure`, prints the scorecard, and (``register``)
    persists the winner CANDIDATE — so the likelihood/moment fit and the
    calibration-scored structural fit are reconciled in one command. Returns a
    process exit code.
    """
    bucket = (bucket or resolve_calibration_bucket(league)).upper()
    artifacts = load_graded_artifacts(
        league=league,
        db=db,
        historical_db=historical_db,
        historical_only=historical_only,
        include_historical=include_historical,
    )
    if not artifacts:
        logger.error(
            "No graded artifacts for league=%s; cannot emit a structure candidate.", league
        )
        return 1
    grid = _KNOB_DEFAULT_GRID[knob]
    identity = _KNOB_IDENTITY[knob]
    if identity not in grid:
        grid = tuple(sorted({*grid, identity}))
    report, winner = tune_backend_structure(
        artifacts,
        backend_name=backend_name,
        competition_bucket=bucket,
        knob=knob,
        grid=grid,
        base_params=base_params,
        validation_start=validation_start,
        holdout_start=holdout_start,
        n_iterations=n_iterations,
        priors_as_of_date=priors_as_of,
    )
    _print_report(report, winner)
    if winner is None:
        return 1
    if register:
        store = TraceStore(db_path=db)
        try:
            incumbent = get_production_parameter_profile(store, backend_name, bucket)
            if incumbent is not None:
                winner = winner.model_copy(update={"incumbent_id": incumbent.profile_id})
            register_parameter_profile(store, winner)
        finally:
            store.close()
        logger.info(
            "Registered CANDIDATE %s (gate with omega-promote-parameter-profile).",
            winner.profile_id,
        )
    else:
        print("\n[dry-run] not registered. Add --register-structure to persist the CANDIDATE.")
    return 0


def _print_report(report, winner) -> None:
    print(
        f"\n{report.backend_name} / {report.competition_bucket}  "
        f"selection={report.selection_metric}  "
        f"val_events={report.n_validation_events} holdout_events={report.n_holdout_events}"
    )
    print(f"{'profile_id':<52}{'val_raw_ece':>12}{'n_val':>7}{'eligible':>9}")
    print("-" * 80)
    for s in sorted(report.scores, key=lambda s: (s.selection_value is None, s.selection_value)):
        sel = f"{s.selection_value:.4f}" if s.selection_value is not None else "   n/a"
        print(f"{s.profile_id:<52}{sel:>12}{s.n_validation:>7}{str(s.eligible):>9}")
    print("-" * 80)
    if report.winner_profile_id is None:
        print(f"NO WINNER: {report.selection_note}")
        return
    m = winner.metrics
    floor_src = "cv_calibration_error" if m.get("cv_n_folds") else "calibration_error"
    print(
        f"WINNER {winner.profile_id}\n"
        f"  params={winner.params}\n"
        f"  holdout raw_ece={m.get('calibration_error')}  brier={m.get('brier_score')}  "
        f"n_eval={m.get('n_eval')}\n"
        f"  raw CV-ECE={m.get('cv_calibration_error')} (folds={m.get('cv_n_folds')})  "
        f"-> gate reads {floor_src}\n"
        f"  promotion floor 0.05: "
        f"{'CLEARS' if (m.get(floor_src) or 1.0) <= 0.05 else 'ABOVE FLOOR'}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Tune a backend's structural sharpness knob to minimize raw OOS ECE."
    )
    p.add_argument("--backend", required=True, help="Registered backend name")
    p.add_argument("--league", required=True, help="League to load traces for (e.g. EPL)")
    p.add_argument(
        "--bucket",
        default=None,
        help="competition_bucket (default: resolve_calibration_bucket(league))",
    )
    p.add_argument("--knob", default=None, help="Knob to sweep (default per backend)")
    p.add_argument("--grid", default=None, help="Comma-separated knob values (default per knob)")
    p.add_argument(
        "--base-params",
        default="{}",
        help="JSON of required priors merged into every candidate, e.g. '{\"rho\": -0.12}'",
    )
    p.add_argument("--validation-start", required=True, help="YYYY-MM-DD (incl.)")
    p.add_argument("--holdout-start", required=True, help="YYYY-MM-DD; strictly after validation")
    p.add_argument("--n-iterations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=5)
    p.add_argument("--priors-as-of", default=None, help="Pin priors_* snapshot date on the profile")
    p.add_argument("--dataset-manifest-id", default=None)
    p.add_argument("--db", default=None, help="Live trace DB (default: production)")
    p.add_argument("--historical-db", default=None)
    p.add_argument("--include-historical", action="store_true")
    p.add_argument("--historical-only", action="store_true")
    p.add_argument("--register", action="store_true", help="Persist the winner as a CANDIDATE")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.historical_only and args.include_historical:
        logger.error("--historical-only and --include-historical are mutually exclusive.")
        return 1
    if (args.historical_only or args.include_historical) and not args.historical_db:
        logger.error("--historical-only/--include-historical require --historical-db.")
        return 1

    knob = args.knob or _BACKEND_DEFAULT_KNOB.get(args.backend)
    if knob is None:
        logger.error(
            "No default knob for backend %r; pass --knob (one of %s).",
            args.backend,
            sorted(_VALID_KNOBS),
        )
        return 1
    if knob not in _VALID_KNOBS:
        logger.error("Unknown --knob %r; choose from %s.", knob, sorted(_VALID_KNOBS))
        return 1

    if args.grid:
        try:
            grid = tuple(float(x) for x in args.grid.split(","))
        except ValueError:
            logger.error("--grid must be comma-separated numbers, e.g. 0.8,0.9,1.0")
            return 1
    else:
        grid = _KNOB_DEFAULT_GRID[knob]
    identity = _KNOB_IDENTITY[knob]
    if identity not in grid:
        grid = tuple(sorted({*grid, identity}))  # always score the production baseline

    try:
        base_params = json.loads(args.base_params)
        if not isinstance(base_params, dict):
            raise ValueError("must be a JSON object")
    except ValueError as exc:
        logger.error("--base-params is not a JSON object: %s", exc)
        return 1

    bucket = (args.bucket or resolve_calibration_bucket(args.league)).upper()

    artifacts = _load_artifacts(args)
    if not artifacts:
        logger.error("No graded artifacts loaded for league=%s.", args.league)
        return 1
    logger.info(
        "Loaded %d graded artifacts for %s (bucket=%s).", len(artifacts), args.league, bucket
    )

    try:
        report, winner = tune_backend_structure(
            artifacts,
            backend_name=args.backend,
            competition_bucket=bucket,
            knob=knob,
            grid=grid,
            base_params=base_params,
            validation_start=args.validation_start,
            holdout_start=args.holdout_start,
            n_iterations=args.n_iterations,
            seed=args.seed,
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            dataset_manifest_id=args.dataset_manifest_id,
            priors_as_of_date=args.priors_as_of,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    _print_report(report, winner)

    if winner is None:
        return 1
    if not args.register:
        print("\n[dry-run] not registered. Re-run with --register to persist the CANDIDATE.")
        return 0

    store = TraceStore(db_path=args.db)
    try:
        incumbent = get_production_parameter_profile(store, args.backend, bucket)
        if incumbent is not None:
            winner = winner.model_copy(update={"incumbent_id": incumbent.profile_id})
        register_parameter_profile(store, winner)
    finally:
        store.close()
    logger.info(
        "Registered CANDIDATE %s. Gate it with: omega-promote-parameter-profile "
        "--profile-id %s --auto --confirm-backtest-parity --parity-report <f> "
        "--confirm-clv-non-regression --clv-report <f>",
        winner.profile_id,
        winner.profile_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
