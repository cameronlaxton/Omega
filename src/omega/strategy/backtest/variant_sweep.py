"""Backend parameter-profile variant sweep (Phase 8 P8.1).

The lab axis that compares competing :class:`BackendParameterProfile` candidates
for one backend on a frozen historical dataset, scoring each on its **raw**
(pre-calibration) forecast quality, selecting a winner on a **validation** split,
and scoring only that winner once on a **sealed holdout**. The winner's holdout
metrics are what feed :func:`omega.trace.parameter_profiles.promote_parameter_profile`
(P8.2), so a backend's structural parameters must clear the quality floor on their
*uncalibrated* output — the whole point of the rail.

This is a thin orchestration over existing machinery — it adds **no** new
simulation, grading, metric, partition, or promotion logic:

* simulation — the production seam ``OmegaSimulationEngine.run_fast_game_simulation``
  with a per-call ``backend=`` override + ``prior_payload=`` params + ``exact=True``;
* no-leak partition — ``omega.historical.walk_forward.partition_fold`` (train/select
  strictly before the cutoff), the single place the no-future-leak rule lives;
* metrics — ``omega.historical.metrics.probability_metrics`` →
  ``CalibrationFitter.evaluate``, the single ECE/Brier/log-loss implementation;
* dataset — frozen :class:`FrozenArtifact` objects (manifest-hashed upstream).

Outcome-blindness is preserved: the engine never sees an outcome; the
``(raw_prob, outcome)`` pair is formed only *after* the simulation decides.
``exact=True`` removes Monte-Carlo selection noise (optimizer's curse) so a variant
wins on signal, not sampling luck.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omega.core.simulation.backends import resolve_game_backend
from omega.core.simulation.engine import OmegaSimulationEngine
from omega.core.simulation.parameter_profile import BackendParameterProfile
from omega.historical.contracts import MetricBlock, WalkForwardConfig
from omega.historical.metrics import probability_metrics
from omega.historical.walk_forward import partition_fold
from omega.strategy.artifacts import FrozenArtifact

# Raw-metric fields a sweep may rank on (all "lower is better").
SELECTION_METRICS = ("raw_ece", "raw_brier", "raw_log_loss")
_FAR_FUTURE = "9999-12-31"


class VariantScore(BaseModel):
    """One candidate's raw-metric scorecard within a sweep."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    validation: MetricBlock
    holdout: MetricBlock | None = None  # scored only for the winner (sealed holdout)
    n_validation: int
    n_holdout: int = 0
    selection_value: float | None = None  # the ranked raw metric; None = not scorable
    eligible: bool = True


class VariantSweepReport(BaseModel):
    """Side-by-side raw-metric comparison of backend parameter-profile candidates."""

    model_config = ConfigDict(extra="forbid")

    backend_name: str
    competition_bucket: str
    selection_metric: str
    scores: list[VariantScore] = Field(default_factory=list)
    winner_profile_id: str | None = None
    n_candidates: int = 0
    validation_start: str
    holdout_start: str
    n_validation_events: int = 0
    n_holdout_events: int = 0
    exact: bool = True
    seed: int = 42
    n_iterations: int = 2000
    holdout_sealed: bool = True
    dataset_manifest_id: str | None = None
    dataset_hash: str | None = None


def _artifact_records(
    artifacts: list[FrozenArtifact], validation_start: str
) -> list[dict]:
    """Eval universe: artifacts on/after ``validation_start`` with a graded outcome.

    Shaped as ``{"_dt": date, "art": artifact}`` so ``partition_fold`` (which keys
    on ``_dt``) can split them with the identical no-leak discipline used by the
    calibration walk-forward. Anything before ``validation_start`` is the upstream
    fit window and is not the lab's concern.
    """
    records: list[dict] = []
    for a in artifacts:
        oc = a.outcome or {}
        if oc.get("home_score") is None or oc.get("away_score") is None:
            continue
        if a.date < validation_start:
            continue
        records.append({"_dt": a.date, "art": a})
    return records


def _eval_variant(
    engine: OmegaSimulationEngine,
    backend,
    params: dict | None,
    records: list[dict],
    *,
    n_iterations: int,
    seed: int,
    exact: bool,
) -> tuple[MetricBlock, int]:
    """Simulate every record with this variant and score RAW home-win calibration.

    Outcome-blind: the engine is called with no outcome; the ``(raw, y)`` pair is
    formed only after. ``y`` mirrors the calibration game plane (``_game_pair``):
    home win -> 1, otherwise 0.
    """
    raw: list[float] = []
    outs: list[int] = []
    for r in records:
        a: FrozenArtifact = r["art"]
        sim = engine.run_fast_game_simulation(
            home_team=a.home_team,
            away_team=a.away_team,
            league=a.league,
            n_iterations=n_iterations,
            home_context=a.home_context or None,
            away_context=a.away_context or None,
            seed=a.simulation_seed if a.simulation_seed is not None else seed,
            spread_home=a.odds.get("spread_home"),
            over_under=a.odds.get("over_under"),
            prior_payload=params or None,
            backend=backend,
            exact=exact,
        )
        if not sim.get("success"):
            continue
        oc = a.outcome or {}
        hs, as_ = oc.get("home_score"), oc.get("away_score")
        if hs is None or as_ is None:
            continue
        raw.append(sim["home_win_prob"] / 100.0)
        outs.append(1 if hs > as_ else 0)
    # probability_metrics(raw, raw, outs): passing raw as both raw + "calibrated"
    # yields the raw-only metrics from the single shared evaluator (identity cal).
    return probability_metrics(raw, raw, outs), len(raw)


def sweep_backend_variants(
    artifacts: list[FrozenArtifact],
    candidates: list[BackendParameterProfile],
    *,
    validation_start: str,
    holdout_start: str,
    n_iterations: int = 2000,
    seed: int = 42,
    exact: bool = True,
    selection_metric: str = "raw_ece",
    min_validation_samples: int = 20,
    dataset_manifest_id: str | None = None,
    dataset_hash: str | None = None,
) -> VariantSweepReport:
    """Compare backend parameter-profile candidates without leakage.

    All candidates must name the same registered backend and the same
    ``competition_bucket`` (a sweep is within one backend+bucket). Each is scored
    on the validation window ``[validation_start, holdout_start)``; the winner
    (lowest ``selection_metric``) is then scored once on the sealed holdout
    ``[holdout_start, end]``. The holdout is never used for selection.

    Raises ``ValueError`` (fail closed) on an empty candidate list, mismatched
    backend/bucket across candidates, or an unregistered backend.
    """
    if not candidates:
        raise ValueError("sweep requires at least one candidate parameter profile")
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    if holdout_start <= validation_start:
        raise ValueError("holdout_start must be strictly after validation_start")

    backend_name = candidates[0].backend_name
    bucket = candidates[0].competition_bucket
    for c in candidates:
        if c.backend_name != backend_name or c.competition_bucket != bucket:
            raise ValueError(
                "a sweep is within one (backend_name, competition_bucket); got mixed "
                f"{c.backend_name!r}/{c.competition_bucket!r} vs {backend_name!r}/{bucket!r}"
            )
    backend = resolve_game_backend(backend_name)
    if backend is None:
        raise ValueError(f"backend {backend_name!r} is not registered; cannot sweep")

    records = _artifact_records(artifacts, validation_start)
    # Reuse the calibration walk-forward's no-leak primitive: validation is strictly
    # before holdout_start, holdout is the strictly-after remainder (sealed).
    config = WalkForwardConfig(mode="expanding")
    validation_recs, holdout_recs, _ = partition_fold(
        records, holdout_start, _FAR_FUTURE, config
    )
    # Hard guard mirroring run_walk_forward's: no validation event may reach the
    # holdout window.
    assert all(r["_dt"] < holdout_start for r in validation_recs), "variant-sweep validation/holdout leak"

    engine = OmegaSimulationEngine()

    scores: list[VariantScore] = []
    for c in candidates:
        block, n_val = _eval_variant(
            engine, backend, c.params, validation_recs,
            n_iterations=n_iterations, seed=seed, exact=exact,
        )
        sel = getattr(block, selection_metric)
        eligible = n_val >= min_validation_samples and sel is not None
        scores.append(
            VariantScore(
                profile_id=c.profile_id,
                validation=block,
                n_validation=n_val,
                selection_value=sel if eligible else None,
                eligible=eligible,
            )
        )

    eligible_scores = [s for s in scores if s.eligible and s.selection_value is not None]
    winner_id: str | None = None
    if eligible_scores:
        winner = min(eligible_scores, key=lambda s: s.selection_value)
        winner_id = winner.profile_id
        # Seal: score ONLY the winner on the holdout, exactly once.
        winner_candidate = next(c for c in candidates if c.profile_id == winner_id)
        holdout_block, n_hold = _eval_variant(
            engine, backend, winner_candidate.params, holdout_recs,
            n_iterations=n_iterations, seed=seed, exact=exact,
        )
        winner.holdout = holdout_block
        winner.n_holdout = n_hold

    return VariantSweepReport(
        backend_name=backend_name,
        competition_bucket=bucket,
        selection_metric=selection_metric,
        scores=scores,
        winner_profile_id=winner_id,
        n_candidates=len(candidates),
        validation_start=validation_start,
        holdout_start=holdout_start,
        n_validation_events=len(validation_recs),
        n_holdout_events=len(holdout_recs),
        exact=exact,
        seed=seed,
        n_iterations=n_iterations,
        dataset_manifest_id=dataset_manifest_id,
        dataset_hash=dataset_hash,
    )
