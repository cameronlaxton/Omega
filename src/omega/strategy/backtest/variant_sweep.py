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

Two planes share one selection/seal skeleton (:func:`_select_and_seal`):

* **game** — :func:`sweep_backend_variants` re-simulates :class:`FrozenArtifact`
  records through the engine seam (``run_fast_game_simulation`` with a
  ``backend=`` override) and pairs on the home-win plane;
* **prop** — :func:`sweep_prop_backend_variants` re-simulates
  :class:`FrozenPropArtifact` records through the prop-backend seam
  (``resolve_prop_backend(...).run(PropSimulationInput(...))`` — props have no
  engine method) and pairs via the single shared prop grading rule
  (``prop_pairs_for_trace``, the same extractor the calibration fitter and the
  historical walk-forward use), so prop grading cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from omega.core.calibration.fitter import prop_pairs_for_trace
from omega.core.simulation.backends import (
    PropSimulationInput,
    resolve_game_backend,
    resolve_prop_backend,
)
from omega.core.simulation.engine import OmegaSimulationEngine
from omega.core.simulation.parameter_profile import BackendParameterProfile
from omega.historical.contracts import MetricBlock, WalkForwardConfig
from omega.historical.metrics import probability_metrics
from omega.historical.walk_forward import partition_fold
from omega.strategy.artifacts import FrozenArtifact, FrozenPropArtifact

# Raw-metric fields a sweep may rank on (all "lower is better").
SELECTION_METRICS = ("raw_ece", "raw_brier", "raw_log_loss")
_FAR_FUTURE = "9999-12-31"


class VariantScore(BaseModel):
    """One candidate's raw-metric scorecard within a sweep."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    validation: MetricBlock
    holdout: MetricBlock | None = None  # scored only for the winner (sealed holdout)
    n_validation: int  # events SCORED (the common intersection across all candidates)
    n_simulated: int = 0  # events this candidate individually simulated (skip transparency)
    n_holdout: int = 0
    selection_value: float | None = None  # the ranked raw metric; None = not scorable
    eligible: bool = True


class VariantSweepReport(BaseModel):
    """Side-by-side raw-metric comparison of backend parameter-profile candidates."""

    model_config = ConfigDict(extra="forbid")

    backend_name: str
    competition_bucket: str
    plane: str = "game"  # which decision plane was scored ("game" | "prop")
    selection_metric: str
    scores: list[VariantScore] = Field(default_factory=list)
    winner_profile_id: str | None = None
    selection_inconclusive: bool = False  # candidates indistinguishable on the scored plane
    selection_note: str | None = None  # why there is no winner, when there isn't
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


def _artifact_records(artifacts: list[FrozenArtifact], validation_start: str) -> list[dict]:
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


def _simulate_pairs(
    engine: OmegaSimulationEngine,
    backend,
    params: dict | None,
    records: list[dict],
    *,
    n_iterations: int,
    seed: int,
    exact: bool,
) -> dict[int, tuple[float, int]]:
    """Simulate each record with this variant; return ``{record_index: (raw, y)}``.

    Outcome-blind: the engine is called with no outcome; the ``(raw, y)`` pair is
    formed only after. ``y`` mirrors the calibration game plane (``_game_pair``):
    home win -> 1, otherwise 0. Records the variant skips (sim unsuccessful, or no
    graded outcome) are simply ABSENT from the map — callers then score every
    candidate on the INTERSECTION of indices all candidates produced, so a variant
    that skips more games cannot win on a smaller, easier subset.
    """
    pairs: dict[int, tuple[float, int]] = {}
    for idx, r in enumerate(records):
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
        pairs[idx] = (sim["home_win_prob"] / 100.0, 1 if hs > as_ else 0)
    return pairs


def _score_pairs(pairs: dict[Any, tuple[float, int]], indices: list[Any]) -> MetricBlock:
    """Raw-only metrics over ``indices`` via the single shared evaluator.

    ``probability_metrics(raw, raw, outs)`` passes raw as both raw + "calibrated",
    so the identity calibration yields the raw-only block.
    """
    raw = [pairs[i][0] for i in indices]
    outs = [pairs[i][1] for i in indices]
    return probability_metrics(raw, raw, outs)


def _validate_sweep_candidates(
    candidates: list[BackendParameterProfile],
    *,
    selection_metric: str,
    validation_start: str,
    holdout_start: str,
) -> tuple[str, str]:
    """Shared fail-closed input guards; returns the sweep's (backend, bucket).

    Raises ``ValueError`` on an empty candidate list, an unknown selection
    metric, a holdout that does not strictly follow validation, or mixed
    backend/bucket across candidates (a sweep is within one of each).
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
    return backend_name, bucket


def _select_and_seal(
    candidates: list[BackendParameterProfile],
    val_pairs: dict[str, dict[Any, tuple[float, int]]],
    simulate_holdout: Callable[[BackendParameterProfile], dict[Any, tuple[float, int]]],
    *,
    selection_metric: str,
    min_validation_samples: int,
    min_selection_margin: float,
) -> tuple[list[VariantScore], str | None, bool, str | None]:
    """Plane-agnostic selection + holdout seal shared by the game and prop sweeps.

    Scores every candidate on the INTERSECTION of the pair keys all candidates
    produced (a variant cannot win on a smaller, easier subset), applies the
    eligibility floor and the noise-margin guard, then scores ONLY the winner on
    the sealed holdout, exactly once, via ``simulate_holdout``. Returns
    ``(scores, winner_profile_id, selection_inconclusive, selection_note)``.
    """
    common = sorted(set.intersection(*[set(p) for p in val_pairs.values()])) if val_pairs else []

    scores: list[VariantScore] = []
    for c in candidates:
        pairs = val_pairs[c.profile_id]
        block = _score_pairs(pairs, common)
        sel = getattr(block, selection_metric)
        eligible = len(common) >= min_validation_samples and sel is not None
        scores.append(
            VariantScore(
                profile_id=c.profile_id,
                validation=block,
                n_validation=len(common),
                n_simulated=len(pairs),
                selection_value=sel if eligible else None,
                eligible=eligible,
            )
        )

    eligible_scores = [s for s in scores if s.eligible and s.selection_value is not None]
    winner_id: str | None = None
    inconclusive = False
    note: str | None = None
    if not eligible_scores:
        note = (
            f"no candidate reached min_validation_samples={min_validation_samples} on the "
            f"{len(common)}-event common set"
        )
    elif len(eligible_scores) == 1:
        winner_id = eligible_scores[0].profile_id
    else:
        ranked = sorted(eligible_scores, key=lambda s: s.selection_value)
        margin = ranked[1].selection_value - ranked[0].selection_value
        if margin < min_selection_margin:
            # The swept knobs do not move the scored plane enough to separate the top
            # candidates (e.g. first_half_share is ~invisible to home-win ECE). Refuse
            # to promote on noise rather than pick arbitrarily.
            inconclusive = True
            note = (
                f"top candidates within {min_selection_margin} on {selection_metric} "
                f"(margin={margin:.6f}); no discriminating signal on the scored plane"
            )
        else:
            winner_id = ranked[0].profile_id

    if winner_id is not None:
        # Seal: score ONLY the winner on the holdout, exactly once.
        winner_score = next(s for s in scores if s.profile_id == winner_id)
        winner_candidate = next(c for c in candidates if c.profile_id == winner_id)
        holdout_pairs = simulate_holdout(winner_candidate)
        winner_score.holdout = _score_pairs(holdout_pairs, sorted(holdout_pairs))
        winner_score.n_holdout = len(holdout_pairs)

    return scores, winner_id, inconclusive, note


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
    min_selection_margin: float = 1e-4,
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
    backend_name, bucket = _validate_sweep_candidates(
        candidates,
        selection_metric=selection_metric,
        validation_start=validation_start,
        holdout_start=holdout_start,
    )
    backend = resolve_game_backend(backend_name)
    if backend is None:
        raise ValueError(f"backend {backend_name!r} is not registered; cannot sweep")

    records = _artifact_records(artifacts, validation_start)
    # Reuse the calibration walk-forward's no-leak primitive: validation is strictly
    # before holdout_start, holdout is the strictly-after remainder (sealed).
    config = WalkForwardConfig(mode="expanding")
    validation_recs, holdout_recs, _ = partition_fold(records, holdout_start, _FAR_FUTURE, config)
    # Hard guard mirroring run_walk_forward's: no validation event may reach the
    # holdout window.
    assert all(r["_dt"] < holdout_start for r in validation_recs), (
        "variant-sweep validation/holdout leak"
    )

    engine = OmegaSimulationEngine()

    # Simulate every candidate over the validation set, then score them all on the
    # INTERSECTION of events every candidate produced — so a variant cannot win on a
    # smaller, easier subset (e.g. one whose lambda_scale skipped hard games).
    val_pairs = {
        c.profile_id: _simulate_pairs(
            engine,
            backend,
            c.params,
            validation_recs,
            n_iterations=n_iterations,
            seed=seed,
            exact=exact,
        )
        for c in candidates
    }
    scores, winner_id, inconclusive, note = _select_and_seal(
        candidates,
        val_pairs,
        lambda cand: _simulate_pairs(
            engine,
            backend,
            cand.params,
            holdout_recs,
            n_iterations=n_iterations,
            seed=seed,
            exact=exact,
        ),
        selection_metric=selection_metric,
        min_validation_samples=min_validation_samples,
        min_selection_margin=min_selection_margin,
    )

    return VariantSweepReport(
        backend_name=backend_name,
        competition_bucket=bucket,
        plane="game",
        selection_metric=selection_metric,
        scores=scores,
        winner_profile_id=winner_id,
        selection_inconclusive=inconclusive,
        selection_note=note,
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


# ---------------------------------------------------------------------------
# Prop plane
# ---------------------------------------------------------------------------


def _gradeable_prop_rows(art: FrozenPropArtifact) -> bool:
    """True when the artifact carries at least one non-push/void over/under row."""
    return any(
        (r.get("side") or "").lower() in ("over", "under")
        and r.get("result") not in ("push", "void")
        for r in art.prop_outcomes
    )


def _prop_artifact_records(
    artifacts: list[FrozenPropArtifact], validation_start: str
) -> list[dict]:
    """Eval universe: prop artifacts on/after ``validation_start`` with a gradeable row.

    Same ``{"_dt": date, "art": artifact}`` shape as :func:`_artifact_records`,
    so ``partition_fold`` splits both planes with the identical no-leak rule.
    """
    return [
        {"_dt": a.date, "art": a}
        for a in artifacts
        if a.date >= validation_start and _gradeable_prop_rows(a)
    ]


def _simulate_prop_pairs(
    backend,
    params: dict | None,
    records: list[dict],
    *,
    n_iterations: int,
    seed: int,
    exact: bool,
) -> dict[tuple[int, int], tuple[float, int]]:
    """Simulate each prop record with this variant; return ``{(rec, j): (raw, y)}``.

    The prop analogue of :func:`_simulate_pairs`, invoked at the production prop
    seam (``backend.run(PropSimulationInput(...))`` — the same direct dispatch
    ``analyze_player_prop`` uses; props have no engine method). The candidate's
    structural params ride ``prior_payload`` on top of the artifact's own base
    ``nb_dispersion_k``, exactly how a promoted profile would reach the backend.

    Outcome-blind: the backend prices the over/under first; the ``(raw, y)``
    pairs are formed only after, by the single shared grading rule
    (``prop_pairs_for_trace``) over the artifact's attached outcome rows — so
    push/void exclusion and side selection cannot drift from the calibration
    fitter. Keys are ``(record_index, pair_index)``; pair survivorship depends
    only on the outcome rows (never on the knob — parametric backends always
    price both sides), so the key set is stable across candidates and the
    callers' intersection discipline holds. A record whose simulation raises is
    simply ABSENT from the map, excluding it for every candidate.
    """
    pairs: dict[tuple[int, int], tuple[float, int]] = {}
    for idx, r in enumerate(records):
        a: FrozenPropArtifact = r["art"]
        prior = dict(params or {})
        prior.setdefault("nb_dispersion_k", a.nb_dispersion_k)
        try:
            sim = backend.run(
                PropSimulationInput(
                    player_name=a.player_name,
                    league=a.league,
                    stat_type=a.stat_type,
                    line=a.line,
                    projection_mean=a.projection_mean,
                    n_iter=n_iterations,
                    seed=a.simulation_seed if a.simulation_seed is not None else seed,
                    projection_std=a.projection_std,
                    prior_payload=prior,
                    exact=exact,
                )
            )
        except ValueError:
            # Prop backends fail loud on unusable inputs (vs the game engine's
            # success=False); absence from the map is the sweep's skip semantic.
            continue
        graded = prop_pairs_for_trace(
            {
                "predictions": {
                    "over_prob": sim.get("over_prob"),
                    "under_prob": sim.get("under_prob"),
                },
                "_prop_outcomes": a.prop_outcomes,
            }
        )
        for j, pair in enumerate(graded):
            pairs[(idx, j)] = pair
    return pairs


def sweep_prop_backend_variants(
    artifacts: list[FrozenPropArtifact],
    candidates: list[BackendParameterProfile],
    *,
    validation_start: str,
    holdout_start: str,
    n_iterations: int = 2000,
    seed: int = 42,
    exact: bool = True,
    selection_metric: str = "raw_ece",
    min_validation_samples: int = 20,
    min_selection_margin: float = 1e-4,
    dataset_manifest_id: str | None = None,
    dataset_hash: str | None = None,
) -> VariantSweepReport:
    """Compare PROP backend parameter-profile candidates without leakage.

    The prop-plane twin of :func:`sweep_backend_variants`: identical fail-closed
    guards, no-leak partition, raw-metric intersection selection, and sealed
    holdout — only the simulation seam differs (the prop-backend registry). A
    prop record is one-to-many: every gradeable attached outcome row is its own
    ``(raw_prob, outcome)`` decision, mirroring the calibration prop plane.

    ``exact=True`` is honored by parametric prop backends (negative binomial's
    closed-form CDF — zero MC selection noise); MC-only backends ignore the
    flag, the same contract as the game path.
    """
    backend_name, bucket = _validate_sweep_candidates(
        candidates,
        selection_metric=selection_metric,
        validation_start=validation_start,
        holdout_start=holdout_start,
    )
    backend = resolve_prop_backend(backend_name)
    if backend is None:
        raise ValueError(f"prop backend {backend_name!r} is not registered; cannot sweep")

    records = _prop_artifact_records(artifacts, validation_start)
    config = WalkForwardConfig(mode="expanding")
    validation_recs, holdout_recs, _ = partition_fold(records, holdout_start, _FAR_FUTURE, config)
    assert all(r["_dt"] < holdout_start for r in validation_recs), (
        "prop variant-sweep validation/holdout leak"
    )

    val_pairs = {
        c.profile_id: _simulate_prop_pairs(
            backend,
            c.params,
            validation_recs,
            n_iterations=n_iterations,
            seed=seed,
            exact=exact,
        )
        for c in candidates
    }
    scores, winner_id, inconclusive, note = _select_and_seal(
        candidates,
        val_pairs,
        lambda cand: _simulate_prop_pairs(
            backend,
            cand.params,
            holdout_recs,
            n_iterations=n_iterations,
            seed=seed,
            exact=exact,
        ),
        selection_metric=selection_metric,
        min_validation_samples=min_validation_samples,
        min_selection_margin=min_selection_margin,
    )

    return VariantSweepReport(
        backend_name=backend_name,
        competition_bucket=bucket,
        plane="prop",
        selection_metric=selection_metric,
        scores=scores,
        winner_profile_id=winner_id,
        selection_inconclusive=inconclusive,
        selection_note=note,
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
