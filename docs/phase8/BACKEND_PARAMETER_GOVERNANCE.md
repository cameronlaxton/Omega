# Phase 8 — Backend-Parameter Governance + Soccer Structural Pilot

## Context

Calibration was being asked to repair raw probability errors that structurally
belong in the simulation/input model, and the backend parameters that generate
those raw probabilities were **ungoverned** (latest-`as_of_date`-wins, no gate, no
trace provenance). A calibration profile is a monotone 1-D transform of a marginal
probability; it cannot repair a wrong joint/tail (e.g. soccer draw mass set by the
Dixon-Coles `rho` + the xG→λ mapping, not by a moneyline map). Phase 8 gives
backend parameters the **same** versioned, traced, fail-closed, held-out-evaluated
discipline calibration already has — and enforces a hard boundary so the two stop
fighting over the same error.

Architecture verdict (unchanged from Phase 7): **one engine platform + backend
registry with sport-specific deterministic backends**, now with an explicit
governed parameter-profile layer. Not separate engines (would fork
calibration/edge/staking/grading/trace); not one generic backend (the sports need
genuinely different joint structure).

## What landed (P8.0–P8.2)

### P8.0 — the governance rail (sport-agnostic, additive)

- **Shared gate engine** — `omega/core/governance/promotion_gates.py`. The
  profile-type-agnostic fail-closed gates (sample size, ECE floor, Brier
  improvement, log-loss non-regression, evidence-backed parity/CLV) lifted out of
  calibration behind a structural `GateCandidate` protocol. `calibration/promotion.py`
  re-exports it, bit-identical. **One** promotion definition for both planes.
- **Governed `BackendParameterProfile`** — `omega/core/simulation/parameter_profile.py`
  + the `parameter_profiles` table (schema V19) + `omega/trace/parameter_profiles.py`.
  `register` / `get_production` / `promote_parameter_profile` mirror the
  calibration registry; exactly one PRODUCTION per `(backend_name,
  competition_bucket)` (partial unique index); promotion composes the shared gate
  on the candidate's **RAW** (pre-calibration) held-out metrics. Per-entity priors
  (`priors_nfl_dispersion`, `priors_tennis*`) stay immutable as-of-date snapshots
  that a profile **pins** via `priors_as_of_date` — governance concentrated in the
  profile envelope, not duplicated per row.
- **Trace provenance** — `traces.parameter_profile_ref` (schema V20) stamped on
  every persist. A probability is now attributable to its exact parameter set;
  `parameter_pin_status` emits a loud `freshness=unpinned` event when a trace has
  no ref, so a non-reproducible live re-read in replay/lab is never silent.

### P8.1 — the lab backend-variant axis

`omega/strategy/backtest/variant_sweep.py` — `sweep_backend_variants()` compares
competing parameter-profile candidates on a frozen dataset, **selects on RAW
validation ECE**, and scores only the winner once on a **sealed holdout**. Thin
orchestration, zero duplicated logic: reuses the production sim seam
(`run_fast_game_simulation(backend=, prior_payload=, exact=True)`), the no-leak
`partition_fold`, the single `probability_metrics` evaluator, and `FrozenArtifact`.
`exact=True` removes MC selection noise (optimizer's curse).

### P8.2 — soccer structural pilot (touches the live backend, safely)

- **Backend** (`soccer_bivariate_poisson.py`) now reads `home_advantage`,
  `lambda_scale`, and `first_half_share` from `prior_payload`, **each defaulting to
  the historical constant** — output is bit-identical when no profile is injected
  (the live World Cup state); a promoted profile overrides them. It echoes
  `parameter_profile_ref` for provenance.
- **Gatherer** (`build_game_prior_payload`) merges a promoted soccer
  `BackendParameterProfile`'s params + stamps the ref, alongside the Dixon-Coles
  rho. No production profile ⇒ payload unchanged (purely additive). Replay-safe: a
  caller-supplied ref/param is never overwritten.
- **CLI** — `omega-promote-parameter-profile` (fail-closed; mirrors
  `omega-promote-profile`).

## The boundary: backend tuning ⊥ calibration fitting

1. **Different objectives.** Backend-parameter tuning minimizes RAW
   (pre-calibration) forecast error on validation. Calibration minimizes the
   *residual* miscalibration of an already-promoted backend+params version.
2. **Strict ordering.** Promote backend params (raw metrics) → THEN fit/promote
   calibration on top, bound to that version. Fix the joint/tail first, calibrate
   the residual second.
3. **Surfaced, not absorbed.** If raw validation ECE can't clear the floor by
   tuning structure, that is reported as a model finding — not muffled by a
   calibration clip.

`rho` stays on its existing live Dixon-Coles path for now (mid-tournament safety);
folding it into the parameter profile is a follow-up.

## Operator runbook — soccer structural fit → sweep → promote

The fit is composition of existing pieces (no dedicated fit CLI yet):

1. Build a frozen `FrozenArtifact` dataset for the competition (manifest-hashed,
   outcomes attached at grading time only) over a window that ends before
   `validation_start`.
2. Construct ≥2 candidate `BackendParameterProfile`s over a structural-knob grid
   (`home_advantage`, `lambda_scale`, `first_half_share`); `register_parameter_profile`
   each (CANDIDATE).
3. `sweep_backend_variants(artifacts, candidates, validation_start=…,
   holdout_start=…, exact=True)` → side-by-side raw-validation metrics + a winner
   scored once on the sealed holdout.
4. Stamp the winner's holdout raw metrics onto its profile, then
   `omega-promote-parameter-profile --profile-id <winner> --auto
   --confirm-backtest-parity --parity-report … --confirm-clv-non-regression
   --clv-report …` (fail-closed).
5. Re-fit the FIFA_INTL / EPL calibration profile **bound to** the promoted backend
   version (P8.3), and confirm residual ECE clears the floor.

## Deferred

- **P8.3** — bind calibration profiles to `backend_component_version` /
  `param_profile_id`; calibration promotion fails closed on backend mismatch
  (turns today's after-the-fact audit into a gate).
- **P8.4** — tennis (`priors_tennis*` status + pressure/SPW structure profile) and
  NFL (`nfl_nb_v2`: score correlation + team/context dispersion) adopt the rail.
- Fold soccer `rho` into the parameter profile; a dedicated
  `omega-fit-soccer-structure` convenience CLI.

## Hard-constraint compliance

No duplicated calibration/edge/staking/grading/promotion logic (shared gate;
lab reuses grading) · no parallel pipeline (one `analyze()` path, one backtest
engine, isolated-DB guard) · deterministic logic in Python backends, never LLM
text · replay reproducible + outcome-blind (provenance ref makes it *more* so) ·
train/validation/holdout discipline reused from `walk_forward` · holdout never
tuned on (selection on validation, single-shot promotion) · every backend
parameter profile versioned + traceable · historical replay isolated from
production traces · promotion fail-closed (auto-promote default-off,
evidence-backed).
