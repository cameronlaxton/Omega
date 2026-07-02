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

## P8.5 — the structural-tuning loop, closed

P8.0–P8.2 built the rail, the lab axis, and the live-backend seam, but the loop was
**open**: `sweep_backend_variants` had no production caller, the calibration
sharpness lever was not reachable through the sim seam, and the structural fitters
optimized likelihood / moments, not the ECE the gate measures. P8.5 connects the
existing pieces and adds the missing lever.

- **Mean-preserving sharpness knobs** (identity-default ⇒ the engine is
  bit-identical until a profile is promoted), each matched to its archetype's
  variance structure:
  - `margin_sd_scale` — Normal score-SD multiplier (NBA / american-football);
    mean-independent temperature scaling done in-model.
  - `lambda_gap_scale` — mean-**total**-preserving Poisson λ-gap compression
    (soccer DC, MLB, NHL): softens moneyline/spread/draw at fixed `E[total]`, so
    over/under calibration is untouched. The honest Poisson analogue of shrinkage —
    a blunt both-λ multiplier would wrongly move the total.
  - `nb_k_scale` — negative-binomial dispersion-`k` multiplier (NFL scores, props).

  The knobs apply at the shared **parameter** point (λ / σ / k), so the MC sampler
  and the exact evaluator consume identical adjusted params — MC and exact cannot
  drift. `home_advantage` (already reachable) is also sweepable, as a **bias**
  corrector for the systematic-mean-error case.
- **Driver CLI** — `omega-fit-backend-structure` (the dedicated fit CLI P8.2 noted
  was missing). Thin orchestration, no new sim/metric/promotion logic: load graded
  historical artifacts → build a candidate grid over one knob (identity always
  included) → `sweep_backend_variants` (raw-ECE selection, sealed holdout, noise
  guard) → compute the winner's no-leak **raw cross-validated ECE** (the metric
  `ECE_FLOOR` reads) via the shared `raw_oos` estimator → `register_parameter_profile`
  the winner as CANDIDATE. `--dry-run` is the default.
- **Calibration-aware domain fitters.** `omega-fit-dixon-coles` and
  `omega-fit-nfl-dispersion` gained an opt-in `--emit-structure-candidate` that, after
  their existing NLL/method-of-moments fit, chains the same structural sweep with the
  just-fitted params as the base (`rho` for soccer; the NFL game-score bucket for NB) —
  so the likelihood/moment fit and the calibration-scored fit are reconciled in one
  command. Default-off, so existing fitter behavior is unchanged.
- **Triage first.** `omega-cv-calibration-diagnostic` decides *which* buckets are
  structure-bound, so degrees of freedom are spent only where the residual is real:
  **MISCALIBRATED** → tune here; **AT FLOOR** → provisional-maturity calibration,
  not structure; **OVER-FIT / CALIBRATED** → identity profile + calibration
  `method=none` (the diagnostic's own OVER-FIT guard prevents adding noise).

## P8.3 — calibration/backend binding, closed

A calibration profile corrects the residual miscalibration of ONE raw-probability
substrate: a backend (name + component version) running one governed
`BackendParameterProfile`. Before P8.3 nothing recorded which substrate a
`CalibrationProfile` was fit against, so promoting a new backend parameter
profile (or bumping a backend version) silently left a stale calibration map
applying to probabilities it was never fit on. P8.3 makes the binding explicit
and enforced:

- **Declaration** — `CalibrationProfile.backend_binding`
  (`CalibrationBackendBinding`: `backend_name`, `backend_component_version`,
  `param_profile_id`). Three statuses (`binding_status()`):
  - `bound` — the fit recorded at least one substrate identity field; applied
    only when the live substrate matches every recorded field.
  - `unpinned` — the fit checked its source traces and found NO substrate
    identity (homogeneously unpinned dataset); declared explicitly, applies
    anywhere, flagged in the audit.
  - `legacy` — the profile predates P8.3; applies as before, flagged in the
    audit. Existing registry profiles keep working; refit to earn a binding.
- **Fit-side recording** — `omega-fit-calibration` derives the substrate of every
  trace that contributes a pair on the requested plane
  (`substrate_ref_for_trace`: the result's own `simulation_backend` /
  `component_version` plus the governed `parameter_profile_ref` echo — recorded
  provenance only, nothing guessed). One substrate → the candidate carries it.
  **Mixed substrates → the slice is refused (fail-closed, no bypass)**; window
  the fit (`--train-start`/`--train-end`) to a single substrate instead. This is
  the strict-ordering contract made mechanical: promote backend params first,
  then fit calibration bound to that exact version.
- **Runtime enforcement** — the single shared selection walk
  (`_get_applicable_profile`, used by production `analyze()` AND the backtest
  engine) checks the binding against the live substrate threaded from the
  simulation that produced the probability (game plane: `sim_result` identity +
  echoed ref, including derivative-market EdgeConsumers; prop plane: the resolved
  prop backend's identity + the echoed ref). A bound profile that mismatches —
  including an *unknown* substrate — is skipped and the walk continues
  (league → sport_family → global → static), so the existing output-mode/
  downgrade policy takes over exactly as if no profile existed. A profile fit on
  one parameter profile is never silently applied on another.
- **Honesty surface** — every `CalibrationAudit` now carries `binding_status`
  (of the applied profile) and `binding_mismatch` (which profile was skipped and
  why, e.g. `param_profile_id_mismatch:fit=...,live=...`); the trace-quality
  payload mirrors the first mismatch as
  `trace_quality.calibration_binding_mismatch` for reports and triage.
- **Market separation unchanged** — binding checks happen after market routing;
  game/prop/draw profiles stay distinct and P8.3 introduces no cross-market
  authorization. Prop-plane profiles produced by the P8.5 prop structural sweep
  bind via the ref `prop_neg_binom` echoes and are consumed end-to-end (tested).

Enforcement lives at *application* time (the registry/selection seam), not as a
new promotion gate: promotion-time cannot know the future live substrate, and the
application check also protects every already-promoted profile the moment the
substrate changes underneath it.

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

## Operator runbook — diagnose → tune → gate

A worked example on the real **FIFA_INTL** soccer bucket (3,750 graded replay
traces), the diagnostic-picked target.

1. **Triage** — classify the bucket's RAW calibration so structure is tuned only
   where the residual is real:
    ```bash
   omega-cv-calibration-diagnostic --league FIFA_INTL --plane game \
     --historical-only --historical-db var/historical/replay_fifa_intl.db
    ```
   Result: raw ECE **0.073**, `mean_pred 0.395` vs `base_rate 0.463` →
   `MISCALIBRATED`, and the mean gap shows a **bias** (home wins under-predicted)
   because the replay used `home_advantage=0.0`. So `home_advantage` is the lever.
   The draw plane is already `CALIBRATED` (0.039) — leave it alone.
2. **Tune** the knob to minimize raw out-of-sample ECE (dry-run first; add
   `--register` to persist the CANDIDATE):
    ```bash
   omega-fit-backend-structure --backend soccer_bivariate_poisson_dc \
     --league FIFA_INTL --knob home_advantage --base-params '{"rho": -0.012705}' \
     --priors-as-of 2026-06-10 --historical-only \
     --historical-db var/historical/replay_fifa_intl.db \
     --validation-start 2018-01-01 --holdout-start 2024-06-01
    ```
   Winner `home_advantage=0.3`: validation raw ECE **0.018**, sealed-holdout raw
   ECE **0.032**, raw **CV-ECE 0.041 — CLEARS the 0.05 floor** (down from the
   baseline 0.070). The grid also shows over-correction (0.45 → 0.022, 0.6 →
   0.048), confirming 0.3 is a genuine optimum, not a monotonic artifact.
3. **Gate** (fail-closed; there is no `--force`):
   ```bash
   omega-promote-parameter-profile --profile-id <winner>          # dry-run: show gates
   omega-promote-parameter-profile --profile-id <winner> --auto \
     --confirm-backtest-parity --parity-report parity.json \
     --confirm-clv-non-regression --clv-report clv.json
   ```
   The candidate passes SAMPLE_SIZE + **ECE_FLOOR (on raw CV-ECE)** + the
   no-incumbent improvement gates; promotion still requires operator
   backtest-parity + CLV evidence artifacts — the structural fix is proven on
   calibration grounds, awaiting that evidence (exactly the fail-closed contract).
4. **Calibrate the residual** bound to the promoted backend version (P8.3,
   now automatic): re-run `omega-fit-calibration` on traces produced by the
   promoted params — the candidate records that substrate as its
   `backend_binding`, and runtime selection will only apply it there. With the
   raw output already under the floor, the calibration map is gentle or
   `method=none`.

## Deferred

- ~~P8.3~~ — **done** (see "P8.3 — calibration/backend binding, closed" above).
  Follow-ups that stay open:
  - `FrozenArtifact` does not carry `prior_payload`, so the strategy backtest
    re-simulates ungoverned and honestly reports no `param_profile_id` — bound
    profiles are (correctly) skipped there. Pinning artifact-level params
    (the lab's `parameter_pin_status` machinery is the seam) would let
    backtests replay the governed substrate.
  - Sport-family / GLOBAL bucket fits span leagues and will usually mix
    substrates → they refuse to fit under P8.3 and stay legacy until someone
    decides what a multi-substrate bucket binding should mean.
- **P8.4** — tennis (`priors_tennis*` status + pressure/SPW structure profile) and
  NFL (`nfl_nb_v2`: score correlation + team/context dispersion) adopt the rail.
- Fold soccer `rho` into the parameter profile (it stays on the live Dixon-Coles
  path for mid-tournament safety). The dedicated convenience CLI is **done** as of
  P8.5 — `omega-fit-backend-structure`, backend-generic.
- ~~Prop-plane structural sweep~~ — **done**. `sweep_prop_backend_variants`
  (same selection/seal skeleton as the game sweep, prop-backend seam, shared
  `prop_pairs_for_trace` grading, exact NB CDF) + `omega-fit-backend-structure
  --plane prop --stat <stat>` tune `nb_k_scale` against raw prop ECE and register
  CANDIDATEs bucketed per (league, stat) via `resolve_prop_calibration_bucket`
  (e.g. `prop_neg_binom` / `NFL__RUSHING_YARDS`).
- ~~Prop-plane consumption path~~ — **done**. `_merge_prop_parameter_profile`
  (the prop analogue of `_merge_parameter_profile`) runs inside
  `inject_prop_priors` at the gatherer layer: it merges the PRODUCTION profile's
  knobs + `parameter_profile_ref` into `player_context` (never overwriting
  caller-supplied values; skipped under `OMEGA_REPLAY_MODE=1` or an embedded
  ref), `analyze_player_prop` forwards `nb_k_scale`/`parameter_profile_ref` into
  the backend's `prior_payload`, and `prop_neg_binom` echoes the ref onto its
  result (V20 trace column) **and** the applied `nb_k_scale` into
  `distribution_params` — persisted `k` is post-scale once a profile is live, so
  `prop_trace_to_frozen_artifact` divides the echoed scale back out to keep the
  frozen base `k` pre-scale for future sweeps. Bit-identical while no prop
  profile is promoted.

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
