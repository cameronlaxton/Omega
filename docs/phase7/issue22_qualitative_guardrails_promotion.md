# Issue #22 — Qualitative Evidence Guardrails: Promotion Runbook

Status: implementation complete (Phases 1–6 merged on `issue-22-qualitative-guardrails`).
Production default: **all guardrail flags OFF** — the live engine is unchanged by the merge.

This runbook covers Phase 7: how to promote the guardrails safely, and the
checks that gate promotion. The deterministic engine remains authoritative; no
qualitative signal bypasses caps, calibration gates, or authorization gates.

## What ships behind flags

Three `AdjustmentPolicy` flags (all default `False`) plus their parameters:

| Flag | Effect when on | Params |
|------|----------------|--------|
| `enable_confidence_weighting` | Scales each signal's factor deviation by the agent's confidence (sequence step 6). | — |
| `enable_correlation_damping` | Damps co-occurring same-family signals instead of stacking them (steps 4–5). | `correlation_damping_weight` (0.5), `family_cap`, `plane_cap` (None = no clamp) |
| `enable_competition_strength_index` | Applies the structural soccer competition-strength index to team-context inputs before lambda derivation. | — |

## ⚠️ Live-effect surface (read before promoting)

The flags are **not uniformly evidence-mode gated**. Turning a flag on is a real
live change on these paths, regardless of the policy `mode` (`shadow`/`live`):

- **Markov game path** (`enable_confidence_weighting`, `enable_correlation_damping`):
  `service._game_evidence_plan_for` threads the production policy into
  `compute_transition_modifier_adjustment` and feeds `transition_modifiers` to
  the backend **unconditionally**. Flag on ⇒ Markov modifiers change immediately
  (e.g. WNBA, any `markov_state*` backend).
- **Soccer path** (`enable_competition_strength_index`): `service._competition_strength_index`
  is gated only on the flag + soccer archetype, then applied in
  `SoccerPoissonBackend` before lambda derivation. Flag on ⇒ soccer lambdas
  change immediately.
- **Handler / plane prop path** (`enable_confidence_weighting`, `enable_correlation_damping`):
  factors are *recorded* whenever the flag is on but only *applied* to the live
  prediction when `mode == "live"`. In `shadow` the trace shows the
  counterfactual without moving predictions.

Implication: promoting any flag to the production policy is a production-behavior
change on at least one path. It must be backtest-validated first.

## Promotion checks (the gate)

1. **Focused tests** — green across the per-phase suites and the consolidated gate:
   - `tests/core/test_confidence_weighting.py`, `test_correlation_damping.py`,
     `test_markov_richer_return.py`, `test_competition_strength_index.py`,
     `test_adjustment_policy_flags.py`, `tests/strategy/test_qualitative_feedback.py`
   - `tests/core/test_issue22_promotion_parity.py` (this gate): production flags
     off, flags-off replay parity to legacy output, the flag is the only lever,
     all-flags-on stays within caps.
2. **Replay parity** — `test_issue22_promotion_parity.py::TestReplayParityFlagsOff`
   pins that with flags off the engine output is bit-identical to legacy on the
   handler and Markov paths. Re-run on any change to the evidence pipeline.
3. **Calibration evidence** — promotion to a *live* effect additionally requires
   the repo's standard bar: improvement on held-out calibration quality with no
   material benchmark degradation (see `omega/core/calibration/CLAUDE.md`). Use
   the historical lab / backtest path on real outcomes; do not flip live blind.

## Recommended staged rollout

Each step is reversible and adds no schema migration.

1. **Enrich traces first (low risk).** Promote a candidate with
   `enable_confidence_weighting` + `enable_correlation_damping` on, **handler
   path in `shadow`**. Caveat: this still changes the Markov path live, so scope
   the first shadow promotion to leagues with no `markov_state*` backend, or
   accept the Markov change and validate it (step 3).
2. **Observe.** Run `omega-report-qualitative-feedback` (the Phase 6 gate) and
   `omega-score-evidence-signals` to confirm the enriched traces are sufficient
   and the signals are scoring sensibly.
3. **Backtest the live deltas.** For each path a flag touches (Markov modifiers,
   soccer lambdas, prop factors), run the historical backtest/lab on attached
   outcomes and confirm held-out calibration improves. Treat each flag
   independently; `competition_strength_index` needs soccer replay data with
   per-side strength inputs.
4. **Promote per flag.** Only after step 3 passes for a path, enable that flag
   in the production policy (and set `mode='live'` for the handler path) via the
   adjustment-policy registry workflow. Promote one flag at a time.

## How to create / promote a candidate

The registry workflow already exists (`AdjustmentPolicyRegistry`):

- Create a candidate by copying the seed policy and setting the desired flags
  (and `family_cap` / `plane_cap` / `correlation_damping_weight`), then
  `register()` it as `CANDIDATE`.
- Promote with the existing gated path (`promote()` archives the incumbent;
  fail-closed — no `--force`). Flip the handler path live with
  `set_mode(policy_id, "live")` as a separate, auditable step.

## Rollback

Revert the production policy to the seed (all flags off, `mode='shadow'`) via the
registry. No code revert or schema migration is required; the guardrail code
stays dormant behind the flags.
