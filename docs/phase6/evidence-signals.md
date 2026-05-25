# Phase 6i — Structured Evidence Loop

**Status:** implemented · **Schema:** trace DB V9 · **Default rollout:** shadow mode

## Problem

The LLM agent reasons richly — defensive matchup, recent form, elimination-game
usage spikes, outlier regression — but historically that reasoning was silently
discarded. The agent wrote free-text notes and ad-hoc keys into the open
`player_context` / `game_context` dicts; the deterministic engine reads only
`{stat}_mean` / `{stat}_std` (props) or `off_rating` / `def_rating` / `pace`
(games) and ignored everything else. Qualitative intelligence collapsed to an
invisible, unattributable, unbacktestable bump in a single mean, and there was
no way to ask "was that reasoning signal predictive?"

## Design

A three-phase loop, multi-sport across all nine engine archetypes.

### Phase A — Capture (`omega/core/contracts/evidence.py`)

`EvidenceSignal` is a typed Pydantic model: `signal_type`, `category`, `plane`
(player/game), `value`, `source`, `confidence`, `window`, `direction`,
`stat_key`, `note`. A sport-tagged `SIGNAL_REGISTRY` declares which archetypes
each signal type applies to. Both `PlayerPropRequest` and `GameAnalysisRequest`
gained an `evidence: list[EvidenceSignal]` field (`default_factory=list` — zero
behavior change for existing callers). The validator **warns, never raises** on
unknown signal types, so a newer agent is always forward-compatible.

Evidence is part of `_stable_input_hash()` (different evidence → different
simulation → different `trace_id`). On persist, `TraceStore` explodes
`input_snapshot.evidence` into the queryable `evidence_signals` table (V9).

### Phase B — Apply (`omega/core/simulation/evidence_handlers.py`)

The LLM supplies evidence *values*; the engine decides *how* to apply them,
deterministically. Each `signal_type` maps to a pure handler that produces a
`(target, factor)` pair from coefficients in a versioned `AdjustmentPolicy`
(`omega/core/calibration/adjustment_policy.py`, stored in
`adjustment_policies.json`). Every factor is capped to `1 ± coeffs['cap']`.

`_apply_game_context()` was **extended** (not forked) to consume player-plane
signals; `_apply_game_evidence()` is the game-plane twin that scales team
`off_rating`. Both share one handler registry and one policy artifact.

Markov game analysis is backend-specific: it consumes mapped evidence through
`transition_modifiers` and does not also apply handler-based `off_rating`
scaling. When matching game-plane and player-plane signals appear in one
request, game-plane execution wins and the player-plane duplicate is persisted
as `applied=false` with `suppressed_by_game_plane_dedup`.

**Shadow mode** (the default) computes and records every adjustment in the trace
but does not apply it to the live prediction. The seed policy `adj_v1_seed`
transcribes the legacy hardcoded factors and is shadow — so Phases A and B are
behavior-neutral until a policy is promoted to `mode='live'`.

### Phase C — Score and re-fit

`omega/strategy/signal_performance.py` + `scripts/score_evidence_signals.py`
JOIN `evidence_signals` to attached outcomes and measure, per
`(signal_type, source, obs_window, league)`, whether each signal's `direction`
matched the realized result and whether its `confidence` matched its empirical
accuracy. Aggregates land in the `signal_performance` table (V9) and surface in
`report_calibration.py` §6B for the agent at session start.

`scripts/fit_adjustment_policy.py` reads `signal_performance` and derives a
`reliability_weight ∈ [0,1]` per signal type:

```
reliability_weight = clamp( 2 * (weighted_accuracy - 0.5), 0, 1 )
```

The evidence evaluator damps every handler's deviation by this weight:
`effective_factor = 1 + reliability_weight * (raw_factor - 1)`. A signal type
that scored at chance is damped to a no-op; one that proved predictive keeps its
strength. This is the single deterministic seam closing Phase C back into
Phase B. The re-fit is written as a **CANDIDATE** policy;
`scripts/promote_adjustment_policy.py` performs a gated, operator-driven
promotion (and optional `--go-live` flip).

## Invariants

- Same request + same policy → same `trace_id` and same adjustments (handlers
  are pure: no RNG, no clock, no I/O).
- Engine ignores any `signal_type` it has no handler for.
- Every adjustment is capped, recorded with its `policy_version` and
  `evidence_mode`, and persisted for retrospective scoring even in shadow mode.
- Going live is one explicit step: promote a policy and set `mode='live'`.

## Rollback

- Phase A — `evidence` defaults to empty; ignoring the field is a no-op.
- V9 — additive tables; drop them and the `schema_versions` row to revert.
- Phase B — shadow is the default; live rollback is one registry edit
  (`set_mode(..., 'shadow')`) or re-promoting `adj_v1_seed`, which equals
  today's engine behavior exactly.
- Phase C — scoring scripts are read-only on predictions.

## Operational sequence

```
analyze() with evidence            # capture + shadow-apply, persist V9 rows
scripts/fetch_outcomes_*.py        # attach outcomes
scripts/score_evidence_signals.py  # populate signal_performance
scripts/report_calibration.py      # §6B surfaces signal performance to the agent
scripts/fit_adjustment_policy.py   # CANDIDATE policy from measured accuracy
scripts/promote_adjustment_policy.py --confirm-backtest --go-live   # gated live flip
```
