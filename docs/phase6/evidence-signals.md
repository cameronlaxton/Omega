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
scripts/backfill_evidence_signals.py --dry-run   # find pre-V9 traces missing rows
scripts/backfill_evidence_signals.py --apply      # re-explode their own snapshots
scripts/score_evidence_signals.py  # populate signal_performance
scripts/report_calibration.py      # §6B surfaces signal performance to the agent
scripts/fit_adjustment_policy.py   # CANDIDATE policy from measured accuracy
scripts/promote_adjustment_policy.py --confirm-backtest --go-live   # gated live flip
```

## Eligibility, QA verdicts, and evidence backfill

These are addenda to the loop above and govern *which* traces calibration and
evidence learning may use, and how missing `evidence_signals` rows are repaired.

### Two separate eligibilities (single source of truth: `omega/trace/eligibility.py`)

- **Evidence is NOT required for probability calibration.** A trace with
  `evidence_status="empty"` is still probability-calibration eligible when its
  predictions, engine result, provided context, complete identity, and QA
  verdict are otherwise valid. Empty evidence is recorded for audit but is never
  a calibration exclusion reason.
- **Evidence IS required for evidence-signal learning.** Empty evidence blocks
  *only* the retrospective evidence-scoring path — never probability
  calibration. A trace can therefore be probability-gradeable while
  evidence-learning-ineligible.

`omega/trace/eligibility.py` is the one place these predicates live;
`service.py` (write side), `PersistableTrace.calibration_eligibility()`, and the
`TraceStore.query_traces` filter all defer to it. The canonical persisted gate is
`trace_quality.calibration_eligible` in the full_trace blob.

Status vocabularies — probability calibration: `eligible`, `pending_outcome`,
`ineligible_qa_failed`, `ineligible_missing_prediction`,
`ineligible_missing_outcome`, `ineligible_invalid_trace`,
`ineligible_trace_quality`. Evidence learning: `eligible_original`,
`eligible_recovered_predecision`, `ineligible_empty_evidence`,
`ineligible_unrecoverable`, `ineligible_qa_failed`, `ineligible_invalid_evidence`.

### Trace-scoped QA verdicts (schema V12: `trace_qa_verdicts`)

Quality-gate failures are scoped to a single trace
(`session_sidecar.quality_gate_verdict_for_trace`) using the per-event
`trace_ids` list, event timestamps vs. the trace's `ran_at`, and pre-trace setup
failures. A failed gate tied to one trace **no longer condemns the whole
session**. Verdict scopes: `trace_id`, `timestamp_window`, `pre_trace_fatal`,
`session_fallback` (conservative catch-all for legacy/unstructured sidecars),
`unrelated_session_failure`, `no_sidecar`.

- A valid trace artifact is **always** persisted to the ledger; only a
  malformed/invalid artifact is rejected.
- A QA-failed trace is **ledger-preserved but calibration-ineligible**
  (`trace_quality.calibration_eligible=False`, reason `qa_failed`), and the
  verdict is recorded in `trace_qa_verdicts`. It stays ineligible unless later
  explicitly revalidated; **no `--allow-audit-only-qa-failed` / deprecated
  `--force-ingest-qa-failed` flag can confer calibration eligibility.**

### Evidence backfill is RE-DERIVATION, not recovery

`scripts/backfill_evidence_signals.py` re-explodes `evidence_signals` rows from a
trace's own frozen **`input_snapshot.evidence`** — the only legitimate
pre-decision source. Evidence is never lost; pre-V9 traces simply carry it in
the blob with no table rows. Re-exploded rows are provenance `original` (same
source, materialized late) — there is no separate recovery source.

- Defaults to **dry-run**; `--apply` is required to write.
- Outcomes, box scores, closing lines, engine predictions, EV/edge/Kelly, and
  settlement results are **never** read — they cannot manufacture evidence.
- A trace whose snapshot evidence is genuinely empty is marked **unrecoverable**;
  no fake signals are invented.

### Scorer coverage summary

`scripts/score_evidence_signals.py` reports coverage by status instead of
silently skipping. Empty-evidence traces are reported as an evidence-learning
gap — never as a probability-calibration failure — and producing rows for the
available evidence is success, not breakage:

```
Evidence scoring summary
------------------------
Graded traces:                 N
Evidence-eligible (present):   N
Skipped: empty evidence:       N
Skipped: QA failed:            N
Signal-performance rows:       N
```
