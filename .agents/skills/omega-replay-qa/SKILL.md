---
name: omega-replay-qa
description: Audit Omega trace persistence, replay bundles, calibration dry-runs, and LLM interface quality. Use when reviewing replay reproducibility, trace completeness, seed determinism, no-live-fetch guarantees, prompt/tool boundary compliance, or Phase 6 quant-plane versus replay-plane separation.
---

# Omega Replay QA

## Audit Posture

Always distinguish current repo truth, proposed changes, risks, verification,
and rollback. Treat replay as sampled agent-quality audit, not the default
quant benchmark path.

## Replay Checks

1. Confirm replay uses `ReplayBundle`.
2. Confirm live evidence gathering is disabled.
3. Confirm `decision_date` or `simulation_seed` controls replay seed when
   historical determinism matters.
4. Confirm facts contain only knowable-at-the-time evidence.
5. Confirm expected outputs measure routing, downgrades, trace completeness,
   refusal discipline, and evidence selection.

## Trace Checks

1. Persist initial trace before outcome attachment.
2. Attach outcomes through `TraceStore.attach_outcome`.
3. Do not mutate original trace records ad hoc.
4. Verify trace rows carry schema version, prompt, execution mode, seed, odds
   snapshot, predictions, recommendations, and downgrades where applicable.
5. Prefer `omega_trace_query` and `omega_trace_get` for inspection.

## Calibration Checks

1. Use `omega_calibration_fit_preview` for dry-run review only.
2. Confirm sample size, dataset hash, profile method, metrics, and league.
3. Do not promote a candidate unless an explicit promotion workflow with
   guardrail tests exists.
4. Confirm backtest and production share calibration selection policy.

## Prompt And Tool QA

1. Ensure the LLM-facing prompt says MCP first.
2. Ensure standalone prop scripts are fallback only.
3. Ensure Standard Text is required when deterministic tools cannot run.
4. Ensure no prompt asks the LLM to invent probabilities, edge, EV, Kelly,
   staking, confidence tiers, or trace IDs.

## References

- Read `references/audit-checklist.md` for a compact review checklist and
  failure-mode catalog.
