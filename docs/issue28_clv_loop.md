# Issue #28 — Continuous CLV evidence loop (WS5)

The evidence/context arm learns from **closing-line value (CLV)**: a signal earns
trust only if it carries information the market price did not already contain,
measured incrementally against the close on held-out data. This runbook ties the
pieces into one continuous, operator-gated loop.

Everything below is **fail-closed**: scoring and fitting only *recommend*; nothing
moves a live prediction until an operator promotes a policy. Probation signals and
LLM proposals are scored but never applied.

## The loop (one action plan)

The whole cadence is already composable through `omega-run-action-plan` — no
bespoke orchestrator. A canonical plan (`var/inbox/action_plans/<session>.json`):

```json
{
  "session_id": "clv-loop-2026-06-24",
  "actions": [
    {"type": "ingest_traces", "args": {}},
    {"type": "fetch_outcomes", "args": {"leagues": ["nba", "wnba", "mlb", "soccer", "nhl"]}},
    {"type": "fetch_closing_lines", "args": {"league": "nba"}},
    {"type": "score_evidence_signals", "args": {"league": "NBA", "window_days": 45}},
    {"type": "fit_adjustment_policy", "args": {"league": "NBA"}},
    {"type": "report_calibration", "args": {"league": "NBA"}}
  ]
}
```

Run it: `omega-run-action-plan var/inbox/action_plans/clv-loop-2026-06-24.json`
(add `--dry-run` to validate + print the commands first).

What each step contributes:

1. **ingest_traces** — load filed analysis traces (carrying `evidence` + the
   agent's `reasoning_narrative`, surfaced into the queryable `traces.llm_reasoning`
   column).
2. **fetch_outcomes** — attach realized results.
3. **fetch_closing_lines** — attach the market close (the CLV substrate). Thin
   coverage here is fine: scoring degrades gracefully to realized direction.
4. **score_evidence_signals** — score each signal's **CLV alignment** + signed CLV
   cents (plus the legacy direction accuracy as an audit column) into
   `signal_performance`.
5. **fit_adjustment_policy** — derive a CLV-primary `reliability_weight` per
   signal, apply the **Dual-Gate + FDR** bar (bootstrap/normal lower bound,
   power-derived `N_min`, Benjamini-Hochberg), and emit **lifecycle
   recommendations** (probation→active / →rejected, active→deprecated). Writes a
   CANDIDATE policy — never promotes.
6. **report_calibration** — the unified scorecard (§6B): per signal — lifecycle,
   n, CLV-alignment, CLV cents, `reliability`, verdict; plus recommended lifecycle
   transitions and any market-aware deference profiles. The agent vocabulary drops
   `deprecated` signals automatically.

## Operator-gated graduation (manual, by design)

The loop recommends; a human commits. These steps are intentionally **not** in the
automated plan:

- **Evidence policy + signal lifecycle:**
  `omega-promote-adjustment-policy --candidate-id <id> --auto --confirm-backtest
  [--apply-lifecycle-recommendations]`. The flag binds the fit's recommended
  transitions into the production policy's `signal_lifecycle` override map
  (validated, fail-closed). Add `--go-live` only to let evidence move predictions.
- **Calibration profiles** (including market-aware): `omega-promote-profile` runs
  the existing fail-closed gate (held-out improvement required).

## Agent proposals

`omega_propose_signal` (MCP) / `omega-propose-signal` register a typed hypothesis
over the whitelisted feature vocabulary into `signal_proposals` as `probation`.
Proposals are scored exactly like active signals; every proposal is another
hypothesis in the FDR correction, so an expanded proposal space only tightens the
graduation bar. Graduation injects the proposal's `feature_combo` into the policy
coefficients and flips its lifecycle — operator-gated, like everything else. The
injected coefficient also carries a fitted `reliability_weight` derived from the
proposal's scored `clv_aligned` (full trust when no CLV is yet on record), so a
graduated proposal moves predictions at its measured magnitude rather than the
policy's conservative `unfitted_reliability_prior` sliver.
