# Phase 6 Automation Playbook

Omega automation is repo-local and command-gated. Action plans may move existing
artifacts through deterministic ledger paths, but they must not create betting
recommendations, compute protected values, or promote adjustment policies live.

## Operator Loop

1. Analyze through `omega.core.contracts.service.analyze()` or MCP tools.
2. Export the full trace block to `inbox/traces/<trace_id>.json`.
3. If the user confirms they placed a wager, attach the real `bet_record` to
   that same trace export. A bet record is user-confirmed metadata only.
4. Run trace intake with `inbox/action_plans/templates/daily_trace_intake.json`.
5. Near close, run `bet_closing_line_capture.json` to capture closing lines for
   pending confirmed `bet_records`.
6. After games finish, run `daily_outcome_evidence_loop.json` to attach outcomes,
   score evidence signals, and refresh calibration health.
7. Weekly, run `weekly_shadow_review.json` to fit shadow-only adjustment-policy
   candidates when enough scored signal data exists.

## Template Semantics

- `daily_trace_intake`: runs `scripts/ingest_traces.py`; review failed files,
  missing identity warnings, empty evidence warnings, and bet-record counts.
- `bet_closing_line_capture`: runs `scripts/fetch_closing_lines.py` for pending
  confirmed bet records. This is not a bet discovery or recommendation action.
- `daily_outcome_evidence_loop`: runs outcome attachment, evidence scoring, and
  calibration reporting against canonical trace data.
- `weekly_shadow_review`: scores evidence, reports calibration health, and fits
  a shadow-only adjustment-policy candidate. It does not promote live behavior.
- `empty_slate`: validates the scheduler path on days with no work.

Run every template through the dispatcher before executing it:

```bash
python scripts/run_action_plan.py inbox/action_plans/templates/daily_trace_intake.json --dry-run
python scripts/run_action_plan.py inbox/action_plans/templates/daily_trace_intake.json
```

## Expected Summary Counts

Operators should record or review these counts after each loop:

- Trace intake: files processed, files failed, traces persisted, empty-evidence
  warnings, missing identity warnings, bet records recorded.
- Bet follow-up: pending bet records found, closing lines attached, records
  skipped, missing API key or budget failures.
- Outcomes: game outcomes attached, prop outcomes attached, unmatched traces,
  skipped traces, unresolved team/player aliases.
- Evidence scoring: graded traces loaded, evidence-bearing traces, scoreable
  signals, rows written or dry-run rows computed.
- Candidate fitting: league, mode, min samples, signal types fitted, candidate
  id, or explicit reason no candidate was fit.

## Guardrails

- `promote_adjustment_policy --go-live` is never scheduled.
- `fit_adjustment_policy` action plans only allow `mode: "shadow"`.
- `fetch_closing_lines` follows up confirmed `bet_records`; it never fetches
  candidate bets or creates picks.
- Calibration promotion remains an explicit operator action through
  `promote_profile`.
- `adj_v1_seed` remains in shadow unless a separate validation and promotion
  decision changes it.
