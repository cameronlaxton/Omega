# Phase 6 Automation Playbook

Omega automation is repo-local and command-gated. Action plans may move existing
artifacts through deterministic ledger paths, but they must not create betting
recommendations, compute protected values, or promote adjustment policies live.

## Operator Loop

1. Analyze through `omega.core.contracts.service.analyze()` or MCP tools.
2. Export the full trace block to `var/inbox/traces/<trace_id>.json`.
3. If the user confirms they placed a wager, attach the real `bet_record` to
   that same trace export. A bet record is user-confirmed metadata only.
4. Run trace intake with `fixtures/action_plans/daily_trace_intake.json`.
5. Near close, run `bet_closing_line_capture.json` to capture closing lines for
   pending confirmed `bet_records`.
6. After games finish, run `daily_outcome_evidence_loop.json` to attach outcomes,
   score evidence signals, and refresh calibration health.
7. Weekly, run `weekly_shadow_review.json` to fit shadow-only adjustment-policy
   candidates when enough scored signal data exists.

## Template Semantics

- `daily_trace_intake`: runs `omega-ingest-traces`, then renders the derived
  Daily Intake Overview + Trace Ledger under `var/reports/session_reports/`;
  review failed files, missing identity warnings, empty evidence warnings, and
  bet-record counts.
- `bet_closing_line_capture`: runs `omega-fetch-closing-lines` for pending
  confirmed bet records. This is not a bet discovery or recommendation action.
- `daily_outcome_evidence_loop`: runs outcome attachment, ledger settlement,
  evidence scoring, and calibration reporting against canonical trace data.
- `weekly_shadow_review`: scores evidence, reports calibration health, and fits
  a shadow-only adjustment-policy candidate. It does not promote live behavior.
- `empty_slate`: validates the scheduler path on days with no work.
- `full_lifecycle_maintenance`: chains intake, close/outcome follow-up, audit
  rendering, and `omega-validate-all --skip-tests` without creating picks or
  promoting live behavior.

Run every template through the dispatcher before executing it:

```bash
omega-run-action-plan fixtures/action_plans/daily_trace_intake.json --dry-run
omega-run-action-plan fixtures/action_plans/daily_trace_intake.json
```

## Expected Summary Counts

Operators should record or review these counts after each loop:

- Trace intake: files processed, files failed, traces persisted, empty-evidence
  warnings, missing identity warnings, bet records recorded, and the derived
  intake report path.
- Bet follow-up: pending bet records found, closing lines attached, records
  skipped, missing API key or budget failures.
- Outcomes and settlement: game outcomes attached, prop outcomes attached,
  ledger rows settled, unmatched traces, skipped traces, unresolved team/player
  aliases.
- Evidence scoring: graded traces loaded, evidence-bearing traces, scoreable
  signals, rows written or dry-run rows computed.
- Candidate fitting: league, mode, min samples, signal types fitted, candidate
  id, or explicit reason no candidate was fit.

## Guardrails

- `promote_adjustment_policy --go-live` is never scheduled.
- `fit_adjustment_policy` action plans only allow `mode: "shadow"`.
- `fetch_closing_lines` follows up confirmed `bet_records`; it never fetches
  candidate bets or creates picks.
- Non-dry-run `omega-run-action-plan` runs the runtime DB guard after plan
  validation and before dispatch. Scheduled tasks must run from
  `%USERPROFILE%\.omega\workspace\Omega` or set explicit durable
  `OMEGA_TRACE_DB`; auto-redirected, divergent, missing, corrupt, or empty DBs
  fail closed unless `OMEGA_ALLOW_EMPTY_DB=1` is intentionally set.
- Dry-runs still validate every action and report DB safety warnings, but they
  do not raise for unsafe DB identity.
- `render_report` is a derived-artifact action. It is non-fatal by default in
  action plans and must not compute betting recommendations or protected values.
- Calibration promotion remains an explicit operator action through
  `promote_profile`.
- `adj_v1_seed` remains in shadow unless a separate validation and promotion
  decision changes it.
