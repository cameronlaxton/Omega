# Phase 6 Step 4 & 5 — End-to-End Smoke Test

This document walks the operator through a full session of the closing-line +
session-tracking + continuous-improvement loop introduced in Phase 6f. It is
manual on purpose: the bridge between the Claude Project sandbox and the local
DB is intentionally copy/paste (the sandbox cannot reach localhost).

## Prerequisites

- Local repo at HEAD of the Phase 6f changes (this branch).
- Python 3.11+ with the project deps installed.
- A Claude Project (claude.ai) with `omega_lite_standalone.py` and
  `prompts/system_prompt.txt` uploaded to project knowledge.
- Empty `omega_traces.db` (or fresh `--db <path>` for the smoke).

## 1. Sandbox: emit a trace with `session_id`

In a Claude Project chat, ask the agent to analyze any NBA game. Per
`system_prompt.txt` §12, it will mint a `session_id` of the form
`sess-YYYYMMDD-XXXX` and include it in the trace JSON.

Expected emission (§10 + §12.2):

```json
// SAVE AS: inbox/traces/sandbox-XXXX.json
{
  "trace": {
    "trace_id": "sandbox-XXXX",
    "session_id": "sess-20260515-a1b2",
    "ran_at": "...",
    "kind": "game",
    ...
  },
  "bet_record": { ... },
  "clv_capture_instructions": { ... }
}
```

Save the JSON to `inbox/traces/sandbox-XXXX.json`. Then:

```bash
python scripts/ingest_traces.py
```

Verify:

```bash
sqlite3 omega_traces.db "SELECT trace_id, session_id, league FROM traces;"
```

The `session_id` column must be populated.

## 2. Capture closing lines (paid Odds API)

Around T-30 minutes before tip-off, run the closing-line capture script locally:

```bash
python scripts/fetch_closing_lines.py --league NBA
# add --dry-run first to verify matches without writing
```

The script queries pending `bet_records`, calls the Odds API (BetMGM-first),
and writes closing rows directly to `omega_traces.db` via
`TraceStore.attach_closing_line()`. No inbox file is needed.

For missed windows, use the paid historical endpoint:

```python
from omega.integrations.odds_api import OddsApiClient
client = OddsApiClient()
snapshot = client.fetch_historical_odds(league="NBA", date="<ISO-8601 close timestamp>")
```

Verify:

```bash
sqlite3 omega_traces.db "SELECT trace_id, market, selection_descriptor, \
  closing_line, closing_odds, source FROM closing_lines;"
```

Re-running the capture is a no-op for rows already written
(idempotent on `UNIQUE(trace_id, market, selection_descriptor)`).

## 3. Attach outcome

After the game completes:

```bash
python scripts/fetch_outcomes_nba.py
```

This uses the existing ESPN integration (NOT deprecated) to attach scores.
Verify:

```bash
sqlite3 omega_traces.db "SELECT trace_id, home_score, away_score, result FROM outcomes;"
```

## 4. Sandbox: emit session sidecar

At the end of the session (or first message of the next), the agent emits per
§12.3:

```json
// SAVE AS: inbox/sessions/sess-20260515-a1b2.json
{
  "session_id": "sess-20260515-a1b2",
  "started_at": "...",
  "ended_at": "...",
  "model_version": "claude-opus-4-7",
  "agent_notes": "...",
  "exec_stats": {
    "traces_emitted": 1,
    "bets_recorded": 1,
    "webfetch_failures": 0
  }
}
```

Save to `inbox/sessions/sess-20260515-a1b2.json`. No script runs — the report
job reads the sidecar directly.

## 5. Run the report

```bash
python scripts/report_calibration.py --league NBA --window-days 30
cat reports/latest.md
```

Expected sections:

1. Coverage counts (traces, graded, with_bet, with_close).
2. Production calibration profile (likely "None" until step 6 promotes one).
3. Realized metrics (suppressed if < 10 graded traces).
4. CLV (joins bet_records ↔ closing_lines via compute_clv).
5. Sessions table (one row per session_id, joined with sidecar exec_stats).
6. Pending CANDIDATE profiles (empty until step 6).
7. Suggested actions (intentionally blank — LLM fills via §13).

## 6. Calibration fit (requires ≥100 graded NBA traces)

Once you have accumulated 100+ graded NBA traces (4–8 weeks of typical
usage per `HANDOFF_phase6d_to_h.md`):

```bash
python scripts/fit_calibration.py --league NBA --method both
```

Two CANDIDATE profiles register. Verify:

```bash
python scripts/promote_profile.py --list-candidates --league NBA
```

Inspect gate status without promoting:

```bash
python scripts/promote_profile.py --candidate-id <iso_profile_id>
```

Promote with full gates (operator must confirm gates 4 & 5 after manual review):

```bash
python scripts/promote_profile.py --candidate-id <iso_profile_id> --auto \
  --confirm-backtest-parity --confirm-clv-non-regression
```

## 7. Session-start brief loop

In a NEW Claude Project session, upload `reports/latest.md` to project
knowledge. The agent reads it per §13 and emits an action-plan block:

```json
// SAVE AS: inbox/action_plans/sess-XXXX.json
{
  "session_id": "sess-XXXX",
  "actions": [
    { "type": "fit_calibration", "args": {"league": "NBA", "method": "both"} },
    { "type": "report_calibration", "args": {"league": "NBA", "window_days": 30} }
  ]
}
```

Save and run:

```bash
python scripts/run_action_plan.py inbox/action_plans/sess-XXXX.json
```

The runner dispatches each allowed action. Disallowed types (e.g. an LLM-emitted
`inject_shrink_factor`) fail validation and the runner exits 2 without touching
anything — this is the CLAUDE.md boundary in code form.

## Determinism check

Re-run `fit_calibration.py` on the same DB snapshot twice. The dataset_hash and
metrics MUST match bit-for-bit (the train/holdout split is seeded by the
deterministic dataset hash + league). If they diverge, file an issue — that
violates a CLAUDE.md required invariant.

## What this does NOT cover

- Multi-league rotations (NFL, MLB, NHL) — add per-league as data accumulates.
- Player-prop CLV via the Odds API path — supported by schema (`market="player_prop:pts"`)
  but prop market coverage depends on the paid plan and bookmaker availability.
- Automated promotion gates 4 (backtest-replay parity) and 5 (CLV non-regression).
  These currently require operator confirmation via `--confirm-*` flags.
  Full automation is a follow-up commit once enough graded data accumulates to
  make the replay statistically meaningful.
