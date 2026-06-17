# MLB Outcome Grading Operator

Grade all pending MLB bets against final game results, settle the dollar ledger, and produce a verified portfolio summary. Run this after games are final (MLB: after ~2am Eastern).

This prompt is self-contained. Execute each section in order. Do **not** close the session until the Output Validation Gate (Section 6) passes with all items ✓.

---

## Section 0 — Preflight: Run the Outcome Loop

Run the outcome loop as a **single blocking command**. Do NOT call `omega-fetch-outcomes-mlb`, `omega-settle`, or `omega-ingest-traces` individually in sequence. Do NOT poll, sleep, or loop between commands.

```bash
omega-run-action-plan fixtures/action_plans/daily_outcome_evidence_loop.json
```

This is synchronous. It returns only after all 5 steps complete:
1. `omega-ingest-traces` — drains `var/inbox/traces/*.json` into the DB
2. `omega-fetch-outcomes` (mlb + props) — attaches final scores and box-score stats
3. `omega-settle` — settles pending user-confirmed ledger rows with attached outcomes
4. `omega-score-evidence-signals` — retrospectively scores signal predictiveness
5. `omega-report-calibration --league MLB --window-days 30` — writes `var/reports/latest.md`

**If the action plan exits non-zero:** read `stderr` directly and report the first error line verbatim. Do not retry without diagnosing the failure.

**If grading a custom date window (not yesterday)** — use the explicit override instead:

```bash
omega-fetch-outcomes-mlb --since YYYY-MM-DD --until YYYY-MM-DD
omega-fetch-outcomes-props --league MLB --since YYYY-MM-DD --until YYYY-MM-DD
python -m omega.ops.settle_bets --apply --league MLB --provenance all
```

After this section completes, append a `command` audit event to the sidecar documenting which path was taken and its exit status.

---

## Section 1 — Pending Bet Inventory

**Before the outcome loop** (or to audit scope afterward), query the pending ledger:

```bash
# Dry-run shows pending rows without writing
python -m omega.ops.settle_bets --league MLB --provenance all --dry-run
```

Or via MCP:
```python
omega_trace_query(league="MLB", status="pending")
```

Record the authoritative grading scope as a table:

| ledger_id | trace_id | market | selection | odds | stake | bet_date |
|-----------|----------|--------|-----------|------|-------|----------|
| ...       | ...      | ...    | ...       | ...  | ...   | ...      |

Do not expand or contract this scope mid-session. Any bet not in this table at session open is out of scope.

---

## Section 2 — Public API Audit Gate

Run this decision tree for **every bet** before any `bet_ledger` status update. Document the path taken in a `data_provenance` sidecar event for each bet.

```
For each pending bet:

  1. Does the bet have an attached outcome in the DB?
       YES → omega settle --apply handles it. STOP — no further action needed.
       NO  → continue

  2. Is the game cancelled / postponed / not yet played?
       YES → VOID candidate → go to step 3
       NO  → Outcome missing. Re-run fetch_outcomes (Section 0 override path).
              If still unmatched after re-run → flag as UNRESOLVED (not VOID). STOP.

  3. VOID settlement for DNP / no-action player props:
       a. Use the MCP tool:
            omega_trace_void_prop(
              trace_id="<trace_id>",
              player_name="<player>",
              stat_type="<prop_type>",
              reason="dnp"
            )
       b. Then run omega_settle_bets(apply=true, league="MLB") or `omega settle --apply`.
       c. Verify immediately with omega_get_portfolio_summary or `omega-db-status --view-ledger`.
          Expected: status='void', payout_amount=<stake>, net_pnl=0.0

       Do not perform direct `store.grade_ledger_bet()` writes for DNP/no-action prop voids.
```

---

## Section 3 — Grading Summary & PnL Breakdown

After `omega settle --apply` (or the action plan settle step) completes, call:

```python
omega_get_portfolio_summary(league="MLB", start="YYYY-MM-DD", end="YYYY-MM-DD")
```

Scope `start`/`end` to the grading run's date range. Do **not** compute PnL manually — use the values returned by this call.

Build the grading summary from the response:

```
Grading Summary
───────────────────────────────────────────────
Total graded:    N
  Wins:          N  │  Total PnL: +$X.XX
  Losses:        N  │  Total PnL: -$X.XX
  Voids:         N  │  Total PnL: $0.00  (stake returned)
  Pushes:        N  │  Total PnL: $0.00
───────────────────────────────────────────────
Net PnL:        +/-$X.XX
ROI:             X.X%
```

---

## Section 4 — Portfolio Summary (Schema Validation)

Use the `omega_get_portfolio_summary` response from Section 3. Before reporting any numbers, validate the schema.

**Required fields — all must be present and non-null:**
```
base_bankroll, current_bankroll, net_pnl, roi_pct, total_staked,
won, lost, push, void, win_pct, status_counts
```

**Validation (run inline, report result):**

For each required field: present? non-null? If any field fails → schema validation failure. Do not report portfolio numbers until resolved.

Append a `quality_gate` sidecar event:
```json
{
  "event_type": "quality_gate",
  "step": "portfolio_schema_validation",
  "status": "ok",
  "notes": "All 11 required fields present and non-null"
}
```

If validation fails, set `status: "fail"` and list missing fields in `notes`.

---

## Section 5 — Unresolved Items

Report ALL unresolved bets (no outcome, ambiguous match, VOID pending authorization) in this table. Never use narrative prose for unresolved items.

| trace_id | market | reason | missing_data | recommended_action |
|----------|--------|--------|--------------|-------------------|
| ...      | ...    | ...    | ...          | ...               |

**If zero unresolved items, write this line explicitly:**

> **Unresolved Items: None**

---

## Section 6 — Output Validation Gate (pre-close)

**Do not close the session until every item below is ✓.**

Print this table in chat with actual status for each item:

| # | Item | Status |
|---|------|--------|
| 1 | Preflight used `omega-run-action-plan` as single blocking command (no polling) | ✓ / ✗ |
| 2 | Portfolio summary validated against required schema (all 11 fields present/non-null) | ✓ / ✗ |
| 3 | Grading Summary includes per-result-type PnL (wins total, losses total, voids total) | ✓ / ✗ |
| 4 | All unresolved items in table format (or "None" explicit statement) | ✓ / ✗ |
| 5 | Verification section present in chat with commands, outputs, and warnings | ✓ / ✗ |
| 6 | All 7 output sections summarized in chat (not artifact-only) | ✓ / ✗ |
| 7 | Public API audit documented for any direct DB writes (or "no direct writes" explicit statement) | ✓ / ✗ |

If any item is ✗: do not close the session. Resolve the gap or document the blocker explicitly before proceeding.

Append a `step` sidecar event for the gate: `{"event_type": "step", "step": "output_validation_gate", "status": "ok"}` only after all items are ✓.

---

## Section 7 — Verification

Include this block verbatim in the chat response, filled in with actual values:

```
## Verification

Commands run:
  - omega-run-action-plan fixtures/action_plans/daily_outcome_evidence_loop.json
    → Exit: 0 / non-zero | stderr: [none | first error line]
  - omega_get_portfolio_summary(league="MLB", start=..., end=...)
    → Schema validation: pass / fail [missing: ...]

Checks executed:
  - Pending bet inventory: N bets in scope (see Section 1 table)
  - Public API audit: [all via omega settle / omega_trace_void_prop for DNP/no-action prop voids]
  - Portfolio schema: pass / fail
  - Unresolved items: N (see Section 5)

Warnings:
  - [none | list alias mismatches, missing data flags, or downgrade events]
```

---

## Decision Trees

### DT-1: Missing Pitcher / Roster Data

```
Is starter ERA available for this game?
  YES → include in game_context; proceed normally
  NO  →
    Is a probable starter named (stats absent)?
      YES → set starter_era=null; append null_data_audit sidecar event
            Does absence materially affect outcome confidence?
              YES → downgrade grading narrative; flag in Section 5 unresolved table
              NO  → proceed; note "starter_era unavailable" in audit event
      NO  → double-header / emergency start / TBD
             set starter_era=null
             if lineup ambiguous → flag outcome attachment as "low-confidence match"
```

### DT-2: Ambiguous Outcome Matches

```
Does omega-fetch-outcomes-mlb match this trace to a game?
  YES → proceed to settlement
  NO  →
    Is team name in the ESPN alias table (omega/integrations/espn_mlb.py::MLB_TEAMS)?
      YES → re-run with alias correction; if matched → proceed
      NO  →
        Is there exactly one MLB game on that date for that city?
          YES → manual match candidate; require user confirmation before attaching
          NO  → flag as UNRESOLVED in Section 5; do NOT guess or force-match
```

### DT-3: VOID Settlement

See Section 2 step 3. Summary:
- Use `omega_trace_void_prop` for DNP / no-action player-prop voids
- Run settlement after the void outcome is attached
- Direct `store.grade_ledger_bet()` writes are not the documented path for DNP/no-action prop voids
- Verify ledger state after settlement; document in Section 2 and Section 7

### DT-4: Stale Data Thresholds

| Input | Acceptable age | Fallback |
|-------|---------------|---------|
| ESPN scoreboard | Same-day final | Retry after 2am Eastern; flag unresolved if unavailable |
| ESPN box-score (props) | Same-day final | Same as above |
| Ledger pending rows | Any (archive) | No staleness concern |
| Calibration report (`var/reports/latest.md`) | ≤7 days | Run `omega-report-calibration`; note in sidecar if it fails |
| Odds at bet time | Captured at log time | Use `bet_ledger.odds` — do NOT re-fetch current market odds |

---

## Trade-Off Notes

**JSON schema validation cost:** ~150 tokens per run. Accept — overhead is negligible relative to audit value.

**Preflight scope:** `omega-cowork-preflight --formal-output-gate` is NOT required for grading-only sessions (no Bet Card authorization needed). If this session also produces new forward analysis, run the formal output gate explicitly before that analysis step and document it in the sidecar.

**Checklist ownership:** The Output Validation Gate table in Section 6 is externally provided by this prompt. The LLM fills in ✓/✗ per run — it does not regenerate the checklist schema.
