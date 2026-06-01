# Session Bug Report — 2026-05-22

Session: sess-20260521-c001 — startup outcome backfill and calibration health

---

## BUG-SS-1: All historical session sidecars fail SessionSidecar schema validation

**Severity:** High — `report_calibration.py` skips all 6 prior sidecars; sessions table shows `?` for model/closes/webfetch_fail across the board

**Affected files:**
- `var/inbox/sessions/sess-20260515-g7d1.json`
- `var/inbox/sessions/sess-20260518-mlb1.json`
- `var/inbox/sessions/sess-20260518-wcf1.json`
- `var/inbox/sessions/sess-20260519-nba1.json`
- `var/inbox/sessions/sess-20260519-nyk1.json`
- `var/inbox/sessions/sess-20260520-g001.json`

**Extra keys present (not in schema):** `date`, `traces`, `bets_taken`, `game_result`, `notes`, `started_at`, `ended_at`, `created_at`

**Missing required keys:** `opened_at`, `model_version`, `exec_stats`, `agent_notes`, `purpose`, `bankroll`, `bankroll_confirmed`

**Root cause:** The SessionSidecar Pydantic model was updated (added required fields, renamed `started_at`→`opened_at`, `ended_at`→`closed_at`, strict mode blocking extra fields) but prior session writes used the old schema. Agent instruction (OMEGA_COWORK.md §8) now specifies the correct schema — future sessions will be compliant. Historical sidecars need a one-time migration.

**Fix:** Write a migration script that maps old keys to new keys and sets sensible defaults for missing fields. Do NOT delete old sidecars — read, migrate, and overwrite.

---

## BUG-DRY-1: `fetch_outcomes_nba.py --dry-run` double-reports traces as both DRY-attached and unmatched

**Severity:** Low — dry-run output is misleading but live runs are unaffected

**Reproduction:**
```bash
omega-fetch-outcomes-nba --since 2026-05-18 --until 2026-05-20 --dry-run --verbose
```
Output: 8 `DRY` attach lines, then `unmatched=5` listing 5 of the same trace_ids.

**Root cause (hypothesis):** The unmatched list is built from a pre-query of all ungraded traces, then dry-run simulates attaches but doesn't remove matched traces from the unmatched set before printing it.

**Fix:** In dry-run mode, track which trace_ids are successfully matched and exclude them from the unmatched warning.

---

## BUG-PROP-1: `fetch_outcomes_props.py` emits duplicate `missing_fields` warnings per trace

**Severity:** Low — noise only, no data loss

**Reproduction:**
```bash
omega-fetch-outcomes-props --since 2026-05-17 --until 2026-05-20 --verbose
```
Output: `sandbox-667e891d-7aed` listed 3× in missing_fields warnings; `sandbox-fe2718ac-28d4` listed 3×.

**Root cause (hypothesis):** The missing-fields accumulator appends one entry per bet_record referencing the trace, rather than de-duplicating by trace_id before reporting.

**Fix:** Deduplicate the missing_fields warning list by trace_id before printing.

---

## BUG-PROP-2: `first_basket` prop type misclassified as `missing game_date/home/away`

**Severity:** Low — misleading warning message

**Trace:** `sandbox-667e891d-7aed` (KAT first_basket, game_date=2026-05-19, home=New York Knicks, away=Cleveland Cavaliers)

**Symptom:** Script reports `missing game_date/home/away` but the trace has all three fields populated. The actual problem is that `first_basket` is not a supported prop_type for ESPN box-score grading.

**Fix:** Add prop_type check before the game_identity check. If prop_type is unsupported, log as `unsupported_prop_type` (already has its own counter) rather than `missing_fields`.

---

## BUG-OUTCOME-1: `sandbox-fe2718ac-28d4` (Jalen Duren reb) permanently unresolvable

**Severity:** Medium — this trace will never auto-grade

**Root cause:** Pre-Phase-6b trace emitted without `game_date`, `home_team`, `away_team` in input_snapshot. Violates the mandatory `game_context` rule now enforced in OMEGA_COWORK.md §6b.

**Bet record:** `bet_id=c1d0119d` (Jalen Duren Over 8.5 reb, status=won, 5/17)

**Resolution:** Manual outcome attach required. Grade: Jalen Duren had 15 pts / ~10 reb in CLE@NYK G5 2026-05-17 — confirm reb line and attach manually via direct DB write or a repair script.

**Prevention:** `game_context` with `home_team`/`away_team`/`game_date` is now a hard requirement for all analyze() calls per §6b.

