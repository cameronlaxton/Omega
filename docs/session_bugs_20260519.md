# Session Bug Report — 2026-05-19

Session: outcome grading run for all 15 pending bet_records (game date 2026-05-17)

---

## BUG-1: `bet_records` schema missing `session_id` column

**Severity:** Medium  
**Table:** `bet_records`  
**Symptom:** `SELECT session_id FROM bet_records` raises `no such column: session_id`. The `traces` table has `session_id`; `bet_records` does not.  
**Impact:** Any query that tries to join or filter bet_records by session cannot use this column. Reporting and session-scoped audits must go through `traces` via `trace_id` join.  
**Fix:** Add migration: `ALTER TABLE bet_records ADD COLUMN session_id TEXT`. Backfill from `traces.session_id` via `trace_id`. Add to V6+ schema definition.

---

## BUG-2: `fetch_outcomes_props.py` attaches prop outcomes to analysis traces, not bet-record traces

**Severity:** High — grading pipeline is broken for all player props  
**Root cause:** `store.query_traces()` returns traces from the `traces` table. When a bet is confirmed, a new (second) trace is minted with a different `trace_id`. The prop fetch script finds the first/analysis trace; the bet_record points to the confirmation trace. The two trace_id sets are completely disjoint.

**Evidence:**
- 15 bet_records; 0 overlap with 12 prop_outcomes in the DB
- `fetch_outcomes_props.py --verbose` attaches outcomes to traces like `sandbox-2569e47d-dd94` while the bet_record for the same player uses `sandbox-a93f3a51-83d2`

**Impact:** No player prop bet_record can be auto-graded via `prop_outcomes`. Manual/fuzzy fallback required.

**Fix options (pick one):**
1. Store the canonical trace_id in bet_record at confirmation time — only one trace per analysis, reuse the same `trace_id` when converting to a bet.
2. Add a `bet_trace_id` foreign key to `prop_outcomes` that references `bet_records.trace_id` when a bet is known; grade by bet `trace_id` when present, fall back to analysis `trace_id`.
3. Teach `fetch_outcomes_props.py` to also sweep `bet_records` for ungraded prop bets, resolve player/stat from the `traces.full_trace`, and call `attach_prop_outcome` with the bet's `trace_id`.

Option 3 is the lowest-friction fix given the current architecture.

---

## BUG-3: Game outcomes (1-0 binary scores) attached to player prop traces

**Severity:** Medium  
**Source:** `manual:espn_boxscore_20260518`  
**Symptom:** All 15 bet_record traces have entries in the `outcomes` table with `home_score=1, away_score=0` or `away_score=1`. These are not real game scores — they appear to be binary win/loss placeholders.  
**Impact:** `outcomes` join on prop bet traces surfaces meaningless data. Any code that reads `home_score`/`away_score` from the outcomes table for prop bets gets garbage.  
**Fix:** The manual outcome ingestion script should not write game outcomes to prop traces. Either: filter by `traces.kind != 'prop'` before attaching game outcomes, or write a cleanup migration to remove outcomes with `home_score IN (0,1) AND away_score IN (0,1)` where the linked trace is a prop kind.

---

## BUG-4: Bet-record confirmation traces missing game identity fields

**Severity:** High — cascading cause of BUG-2  
**Symptom:** `traces.full_trace → input_snapshot` for all bet_record traces has `home_team=None, away_team=None, game_date=None`. The matchup field is set to the prop label (e.g., `"Donovan Mitchell pts 25.5"`) not the actual game.  
**Impact:** `fetch_outcomes_props.py` skips these traces entirely (logs as `missing game_date/home/away`). They can never be auto-resolved.  
**Fix:** When minting a bet-confirmation trace, copy `home_team`, `away_team`, `game_date` from the source analysis trace's `input_snapshot` into the confirmation trace's `input_snapshot`.

---

## BUG-5: Line discrepancy between trace and bet_record for Donovan Mitchell

**Severity:** Low-medium — data integrity  
**Bet:** `ecb73065` Donovan Mitchell Under pts  
**Trace input_snapshot line:** 25.5 (matchup: "Donovan Mitchell pts 25.5")  
**bet_record.line_taken:** 26.5  
**Impact:** Fuzzy grading against prop_outcomes grades this as WIN (26.0 < 26.5) correctly, but if graded from the trace line (25.5) it would be LOSS (26.0 > 25.5). The correct result is WIN (line_taken=26.5 is authoritative — line moved between analysis and bet placement).  
**Fix:** Ensure bet_record.line_taken is always set from the confirmed odds at time of bet, not from the analysis trace. Add a consistency check at ingest time: if `|line_taken - trace_line| > 1.0`, log a warning.

---

## Grading workaround used this session

Since BUGs 2–4 prevent automated prop grading, stats were resolved manually:
- NBA (CLE @ DET, 2026-05-17, event_id 401871339): fetched via `fetch_box_score("NBA", "401871339")`
- MLB Peter Lambert (TEX @ HOU, event_id 401815385): fetched directly
- MLB Colin Rea (CHC @ CWS, event_id 401815382): fetched directly
- Parlays graded leg-by-leg from the above actuals

All 15 bets are now fully graded. See session results below.

---

## Session Results Summary (2026-05-17 bets, graded 2026-05-19)

| # | Player | Type | Line | Side | Odds | Units | Actual | Grade | P&L |
|---|--------|------|------|------|------|-------|--------|-------|-----|
| 1 | Evan Mobley | pts | 18.5 | over | -110 | 1.0 | 21.0 | WIN | +0.91 |
| 2 | James Harden | ast | 6.5 | over | -130 | 1.0 | 6.0 | LOSS | -1.00 |
| 3 | Cade Cunningham | pts | 27.5 | under | -105 | 1.0 | 13.0 | WIN | +0.95 |
| 4 | Cade Cunningham | 3pm | 2.5 | over | -120 | 1.0 | 0.0 | LOSS | -1.00 |
| 5 | Donovan Mitchell | pts | 26.5 | under | -130 | 15.0 | 26.0 | WIN | +11.54 |
| 6 | Jalen Duren | reb | 8.5 | over | -130 | 15.0 | 9.0 | WIN | +11.54 |
| 7 | Max Strus | reb | 4.5 | over | -115 | 15.0 | 3.0 | LOSS | -15.00 |
| 8 | Paul Reed Jr | pts | 7.5 | over | -140 | 15.0 | 4.0 | LOSS | -15.00 |
| 9 | Peter Lambert | strikeouts | 4.5 | over | -145 | 12.5 | 6.0 | WIN | +8.62 |
| 10 | Colin Rea | strikeouts | 4.5 | under | -125 | 12.5 | 4.0 | WIN | +10.00 |
| 11 | Evan Mobley | reb | 8.5 | over | +100 | 15.0 | 12.0 | WIN | +15.00 |
| 12 | Daniss Jenkins | 3pm | 2.5 | over | +260 | 25.0 | 2.0 | LOSS | -25.00 |
| 13 | Duncan Robinson | 3pm | 2.5 | over | +140 | 20.0 | 3.0 | WIN | +28.00 |
| 14 | Parlay1 (Reed/Duren/Strus) | parlay | — | — | +488 | 20.17 | — | LOSS | -20.17 |
| 15 | Parlay2 (Jenkins/Allen/Mobley) | parlay | — | — | +452 | 15.0 | — | LOSS | -15.00 |

**Record:** 8W – 7L  
**Units staked:** 184.17u  
**Net P&L:** -5.61u  
**ROI:** -3.0%
