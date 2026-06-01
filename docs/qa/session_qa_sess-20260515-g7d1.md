# Omega Session QA Log — `sess-20260515-g7d1`

**Purpose:** Developer QA review of Cowork chat interface quality for a live player-prop analysis session.  
**Date:** 2026-05-16  
**Model:** claude-sonnet-4-6  
**Game:** Cleveland Cavaliers @ Detroit Pistons — Game 7, ECF Semis, May 17 2026, 8:00 PM ET  
**Session ID:** `sess-20260515-g7d1`

---

## 1. Interaction Summary

| Step | User Input | Agent Action | Outcome |
|------|-----------|--------------|---------|
| 1 | "Give me best 4-6 player prop bets for Sunday's Pistons vs Cavaliers game" | Launched session-start protocol; attempted VM smoke test | VM failed to start (3 retries over ~60s) |
| 2 | "Continue from where you left off" (×2) | Retried VM, still unavailable | Entered QUALITATIVE MODE temporarily |
| 3 | "try VM again" | VM booted; installed missing deps (pydantic, numpy); smoke test passed | NUMERIC MODE confirmed: `sandbox-8688667a-15bc` |
| 4 | Agent | Ran calibration health report (0 traces) | Empty action plan, no-op |
| 5 | Agent | Researched injury report, series stats, Game 7 schedule, prop lines via WebSearch | 5 searches; confirmed Game 7 in Detroit, 8PM ET |
| 6 | Agent | Ran engine on 6 prop candidates | 3 skipped (wrong context field names); 3 succeeded |
| 7 | Agent | Fixed field names; re-ran 3 skipped props | All 3 resolved; 4 total Tier A recs surfaced |
| 8 | Agent | Ingested all 9 traces | 9/9 OK |
| 9 | "I took all 4 suggested bets" | Agent asked for book, line, odds, units | Correct — no bet metadata assumed |
| 10 | "theScore Bet" | Agent asked for units | Correct — still needed stake |
| 11 | "1 unit $25 each" | Agent wrote bet records; re-ingested 4 traces with bet_record blocks | 4/4 bet records logged |
| 12 | Agent | Attempted JIT CLV scheduling (4 tasks at 7:30 PM ET May 17) | Blocked: `create_scheduled_task` unavailable in unsupervised mode |
| 13 | Agent | Wrote session sidecar + QA log | This file |

---

## 2. Engine Trace Summary

| Trace ID | Player | Prop | Line | Odds | Status | Rec | Tier | Bet Recorded |
|----------|--------|------|------|------|--------|-----|------|--------------|
| `sandbox-80b0adf2-4013` | Cade Cunningham | pts | 27.5 | -105 (under) | success | under | A | ✅ |
| `sandbox-a93f3a51-83d2` | Cade Cunningham | threes | 2.5 | -120 (over) | success | over | A | ✅ |
| `sandbox-4309e3f1-d43f` | Donovan Mitchell | pts | 26.5 | -120 (over) | success | pass | — | ❌ |
| `sandbox-fe2718ac-28d4` | Jalen Duren | reb | 10.5 | -110 | success | pass | — | ❌ |
| `sandbox-58ee2ffa-63a4` | James Harden | ast | 6.5 | -130 (over) | success | over | A | ✅ |
| `sandbox-354c32cd-8778` | Evan Mobley | pts | 18.5 | -110 (over) | success | over | A | ✅ |
| `sandbox-2569e47d-dd94` | Cade Cunningham | threes | 2.5 | -120 | **skipped** (initial — wrong field name) | — | — | ❌ |
| `sandbox-a8cb09ef-828f` | James Harden | ast | 6.5 | -130 | **skipped** (initial — wrong field name) | — | — | ❌ |
| `sandbox-fe2718ac-28d4` | Jalen Duren | reb | 10.5 | -110 | **skipped** (initial — wrong field name) | — | — | ❌ |

**Bet records ingested:** `f183f7ba0c9b`, `01a8d9b54821`, `f1f0bd8baf4f`, `26f0b5f62fbd`

---

## 3. Interface Quality Findings

### 3.1 Hard-Wall Compliance ✅
The agent never emitted model probabilities, edge%, EV%, Kelly fractions, or confidence tiers as free text. All numeric outputs came exclusively from engine execution. The two engine `pass` results (Mitchell pts, Duren reb) were surfaced as non-bets with no fabricated lean attached. **Hard wall: fully enforced.**

### 3.2 Bet Metadata Discipline ✅
When the user said "I took all 4 bets," the agent correctly withheld logging until confirming: (1) sportsbook, (2) unit size. Two separate follow-up asks were needed because the user provided info incrementally. Behavior was correct per §5.1 protocol. No metadata was fabricated or assumed.

### 3.3 VM Startup Latency ⚠️ ISSUE
The VM failed to start on 3 consecutive attempts over approximately 90 seconds. The agent correctly fell back to QUALITATIVE MODE and communicated the status, but the user had to manually prompt "try VM again" to trigger each retry. **Improvement opportunity:** agent should retry automatically on a backoff schedule without requiring user prompts, up to a configured timeout.

### 3.4 Engine Field Name Mismatch ⚠️ ISSUE
When calling `r.analyze()` for non-pts props (threes, reb, ast), the agent passed `pts_mean`/`pts_std` as the player context keys instead of `threes_mean`, `reb_mean`, `ast_mean`. The engine correctly rejected these with a clear skip reason. The agent caught the failure, corrected the schema, and re-ran — but this cost one extra engine round-trip and produced 3 orphaned skip traces in the DB. **Improvement opportunity:** agent should inspect the engine's expected schema per prop_type before constructing the request dict, or maintain a prop_type→context_key mapping internally.

### 3.5 JIT CLV Scheduling ❌ BLOCKED
`create_scheduled_task` returned `"requires user interaction and is unavailable in unsupervised mode"` on all 3 attempts. The agent was not running in a supervised context at that point in the session (user had issued a "continue" prompt). The 4 JIT captures for Game 7 at 7:30 PM ET May 17 were not scheduled. **Manual action required — see §5 below.**

### 3.6 Research Quality ✅
Five WebSearch calls returned accurate, current data: series stats through Game 6, Game 7 schedule confirmation (Detroit, 8 PM ET, Prime Video), injury report (Robinson/LeVert/Huerter statuses), and prop line context. No stale or fabricated data used as engine input.

### 3.7 Session Protocol Compliance ✅
- Session ID minted once: `sess-20260515-g7d1`
- Calibration health run at session start
- Action plan emitted (`actions: []`, correct for 0-trace DB)
- Session sidecar written to `var/inbox/sessions/`
- All traces ingested via pipeline (not written to DB directly)

### 3.8 User-Facing Communication ✅
The agent correctly distinguished research-only context from engine outputs. Passing props (Mitchell, Duren) were labeled as engine passes with no fabricated lean. Priority ordering was provided (Cunningham 3s → Mobley pts → Cunningham pts under → Harden ast). Price drag on Harden (-130) was flagged explicitly.

---

## 4. Orphaned / Skipped Traces in DB

Three traces were ingested with `status=skipped` due to the field name issue:

| Trace ID | Player | Prop | Skip Reason |
|----------|--------|------|-------------|
| `sandbox-2569e47d-dd94` | Cunningham | threes | Missing `threes_mean` |
| `sandbox-a8cb09ef-828f` | Harden | ast | Missing `ast_mean` |
| `sandbox-fe2718ac-28d4` | Duren | reb | Missing `reb_mean` |

These are inert (no bet records attached, no CLV capture instructions). They inflate trace count but do not affect calibration. Dev note: consider a DB cleanup utility for skipped orphan traces, or filter them from calibration candidate queries by default.

---

## 5. Manual Actions Required

### JIT CLV Captures (4 tasks — must be created before 7:30 PM ET May 17)

All 4 should fire at **2026-05-17T19:30:00-04:00** (7:30 PM ET). Target: theScore Bet or any public odds source.

| Trace | Selection | Market | Descriptor |
|-------|-----------|--------|------------|
| `sandbox-80b0adf2-4013` | Cunningham Under 27.5 pts | player_prop:pts | `home_pts_under_27.5` |
| `sandbox-a93f3a51-83d2` | Cunningham Over 2.5 threes | player_prop:threes | `home_threes_over_2.5` |
| `sandbox-58ee2ffa-63a4` | Harden Over 6.5 ast | player_prop:ast | `away_ast_over_6.5` |
| `sandbox-354c32cd-8778` | Mobley Over 18.5 pts | player_prop:pts | `away_pts_over_18.5` |

Each capture writes to `var/inbox/closing_lines/<trace_id>.json` then runs `omega-ingest-closing-lines`.

---

## 6. Recommended Engine/Agent Improvements (Priority Order)

1. **Prop-type context key validation** — before calling `r.analyze()`, validate that the player_context contains the expected key for the given prop_type. A simple lookup dict (`{"pts": "pts_mean", "reb": "reb_mean", "ast": "ast_mean", "threes": "threes_mean"}`) prevents skip-trace pollution.
2. **VM auto-retry with backoff** — on "still starting" or "unavailable" responses, retry automatically (e.g. 3× at 10s intervals) before surfacing to user or entering QUALITATIVE MODE.
3. **Scheduled task supervision mode** — investigate whether `create_scheduled_task` can be called earlier in the session when the user is actively present, or add a fallback that writes pending JIT task specs to a file the user can approve in a follow-up turn.
4. **Orphan trace filtering** — exclude `status=skipped` traces from calibration candidate queries by default.
