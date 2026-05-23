# Session Log — 2026-05-23

**Session ID:** sess-20260523-nba1  
**Date:** 2026-05-23  
**Game:** OKC Thunder @ San Antonio Spurs — May 22, 2026  
**Final:** OKC 123, SA 108

---

## Outcome Grading — Omega Traces (May 22)

5 props graded | **4W 1L** (80%)

| Player | Stat | Side | Line | Actual | Engine Edge | Tier | Result |
|---|---|---|---|---|---|---|---|
| Chet Holmgren | PTS | UNDER | 14.5 | 14.0 | 21.75% | A | ✓ WIN |
| Chet Holmgren | REB | UNDER | 8.5 | 3.0 | 19.11% | A | ✓ WIN |
| Shai Gilgeous-Alexander | PTS | UNDER | 30.5 | 26.0 | 16.32% | A | ✓ WIN |
| Victor Wembanyama | PTS | OVER | 25.5 | 26.0 | 8.77% | A | ✓ WIN |
| Victor Wembanyama | REB | OVER | 13.5 | 4.0 | 21.89% | A | ✗ LOSS |

**Game result:** OKC won 123–108. SA Spurs -1.5 (trace: sandbox-9079ee10-6dd8, sim_win_prob=0.61) → **LOSS** (OKC covered by 15).

---

## User Bet Log — Personal Results

### Bet 1 — SGP 2-leg LOSS ($20.00 → $0.00)
| Leg | Line | Actual | Result |
|---|---|---|---|
| Wemby REB OVER 13.5 | 13.5 | 4 | ✗ LOSS |
| Wemby PTS OVER 24.5 | 24.5 | 26 | ✓ WIN |

Note: Omega trace analyzed Wemby pts at OVER 25.5 (line discrepancy vs user's 24.5 — drift warning triggered at ingest).

---

### Bet 2 — SGP 3-leg LOSS ($25.00 → $0.00, 50% profit boost applied)
| Leg | Line | Actual | Result |
|---|---|---|---|
| Dylan Harper PTS OVER 10.5 | 10.5 | 6 | ✗ LOSS |
| Stephon Castle PTS OVER 14.5 | 14.5 | 14 | ✗ LOSS (missed by 0.5) |
| SGA Assists OVER 7.5 | 7.5 | 12 | ✓ WIN |

**No Omega traces** for Harper pts, Castle pts, or SGA assists in this game. Bets taken without engine coverage.

---

### Bet 3 — SGP 2-leg WIN ($24.12 → $58.57, net +$34.45)
| Leg | Line | Actual | Result |
|---|---|---|---|
| Chet Holmgren REB UNDER 8.5 | 8.5 | 3 | ✓ WIN |
| Chet Holmgren PTS UNDER 14.5 | 14.5 | 14 | ✓ WIN |

Matched Omega traces: sandbox-a8c98f20-4192 (reb), sandbox-92fda8f7-7ae6 (pts). Both Tier A.

---

### Bet 4 — SGP 2-leg WIN ($75.00 → $132.38, net +$57.38)
| Leg | Line | Actual | Result |
|---|---|---|---|
| SGA Assists OVER 5.5 | 5.5 | 12 | ✓ WIN |
| Stephon Castle Assists OVER 5.5 | 5.5 | 7 | ✓ WIN |

**No Omega traces** for SGA assists or Castle assists. These were user-identified bets, not engine-sourced.

---

## Session P&L Summary

| Bet | Stake | Payout | Net |
|---|---|---|---|
| Bet 1 (SGP Loss) | $20.00 | $0.00 | -$20.00 |
| Bet 2 (SGP Loss) | $25.00 | $0.00 | -$25.00 |
| Bet 3 (SGP Win) | $24.12 | $58.57 | +$34.45 |
| Bet 4 (SGP Win) | $75.00 | $132.38 | +$57.38 |
| **Total** | **$144.12** | **$190.95** | **+$46.83** |

---

## DB Changes This Session

- 6 traces ingested from inbox → DB (`sess-20260522-nba1`)
- 4 bet_records attached (Wemby reb, Wemby pts, Chet pts, Chet reb)
- 5 prop_outcomes attached (source: manual:betmgm_slip_20260522 + manual:espn_boxscore_20260522)
- 1 game outcome attached (OKC 123 SA 108, sandbox-9079ee10-6dd8)
- Box score data sourced from ESPN (event_id: 401873199)

---

## Bugs Found This Session

### BUG-SS-20260523-1: input_snapshot missing game identity on ALL 6 inbox traces

**Severity:** High — blocked ingest for prop traces with bet_records (BUG-4 recurrence)

**Affected traces:** All 6 from sess-20260522-nba1

**Pattern:** Engine writes `home_team`, `away_team`, `game_date`, `player_name`, `prop_type`, `line` into `result` block only. `input_snapshot` is populated with only `player_context` and `game_context`. The ingest validator rejects prop traces with bet_records when these fields are absent from `input_snapshot`.

**Fix applied (session hotfix):** Copied game identity fields from `result` → `input_snapshot` before ingest via `prepare_inbox_traces.py` script.

**Root cause:** Engine's `analyze()` call or trace export is not propagating top-level request fields into `input_snapshot`. The issue is in how traces are constructed at the session level — the `player_name`, `line`, `home_team`, `away_team`, `game_date` fields go into the result but the input_snapshot only captures `player_context`/`game_context` dicts.

**Required fix:** In the engine's trace export (likely `omega/trace/persistable.py` or the analyze() wrapper), ensure `input_snapshot` always includes: `player_name`, `prop_type`, `line`, `home_team`, `away_team`, `game_date` for prop traces. These are top-level request fields and must be preserved.

**Reference:** session_bugs_20260519.md BUG-2 and BUG-4 — same pattern, recurrence indicates the fix was not applied upstream.

---

### BUG-SS-20260523-2: fetch_box_score() and FinalGame object API mismatch

**Severity:** Low — ESPN integration functions have inconsistent call signatures

**Details:**
- `FinalGame` dataclass uses `event_id` attribute, not `game_id` (misleading if using old code)
- `fetch_box_score(event_id)` requires `(league, event_id)` — positional arg order not obvious from name
- `parse_box_score(payload)` requires `(payload, league)` — league arg not in old callers

**Fix:** Document correct call signatures; update any scripts still using old positional args.

---

## Untracked Bets (no Omega trace coverage)

The following markets were bet by user on May 22 with no corresponding engine trace:
- Dylan Harper pts (10.5) → 6 actual
- Stephon Castle pts (14.5) → 14 actual  
- SGA assists (5.5, 7.5) → 12 actual
- Stephon Castle assists (5.5) → 7 actual

**Recommendation:** Add these players to the pre-game prop scan for future SA/OKC matchups. Castle assists and SGA assists were +EV (actual crushed the lines); engine should have coverage on these.
